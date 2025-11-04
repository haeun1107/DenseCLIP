#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate NPZ pseudo-labels that contain BOTH image and label.

- Reads the same input image used by the dataset (npz/h5/img file).
- Saves: {"image": float32 (H,W), "label": int (H,W), ["pred_probs" or "pred_logits": (C,H,W)]}
- Robust to DenseCLIP outputs: (H,W) or (C,H,W)/(H,W,C).
- Injects cfg.model.class_names from dataset.CLASSES before building DenseCLIP.

Usage
-----
python segmentation/generate_npz_synapse.py \
  segmentation/configs/denseclip_fpn_res50_512x512_80k_synapse_new.py \
  work_dirs/denseclip_fpn_res50_512x512_80k_synapse_new/iter_57600.pth \
  --out-dir data/Synapse/pseudo_synapse \
  --save-probs \
  --dtype uint8
"""
import os
import os.path as osp
import argparse
import re
import numpy as np
import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, get_dist_info, init_dist
from mmseg.apis import single_gpu_test, multi_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor

import denseclip  # noqa: F401  # DenseCLIP modules registration

# ------------------------------ helpers --------------------------------------

DEFAULT_SYN_CLASSES = (
    "background","aorta","gallbladder","left kidney","right kidney",
    "liver","pancreas","spleen","stomach"
)

def _npz_save(out_path, image_2d, label_2d, extra=None, dtype_label=np.int32):
    image_2d = np.asarray(image_2d).astype(np.float32)       # (H,W)
    label_2d = np.asarray(label_2d).astype(dtype_label)       # (H,W)
    save_dict = dict(image=image_2d, label=label_2d)
    if extra:
        save_dict.update(extra)
    np.savez_compressed(out_path, **save_dict)

def _to_gray01(img):
    """(H,W) or (H,W,3/4) -> (H,W) float32, scale to [0..1] if 0..255."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.dtype != np.float32:
            img = mmcv.bgr2gray(img.astype(np.uint8))
        else:
            img = img.mean(-1)
    img = img.astype(np.float32)
    if img.max() > 1.5:  # assume 0..255
        img = img / 255.0
    return img

def _resolve_filename(dataset, idx):
    info = None
    for key in ('img_infos', 'data_infos'):
        if hasattr(dataset, key):
            lst = getattr(dataset, key)
            if 0 <= idx < len(lst):
                info = lst[idx]
                break
    if not isinstance(info, dict):
        return None, {}

    img_info = info.get('img_info', info) or {}
    filename = (img_info.get('filename')
                or img_info.get('file_name')
                or info.get('filename'))

    meta = dict(
        z_index=img_info.get('z_index') or info.get('z_index'),
        slice_idx=img_info.get('slice_idx') or info.get('slice_idx')
    )

    # --- 안전한 prefix 결합 ---
    prefix = getattr(dataset, 'img_prefix', None) or getattr(dataset, 'img_dir', None)
    if filename:
        # 이미 절대경로면 건드리지 않음
        if not os.path.isabs(filename) and prefix:
            fn = os.path.normpath(filename)
            pf = os.path.normpath(prefix)
            try:
                # filename이 이미 prefix 하위에 있으면 그대로 둠
                if os.path.commonpath([os.path.join(pf, '')]) == os.path.commonpath([os.path.join(pf, ''), fn]):
                    # 예: filename이 'data/Synapse/train_npz/case0001.npz' 같은 경우
                    filename = fn
                else:
                    filename = os.path.normpath(os.path.join(pf, filename))
            except Exception:
                # 공통 경로 계산 실패하면 보수적으로 join
                filename = os.path.normpath(os.path.join(pf, filename))
    return filename, meta

def _load_input_image_2d(path, meta):
    """
    Load 2D image used for this sample:
      - .npz : ['image'] (handles (1,H,W)/(H,W,1))
      - .h5  : ['image'][slice_idx/z_index] (or parse 'slice###' in name)
      - else : mmcv.imread(gray)
    """
    if not path:
        return None
    ext = osp.splitext(path)[1].lower()
    try:
        if ext == '.npz':
            data = np.load(path, allow_pickle=False)
            img = data.get('image')
            if img is None:
                raise KeyError(f"'image' not found in {path}")
            if img.ndim == 3 and img.shape[0] == 1:  # (1,H,W)
                img = img[0]
            if img.ndim == 3 and img.shape[-1] == 1: # (H,W,1)
                img = img[..., 0]
            return _to_gray01(img)

        if ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(path, 'r') as f:
                if 'image' not in f:
                    raise KeyError(f"'image' dataset not in {path}")
                z = meta.get('slice_idx')
                if z is None:
                    z = meta.get('z_index')
                if z is None:
                    m = re.search(r'slice(\d+)', osp.basename(path))
                    z = int(m.group(1)) if m else 0
                img = f['image'][int(z)]
                return _to_gray01(img)

        if osp.exists(path):
            img = mmcv.imread(path, flag='grayscale')
            return _to_gray01(img)
    except Exception as e:
        print(f"[WARN] failed to load input image from {path}: {e}")
    return None

def _extract_pred_arrays(pred_item):
    """
    Normalize one prediction item:
      returns (label_2d, logits_or_probs_3d or None)
    """
    if isinstance(pred_item, tuple) and len(pred_item) > 0:
        pred_item = pred_item[0]
    if isinstance(pred_item, dict):
        for k in ('seg_pred','sem_seg','pred','segmentation'):
            if k in pred_item:
                pred_item = pred_item[k]
                break
    arr = np.asarray(pred_item)

    if arr.ndim == 2:
        return arr.astype(np.int32), None

    if arr.ndim == 3:
        # treat (C,H,W) when 'C' is the smallest dim and small enough
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:
            label = arr.argmax(axis=0)
            logits = arr.astype(np.float32)
        else:  # (H,W,C)
            label = arr.argmax(axis=-1)
            logits = np.transpose(arr, (2,0,1)).astype(np.float32)
        return label.astype(np.int32), logits

    # fallback
    label = np.squeeze(arr)
    return label.astype(np.int32), None

def _resize_image_like(img2d, target_hw):
    th, tw = target_hw
    if img2d.shape != (th, tw):
        return mmcv.imresize(img2d, (tw, th), interpolation='bilinear')
    return img2d

# ------------------------------- main ----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser("Generate NPZ (image + label) pseudo labels")
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--launcher', choices=['none','pytorch','slurm','mpi'], default='none')
    ap.add_argument('--gpu-collect', action='store_true')
    ap.add_argument('--tmpdir', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--dtype', choices=['uint8','uint16','int32'], default='uint8',
                    help='dtype for label saving')
    ap.add_argument('--save-logits', action='store_true',
                    help='also save raw logits as \"pred_logits\" (C,H,W)')
    ap.add_argument('--save-probs', action='store_true',
                    help='save softmax probs as \"pred_probs\" (C,H,W); overrides save-logits')
    ap.add_argument('--limit', type=int, default=0, help='>0 to save only first N samples')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # 1) dataset 먼저 만들고
    dataset = build_dataset(cfg.data.test)

    # 2) DenseCLIP 필수 인자 주입: class_names
    if not isinstance(cfg.model, dict):
        cfg.model = dict(cfg.model)
    if not cfg.model.get('class_names'):
        classes = list(getattr(dataset, 'CLASSES', DEFAULT_SYN_CLASSES))
        cfg.model['class_names'] = classes
        print(f"[INFO] Injected class_names ({len(classes)}) into cfg.model.")

    # dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=getattr(cfg.data, 'workers_per_gpu', 2),
        dist=distributed, shuffle=False
    )

    # 3) build model AFTER class_names injection
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # attach classes/palette if missing
    if not getattr(model, 'CLASSES', None):
        model.CLASSES = getattr(dataset, 'CLASSES', None)
    if not getattr(model, 'PALETTE', None):
        model.PALETTE = getattr(dataset, 'PALETTE', None)

    torch.cuda.empty_cache()

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(model, data_loader, show=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, False)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'int32': np.int32}
    to_dtype = dtype_map[args.dtype]

    saved = 0
    for i, pred in enumerate(results):
        if args.limit and saved >= args.limit:
            break

        # (H,W) label + optional (C,H,W) logits
        label2d, logits = _extract_pred_arrays(pred)
        H, W = label2d.shape

        # try to reload the exact input image used by dataset
        src_path, meta = _resolve_filename(dataset, i)
        img2d = _load_input_image_2d(src_path, meta)
        if img2d is None:
            img2d = np.zeros((H, W), dtype=np.float32)  # safe fallback

        # match size to prediction
        if img2d.shape != (H, W):
            img2d = _resize_image_like(img2d, (H, W))

        # output name
        base = osp.splitext(osp.basename(src_path))[0] if src_path else f"{i:06d}"
        out_path = osp.join(args.out_dir, f"{base}.npz")

        # optional extras
        extra = {}
        if args.save_probs and logits is not None:
            x = logits.astype(np.float32)
            x = x - x.max(axis=0, keepdims=True)
            x = np.exp(x)
            probs = x / (x.sum(axis=0, keepdims=True) + 1e-8)
            extra['pred_probs'] = probs.astype(np.float32)
        elif args.save_logits and logits is not None:
            extra['pred_logits'] = logits.astype(np.float32)

        _npz_save(out_path, img2d, label2d, extra=extra, dtype_label=to_dtype)
        saved += 1

    print(f"[NPZ] saved {saved} files → {osp.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()
