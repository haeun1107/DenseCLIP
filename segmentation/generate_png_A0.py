#!/usr/bin/env python3
"""
Generate PNG segmentation masks from a DenseCLIP A0 checkpoint on ISIC-style datasets.

- Keeps the SAME relative path as input image.
  e.g., data/ISIC/ISIC2018_Task1_Training_Input/ISIC_000007.png
     -> out_dir/ISIC2018_Task1_Training_Input/ISIC_000007_segmentation.png

Default assumes ISIC 2018 Task1 (binary: background=0, lesion=1) and writes 8-bit masks
with values {0,255}. For multi-class, omit --binary to save {0..C-1} as 8-bit labels.
"""
import argparse, os, os.path as osp, numpy as np
import mmcv, torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
import denseclip  # noqa

def parse_args():
    p = argparse.ArgumentParser("DenseCLIP A0 → PNG mask writer (ISIC)")
    p.add_argument('config', help='config .py (ISIC dataset test cfg)')
    p.add_argument('checkpoint', help='A0 checkpoint .pth')
    p.add_argument('--out-dir', required=True, help='root dir to save PNG masks')
    p.add_argument('--suffix', default='_segmentation.png',
                   help="output filename suffix (default: '_segmentation.png')")
    p.add_argument('--binary', action='store_true',
                   help='Map class {0,1} → {0,255} and save 8-bit PNG (ISIC Task1).')
    p.add_argument('--class-names', nargs='*', default=None,
                   help="Optional override for model.class_names (e.g., background lesion)")
    p.add_argument('--workers', type=int, default=None, help='dataloader workers; default from cfg')
    return p.parse_args()

# ---------- helpers ----------
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _as_label_map(pred):
    """(C,H,W)/(H,W,C) or (H,W) → (H,W) label map."""
    if isinstance(pred, tuple) and len(pred): pred = pred[0]
    if isinstance(pred, dict):
        for k in ('seg_pred','sem_seg','pred','segmentation'):
            if k in pred: pred = pred[k]; break
    a = np.asarray(pred)
    if a.ndim == 2:
        return a.astype(np.int32)
    if a.ndim == 3:
        # assume channels-first if smaller first dim
        if a.shape[0] < a.shape[-1] and a.shape[0] <= 512:
            return a.argmax(axis=0).astype(np.int32)
        return a.argmax(axis=-1).astype(np.int32)
    return np.squeeze(a).astype(np.int32)

def _rel_img_path(dataset, idx):
    """Return image relative path used by dataset (with extension)."""
    info = None
    for key in ('img_infos','data_infos'):
        if hasattr(dataset, key):
            arr = getattr(dataset, key)
            if 0 <= idx < len(arr): info = arr[idx]; break
    rel = None
    if isinstance(info, dict):
        rel = (info.get('img_info', {}) or {}).get('filename') \
           or info.get('filename') or info.get('file_name')
    return rel.replace('\\','/') if rel else f'{idx:06d}.png'

def _out_path(out_root, rel_img, suffix):
    """Replace filename (without ext) to add suffix, keep directories."""
    d = osp.dirname(rel_img)
    stem = osp.splitext(osp.basename(rel_img))[0]
    # handle .png, .jpg, .jpeg etc.; suffix should include .png
    return osp.join(out_root, d, stem + suffix)

# ---------- main ----------
def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # dataset
    dataset = build_dataset(cfg.data.test)

    # DenseCLIP 계열은 build 시 class_names 필요할 수 있음
    if args.class_names is not None:
        cfg.model['class_names'] = list(args.class_names)
    elif 'class_names' not in cfg.model or cfg.model['class_names'] is None:
        # ISIC Task1 기본값(배경, 병변)
        default_names = ['background', 'lesion']
        cfg.model['class_names'] = list(getattr(dataset, 'CLASSES', default_names))

    workers = args.workers if args.workers is not None else getattr(cfg.data, 'workers_per_gpu', 2)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=workers,
                                   dist=False, shuffle=False)

    # model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    # inference (mmseg single_gpu_test uses rescale=True)
    results = single_gpu_test(model, data_loader, show=False)

    saved = 0
    for i, pred in enumerate(results):
        lab = _as_label_map(pred)  # (H,W) int
        rel = _rel_img_path(dataset, i)
        out_path = _out_path(args.out_dir, rel, args.suffix)
        _ensure_dir(osp.dirname(out_path))

        if args.binary:
            # {0,1} → {0,255}, uint8 single-channel
            img = (lab > 0).astype(np.uint8) * 255
            mmcv.imwrite(img, out_path)  # writes PNG by extension
        else:
            # multi-class 0..C-1 as 8-bit
            img = lab.astype(np.uint8)
            mmcv.imwrite(img, out_path)

        saved += 1

    print(f"[PNG] Saved {saved} masks → {osp.abspath(args.out_dir)}")

if __name__ == '__main__':
    main()
