#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synapse 등 NPZ/H5 기반 테스트셋에서 'image + label(+ pred/probs/logits)' 형태의 NPZ를 생성합니다.

- 입력: cfg.data.test 으로 정의된 데이터셋(슬라이스 단위; NPZ/H5/일반 이미지 경로 모두 호환)
- 출력: out_dir/*.npz
  {
    "image": float32 (H,W),
    "label": int (H,W),                # GT (있을 경우)
    "pred":  int32 (H,W),              # argmax 예측 맵
    ["pred_probs" or "pred_logits": (C,H,W)]  # 옵션
  }

주요 포인트
- 모델 빌드 전에 cfg.model.class_names 를 주입 (dataset.CLASSES → 없으면 Synapse 8클래스 기본값)
- split에 'data/..../file.npz' 같은 풀경로나 basename이 섞여 있어도 안전하게 처리
- (C,H,W)/(H,W,C)/(H,W) 다양한 출력 형식에 대응
- 경로 prefix 중복 join 방지(_maybe_join_prefix)

사용 예시
---------
python segmentation/generate_npz_synapse.py \
  segmentation/configs/denseclip_fpn_res50_512x512_80k_synapse_new.py \
  work_dirs/denseclip_fpn_res50_512x512_80k_synapse_new/iter_57600_best.pth \
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

import denseclip  # noqa: F401  # DenseCLIP 모듈 등록용 임포트

# ------------------------------ 기본 클래스 이름 -------------------------------

DEFAULT_SYN_CLASSES = (
    "background","aorta","gallbladder","left kidney","right kidney",
    "liver","pancreas","spleen","stomach"
)

# ------------------------------ 유틸 함수 --------------------------------------

def _npz_save(out_path, image_2d, label_2d=None, pred_2d=None, extra=None, dtype_label=np.int32):
    """NPZ로 저장: image(float32), label(선택), pred(선택), probs/logits(선택)."""
    image_2d = np.asarray(image_2d).astype(np.float32)  # (H,W)
    save_dict = dict(image=image_2d)
    if label_2d is not None:
        label_2d = np.asarray(label_2d)
        save_dict["label"] = label_2d.astype(dtype_label)
    if pred_2d is not None:
        pred_2d = np.asarray(pred_2d)
        save_dict["pred"] = pred_2d.astype(np.int32)
    if extra:
        save_dict.update(extra)
    np.savez_compressed(out_path, **save_dict)

def _to_gray01(img):
    """(H,W) or (H,W,3/4) → (H,W) float32, 0..255면 0..1 스케일."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.dtype != np.float32:
            img = mmcv.bgr2gray(img.astype(np.uint8))
        else:
            img = img.mean(-1)
    img = img.astype(np.float32)
    if img.max() > 1.5:  # 0..255로 판단되면 normalize
        img = img / 255.0
    return img

def _resolve_filename(dataset, idx):
    """
    데이터셋에서 i번째 샘플의 원본 파일 경로를 복원.
    - CustomDataset류: info['img_info']['filename'] 또는 info['filename']
    - meta에는 slice_idx/z_index 가능하면 함께 기록
    """
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

    # 안전한 prefix 결합
    prefix = getattr(dataset, 'img_prefix', None) or getattr(dataset, 'img_dir', None)
    if filename:
        if not os.path.isabs(filename) and prefix:
            fn = os.path.normpath(filename)
            pf = os.path.normpath(prefix)
            try:
                # filename이 prefix 하위면 그대로, 아니면 join
                common1 = os.path.commonpath([os.path.join(pf, '')])
                common2 = os.path.commonpath([os.path.join(pf, ''), fn])
                filename = fn if common1 == common2 else os.path.normpath(os.path.join(pf, filename))
            except Exception:
                filename = os.path.normpath(os.path.join(pf, filename))
    return filename, meta

def _maybe_join_prefix(path, prefix):
    """path가 이미 prefix 하위면 다시 join하지 않도록 방어."""
    if not path or not prefix or os.path.isabs(path):
        return path
    npath = os.path.normpath(path)
    nprefix = os.path.normpath(prefix)
    try:
        # npath가 이미 nprefix 하위면 그대로 반환
        if os.path.commonpath([nprefix, npath]) == nprefix:
            return npath
    except Exception:
        pass
    if npath.startswith(nprefix + os.sep) or npath == nprefix:
        return npath
    return os.path.normpath(os.path.join(nprefix, npath))

def _load_input_image_2d(path, meta):
    """
    해당 샘플의 원본 2D 이미지를 재로딩:
      - .npz : ['image'] (1xHxW / HxWx1 모두 처리)
      - .h5  : ['image'][slice_idx] (z_index/slice### 인식)
      - 그 외 : mmcv.imread(gray)
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
    모델 예측을 표준 형태로 정규화.
    반환: (pred_label_2d, logits_or_probs_3d or None)
    - (H,W) → 그대로 segmap
    - (C,H,W) / (H,W,C) → argmax + (C,H,W) 로짓 반환
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
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:  # (C,H,W)로 간주
            label = arr.argmax(axis=0)
            logits = arr.astype(np.float32)
        else:  # (H,W,C)
            label = arr.argmax(axis=-1)
            logits = np.transpose(arr, (2,0,1)).astype(np.float32)
        return label.astype(np.int32), logits

    label = np.squeeze(arr)
    return label.astype(np.int32), None

def _resize_like(img2d, target_hw):
    th, tw = target_hw
    if img2d.shape != (th, tw):
        return mmcv.imresize(img2d, (tw, th), interpolation='bilinear')
    return img2d

# ------------------------------- main ----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser("Generate NPZ (image+label+pred) pseudo labels")
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--launcher', choices=['none','pytorch','slurm','mpi'], default='none')
    ap.add_argument('--gpu-collect', action='store_true')
    ap.add_argument('--tmpdir', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--dtype', choices=['uint8','uint16','int32'], default='uint8',
                    help='label 저장 dtype (GT 있을 경우)')
    ap.add_argument('--save-logits', action='store_true',
                    help='raw logits를 "pred_logits"(C,H,W)로 저장')
    ap.add_argument('--save-probs', action='store_true',
                    help='softmax probs를 "pred_probs"(C,H,W)로 저장 (save-logits보다 우선)')
    ap.add_argument('--limit', type=int, default=0, help='>0 이면 처음 N개만 저장')
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

    # 1) Dataset/Dataloader
    dataset = build_dataset(cfg.data.test)

    # 2) 필수 인자 주입: class_names (dataset.CLASSES → 없으면 기본값)
    if not isinstance(cfg.model, dict):
        cfg.model = dict(cfg.model)
    cls_from_ds = getattr(dataset, 'CLASSES', None)
    classes = list(cls_from_ds) if cls_from_ds else list(DEFAULT_SYN_CLASSES)
    classes = [c.strip() for c in classes]  # 공백 정리
    if not cfg.model.get('class_names'):
        cfg.model['class_names'] = classes
    print(f"[INFO] Using class_names ({len(cfg.model['class_names'])}): {cfg.model['class_names']}")
    try:
        print(f"[DEBUG] RANK 0 - class_names length: {len(cfg.model['class_names'])}")
        print(f"[DEBUG] RANK 0 - class_names[0]: {cfg.model['class_names'][0]}")
    except Exception:
        pass
    assert cfg.model.get('class_names'), "class_names must be set in cfg.model for DenseCLIP."

    # 3) Dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=getattr(cfg.data, 'workers_per_gpu', 2),
        dist=distributed, shuffle=False
    )

    # 4) 모델 빌드(주입 이후)
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 메타데이터 보완
    if not getattr(model, 'CLASSES', None):
        model.CLASSES = getattr(dataset, 'CLASSES', None)
    if not getattr(model, 'PALETTE', None):
        model.PALETTE = getattr(dataset, 'PALETTE', None)

    torch.cuda.empty_cache()

    # 5) 추론
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

    # 6) 저장
    os.makedirs(args.out_dir, exist_ok=True)
    dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'int32': np.int32}
    to_dtype = dtype_map[args.dtype]

    # 파일명 복원 리스트 (원본 NPZ/H5 경로 재로딩에 사용)
    file_names = []
    lst = getattr(dataset, "img_infos", None) or getattr(dataset, "data_infos", None) or []
    for info in lst:
        if isinstance(info, dict):
            img_info = info.get("img_info", info)
            fn = img_info.get("filename") or img_info.get("file_name") or info.get("filename")
            file_names.append(fn)
        else:
            file_names.append(None)

    saved = 0
    for i, pred in enumerate(results):
        if args.limit and saved >= args.limit:
            break

        pred_map, maybe_logits = _extract_pred_arrays(pred)

        # 원본 이미지/라벨 재로딩
        src_path, meta = _resolve_filename(dataset, i)
        if not src_path and i < len(file_names):
            src_path = file_names[i]

        # dataset.img_prefix 고려한 안전한 경로 합치기 (중복 join 방지)
        prefix = getattr(dataset, 'img_prefix', None) or getattr(dataset, 'img_dir', None)
        if src_path:
            src_path = _maybe_join_prefix(src_path, prefix)

        image = None
        label = None
        if src_path and osp.splitext(src_path)[1].lower() == ".npz":
            try:
                data_npz = np.load(src_path, allow_pickle=False)
                image = data_npz.get("image", None)
                label = data_npz.get("label", None)
                if image is not None:
                    if image.ndim == 3 and image.shape[0] == 1:
                        image = image[0]
                    if image.ndim == 3 and image.shape[-1] == 1:
                        image = image[..., 0]
                    image = image.astype(np.float32)
            except Exception as e:
                print(f"[WARN] npz load failed: {src_path} ({e})")
        else:
            image = _load_input_image_2d(src_path, meta)

        if image is None:
            # 안전한 fallback
            H, W = pred_map.shape
            image = np.zeros((H, W), dtype=np.float32)

        # 크기 맞추기
        if image.shape != pred_map.shape:
            image = _resize_like(image, pred_map.shape)
        if label is not None and label.shape != pred_map.shape:
            label = mmcv.imresize(label, (pred_map.shape[1], pred_map.shape[0]), interpolation="nearest")

        # 출력 파일명
        base = osp.splitext(osp.basename(src_path))[0] if src_path else f"{i:06d}"
        out_path = osp.join(args.out_dir, f"{base}.npz")

        # 확장 저장 (probs/logits)
        extra = {}
        if args.save_probs and maybe_logits is not None:
            x = maybe_logits.astype(np.float32)
            x = x - x.max(axis=0, keepdims=True)
            x = np.exp(x)
            probs = x / (x.sum(axis=0, keepdims=True) + 1e-8)
            extra['pred_probs'] = probs.astype(np.float32)
        elif args.save_logits and maybe_logits is not None:
            extra['pred_logits'] = maybe_logits.astype(np.float32)

        _npz_save(out_path, image, label, pred_map, extra=extra, dtype_label=to_dtype)
        saved += 1

    print(f"[NPZ] saved {saved} files → {osp.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()


# python segmentation/generate_npz_synapse.py \
#   segmentation/configs/denseclip_fpn_res50_512x512_80k_synapse_new.py \
#   work_dirs/denseclip_fpn_res50_512x512_80k_synapse_new/iter_57600_best.pth \
#   --out-dir data/Synapse/pseudo_synapse \
#   --save-probs \
#   --dtype uint8
