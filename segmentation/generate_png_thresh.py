#!/usr/bin/env python3
"""
Generate PNG segmentation masks from a DenseCLIP A0 checkpoint on ISIC-style datasets.

- Keeps the SAME relative path as input image.
  e.g., data/ISIC/ISIC2018_Task1_Training_Input/ISIC_000007.png
     -> out_dir/ISIC2018_Task1_Training_Input/ISIC_000007_segmentation.png

Supports optional confidence thresholding:
  --threshold 0.5 --bg-index 0
will set any pixel whose max-class confidence < 0.5 to background (class 0).

Default assumes ISIC 2018 Task1 (binary: background=0, lesion=1) and writes 8-bit masks
with values {0,255} if --binary is given. Without --binary, saves raw class ids 0..C-1.
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
    p = argparse.ArgumentParser("DenseCLIP A0 → PNG mask writer (ISIC) w/ optional threshold")
    p.add_argument('config', help='config .py (ISIC dataset test cfg)')
    p.add_argument('checkpoint', help='A0 checkpoint .pth')
    p.add_argument('--out-dir', required=True, help='root dir to save PNG masks')
    p.add_argument('--suffix', default='_segmentation.png',
                   help="output filename suffix (default: '_segmentation.png')")
    p.add_argument('--binary', action='store_true',
                   help='Map class {0,1} → {0,255} and save 8-bit PNG (ISIC Task1).')
    p.add_argument('--class-names', nargs='*', default=None,
                   help="Optional override for model.class_names (e.g., background lesion)")
    p.add_argument('--workers', type=int, default=None,
                   help='dataloader workers; default from cfg')

    # NEW: confidence gating
    p.add_argument('--threshold', type=float, default=None,
                   help='If set: pixels with max prob < threshold become bg-index.')
    p.add_argument('--bg-index', type=int, default=0,
                   help='Background class index to assign to low-confidence pixels (default: 0).')

    return p.parse_args()


# ---------- helpers ----------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _compute_label_and_conf(pred):
    """
    pred can be:
      - (H, W) int map          -> label only, no confidence
      - (C, H, W) score/prob    -> compute argmax + confidence
      - (H, W, C) score/prob    -> same, but channel-last
      - tuple/dict wrappers     -> unwrap common keys
    Returns:
      label2d: (H, W) int32
      conf2d:  (H, W) float32 or None if we couldn't infer confidence
    """
    # unwrap common container types
    if isinstance(pred, tuple) and len(pred):
        pred = pred[0]
    if isinstance(pred, dict):
        for k in ('seg_pred', 'sem_seg', 'pred', 'segmentation'):
            if k in pred:
                pred = pred[k]
                break

    a = np.asarray(pred)

    # Case 1: already label map (H,W)
    if a.ndim == 2:
        label2d = a.astype(np.int32)
        conf2d = None  # we don't know per-pixel confidence here
        return label2d, conf2d

    # Case 2: (C,H,W) or (H,W,C) scores
    if a.ndim == 3:
        # Heuristic: if first dim looks like #classes
        # (usually <=512 and definitely < spatial dims).
        if a.shape[0] < a.shape[-1] and a.shape[0] <= 512:
            # assume (C,H,W)
            # we'll softmax along C to get probs
            logits = a.astype(np.float32)
            # softmax over channel dim 0
            max_logits = np.max(logits, axis=0, keepdims=True)  # (1,H,W)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # (C,H,W)

            label2d = np.argmax(probs, axis=0).astype(np.int32)  # (H,W)
            conf2d = np.max(probs, axis=0).astype(np.float32)    # (H,W)
            return label2d, conf2d
        else:
            # assume (H,W,C)
            logits = a.astype(np.float32)
            max_logits = np.max(logits, axis=-1, keepdims=True)  # (H,W,1)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (H,W,C)

            label2d = np.argmax(probs, axis=-1).astype(np.int32)  # (H,W)
            conf2d = np.max(probs, axis=-1).astype(np.float32)    # (H,W)
            return label2d, conf2d

    # Fallback: squeeze and treat as label map
    squeezed = np.squeeze(a)
    if squeezed.ndim == 2:
        return squeezed.astype(np.int32), None

    # Last resort: argmax on last dim
    label2d = np.argmax(squeezed, axis=-1).astype(np.int32)
    conf2d = np.max(squeezed, axis=-1).astype(np.float32)
    return label2d, conf2d


def _apply_threshold(label2d, conf2d, threshold, bg_index):
    """
    If threshold is not None and we have conf2d,
    any pixel with conf < threshold becomes bg_index.
    Returns new label2d.
    """
    if threshold is None:
        return label2d
    if conf2d is None:
        # we don't have confidence info, so nothing to apply
        return label2d

    out = label2d.copy()
    low_conf_mask = conf2d < threshold
    out[low_conf_mask] = bg_index
    return out


def _rel_img_path(dataset, idx):
    """Return image relative path used by dataset (with extension)."""
    info = None
    for key in ('img_infos', 'data_infos'):
        if hasattr(dataset, key):
            arr = getattr(dataset, key)
            if 0 <= idx < len(arr):
                info = arr[idx]
                break
    rel = None
    if isinstance(info, dict):
        rel = (info.get('img_info', {}) or {}).get('filename') \
              or info.get('filename') \
              or info.get('file_name')
    return rel.replace('\\', '/') if rel else f'{idx:06d}.png'


def _out_path(out_root, rel_img, suffix):
    """Replace filename (without ext) to add suffix, keep directories."""
    d = osp.dirname(rel_img)
    stem = osp.splitext(osp.basename(rel_img))[0]
    # suffix should already include .png or whatever extension you want
    return osp.join(out_root, d, stem + suffix)


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    # load cfg
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # dataset
    dataset = build_dataset(cfg.data.test)

    # class names (DenseCLIP sometimes needs this for text branch)
    if args.class_names is not None:
        cfg.model['class_names'] = list(args.class_names)
    elif 'class_names' not in cfg.model or cfg.model['class_names'] is None:
        default_names = ['background', 'lesion']
        cfg.model['class_names'] = list(getattr(dataset, 'CLASSES', default_names))

    workers = args.workers if args.workers is not None \
        else getattr(cfg.data, 'workers_per_gpu', 2)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False
    )

    # model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    # inference
    results = single_gpu_test(model, data_loader, show=False)

    saved = 0
    for i, pred in enumerate(results):
        # 1) get label + confidence
        label2d, conf2d = _compute_label_and_conf(pred)

        # 2) apply threshold if requested
        label2d_thr = _apply_threshold(
            label2d,
            conf2d,
            threshold=args.threshold,
            bg_index=args.bg_index
        )

        # 3) resolve output path
        rel = _rel_img_path(dataset, i)
        out_path = _out_path(args.out_dir, rel, args.suffix)
        _ensure_dir(osp.dirname(out_path))

        # 4) write PNG
        if args.binary:
            # binary mask → {0,255}
            img = (label2d_thr > 0).astype(np.uint8) * 255
            mmcv.imwrite(img, out_path)
        else:
            # raw class ids 0..C-1
            img = label2d_thr.astype(np.uint8)
            mmcv.imwrite(img, out_path)

        saved += 1

    print(f"[PNG] Saved {saved} masks → {osp.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
