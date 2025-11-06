#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate PNG pseudo segmentation masks from a DenseCLIP A0 checkpoint on ISIC-style datasets.

- Keeps the SAME relative path as input image.
  e.g., data/ISIC/ISIC2018_Task1_Training_Input/ISIC_000007.jpg
     -> out_dir/ISIC2018_Task1_Training_Input/ISIC_000007_segmentation.png

Label convention for TRAINING (default, no --vis-binary):
  - 0   : background
  - 1   : lesion
  - 255 : ignore  (confidence < threshold, if confidence is available)

If you pass --vis-binary, it is ONLY for visualization:
  - 0   -> 0   (black, background)
  - 1   -> 255 (white, lesion)
  - 255 -> 128 (gray, ignore)
"""

import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
import denseclip  # noqa  (register DenseCLIP)

IGNORE_INDEX = 255


def parse_args():
    p = argparse.ArgumentParser("DenseCLIP A0 → PNG pseudo mask writer (ISIC, 0/1/255 + ignore)")
    p.add_argument('config', help='config .py (ISIC dataset test cfg)')
    p.add_argument('checkpoint', help='A0 checkpoint .pth')
    p.add_argument('--out-dir', required=True, help='root dir to save PNG masks')
    p.add_argument('--suffix', default='_segmentation.png',
                   help="output filename suffix (default: '_segmentation.png')")
    p.add_argument('--class-names', nargs='*', default=None,
                   help="Optional override for model.class_names (e.g., background lesion)")
    p.add_argument('--workers', type=int, default=None,
                   help='dataloader workers; default from cfg')

    # confidence gating → low-conf 픽셀을 ignore(255)로 보냄 (scores 있을 때만 의미 있음)
    p.add_argument('--threshold', type=float, default=None,
                   help='if per-pixel scores are available, pixels with max prob < threshold '
                        'become ignore (255). If the model only outputs label maps, this is ignored.')

    # 시각화용 옵션 (학습에는 사용하지 말 것)
    p.add_argument('--vis-binary', action='store_true',
                   help='Save masks as 0(bg), 255(lesion), 128(ignore) for visualization only. '
                        'For training, do NOT use this flag (raw 0/1/255 is better).')

    return p.parse_args()


# ---------- helpers ----------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _compute_label_and_conf(pred):
    """
    pred can be:
      - (H, W) int map          -> label only, no confidence
      - (C, H, W) score/logit   -> argmax + confidence from softmax
      - (H, W, C) score/logit   -> same, but channel-last
      - tuple/dict wrappers     -> unwrap common keys

    Returns:
      label2d: (H, W) int32
      conf2d:  (H, W) float32 or None if we couldn't infer confidence
    """
    # unwrap common container types
    if isinstance(pred, tuple) and len(pred):
        pred = pred[0]
    if isinstance(pred, dict):
        for k in ('seg_pred', 'sem_seg', 'pred', 'segmentation', 'logits'):
            if k in pred:
                pred = pred[k]
                break

    a = np.asarray(pred)

    # Case 1: already label map (H,W)
    if a.ndim == 2:
        label2d = a.astype(np.int32)
        conf2d = None  # we don't know per-pixel confidence here
        return label2d, conf2d

    # Case 2: (C,H,W) or (H,W,C) scores / logits
    if a.ndim == 3:
        # Heuristic: if first dim looks like #classes
        if a.shape[0] < a.shape[-1] and a.shape[0] <= 512:
            # assume (C,H,W)
            logits = a.astype(np.float32)
            max_logits = np.max(logits, axis=0, keepdims=True)  # (1,H,W)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # (C,H,W)

            label2d = np.argmax(probs, axis=0).astype(np.int32)
            conf2d = np.max(probs, axis=0).astype(np.float32)
            return label2d, conf2d
        else:
            # assume (H,W,C)
            logits = a.astype(np.float32)
            max_logits = np.max(logits, axis=-1, keepdims=True)  # (H,W,1)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (H,W,C)

            label2d = np.argmax(probs, axis=-1).astype(np.int32)
            conf2d = np.max(probs, axis=-1).astype(np.float32)
            return label2d, conf2d

    # Fallback: squeeze and treat as label map
    squeezed = np.squeeze(a)
    if squeezed.ndim == 2:
        return squeezed.astype(np.int32), None

    # Last resort: argmax on last dim
    label2d = np.argmax(squeezed, axis=-1).astype(np.int32)
    conf2d = np.max(squeezed, axis=-1).astype(np.float32)
    return label2d, conf2d


def _apply_threshold(label2d, conf2d, threshold, ignore_index=IGNORE_INDEX, warn_state=None):
    """
    If threshold is not None and we have conf2d,
    any pixel with conf < threshold becomes ignore_index (e.g., 255).

    If the model only outputs label maps (conf2d is None),
    threshold is ignored, with a one-time warning.
    """
    if threshold is None:
        return label2d, warn_state
    if conf2d is None:
        if warn_state is not None and not warn_state.get('no_conf_warned', False):
            print("[WARN] --threshold is set but no confidence map available; threshold is ignored "
                  "(model seems to output only label maps).")
            warn_state['no_conf_warned'] = True
        return label2d, warn_state

    out = label2d.copy()
    low_conf_mask = conf2d < threshold
    out[low_conf_mask] = ignore_index
    return out, warn_state


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
    # 입력은 보통 .jpg지만, 여기서는 그냥 경로만 쓰고
    # 출력 확장자는 suffix에서 결정됨.
    return rel.replace('\\', '/') if rel else f'{idx:06d}.jpg'


def _out_path(out_root, rel_img, suffix):
    """Replace filename (without ext) to add suffix, keep directories."""
    d = osp.dirname(rel_img)
    stem = osp.splitext(osp.basename(rel_img))[0]
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

    # class names
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
    warn_state = {'no_conf_warned': False}
    global_unique_vals = set()

    for i, pred in enumerate(results):
        # 1) get label + confidence (if any)
        label2d, conf2d = _compute_label_and_conf(pred)

        # 2) apply threshold: low-conf → ignore_index(255) (if conf2d exists)
        label2d_thr, warn_state = _apply_threshold(
            label2d,
            conf2d,
            threshold=args.threshold,
            ignore_index=IGNORE_INDEX,
            warn_state=warn_state
        )

        global_unique_vals.update(np.unique(label2d_thr).tolist())

        # 3) resolve output path
        rel = _rel_img_path(dataset, i)
        out_path = _out_path(args.out_dir, rel, args.suffix)
        _ensure_dir(osp.dirname(out_path))

        # 4) write PNG
        if args.vis_binary:
            # ✅ VIS ONLY: 0(bg), 255(lesion), 128(ignore)
            img_vis = np.zeros_like(label2d_thr, dtype=np.uint8)
            img_vis[label2d_thr == 1] = 255          # lesion → white
            img_vis[label2d_thr == IGNORE_INDEX] = 128  # ignore → gray
            mmcv.imwrite(img_vis, out_path)
        else:
            # ✅ TRAINING PSEUDO: 0/1/255 그대로 저장
            img_raw = label2d_thr.astype(np.uint8)
            mmcv.imwrite(img_raw, out_path)

        saved += 1

    print(f"[PNG] Saved {saved} masks → {osp.abspath(args.out_dir)}")
    print(f"[INFO] Raw label values present: {sorted(global_unique_vals)}")
    if not args.vis_binary:
        print(" [INFO] Saved raw labels with values like {0,1,255}. "
              "For training, use ignore_index=255 in loss & dataset "
              "to ignore low-confidence pixels. "
              "If images look all-black in a viewer, that's expected "
              "because values are 0/1; use --vis-binary only for visualization.")


if __name__ == '__main__':
    main()



# python segmentation/generate_png_isic.py \
#   segmentation/configs/denseclip_fpn_res50_512x512_40k_isic_30.py \
#   work_dirs/denseclip_fpn_res50_512x512_40k_isic_30/iter_40000.pth \
#   --out-dir data/ISIC/pseudo_70_0.7 \
#   --threshold 0.7
#   --vis-binary