#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_visual_synapse_new.py
Visualize Synapse predictions (mmseg + DenseCLIP) with robust .h5/.npz/.nii slice handling.

- Runs inference on cfg.data.test and saves triptychs:
  [Input | Prediction(overlay) | Ground Truth(overlay)]
- Aligns the grayscale input slice with the SAME slice index that the dataset item represents.
  (tries to read slice hint from dataset.img_infos[i] or from filename patterns)
- If dataset has GT => uses dataset.get_gt_seg_map_by_idx(i).
  If dataset has no GT but .h5 has 'label' => uses the label from the .h5 file.
- Palette: dataset.PALETTE if present, otherwise generated ('bright'/'tab20'/'random').

Usage
-----
python test_visual_synapse_new.py \
  segmentation/configs/denseclip_fpn_res50_512x512_80k_synapse_new.py \
  work_dirs/denseclip_fpn_res50_512x512_80k_synapse_new/iter_57600.pth \
  --show-dir vis/synapse_new \
  --legend --outline --palette tab20 --opacity 0.55 --win -200 300 --max-vis 50
"""

import os
import os.path as osp
import argparse
import re
import math
import numpy as np
import cv2
import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, get_dist_info, init_dist
from mmcv.utils import DictAction
from mmseg.apis import single_gpu_test, multi_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor

import denseclip  # noqa: F401

# ---------- Synapse meta ----------
SYNAPSE_CLASSES = (
    "background", "aorta", "gallbladder",
    "left kidney", "right kidney", "liver",
    "pancreas", "spleen", "stomach"
)

# ---------------------- helpers ----------------------
def _ensure_size(img, size_wh):
    W, H = size_wh
    interp = cv2.INTER_NEAREST if (img.ndim == 2 or img.dtype != np.uint8) else cv2.INTER_LINEAR
    return cv2.resize(img, (W, H), interpolation=interp)

def _add_title(img_bgr, title, font_scale=0.9, thickness=2, pad=8):
    h, w = img_bgr.shape[:2]
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_h = th + base + pad * 2
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    y = pad + th
    cv2.putText(bar, title, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _resolve_path(base, rel, data_root=None):
    if rel is None:
        return None
    if osp.isabs(rel):
        return rel
    path = osp.join(base, rel) if base else rel
    if data_root and not osp.isabs(path):
        path = osp.join(data_root, path)
    return path

def _get_img_rel_anns(dataset, idx):
    """Return (img_rel, ann_rel) from dataset.img_infos[idx] safely."""
    info = dataset.img_infos[idx]
    rel_img = None
    if 'filename' in info:
        rel_img = info['filename']
    elif 'img_info' in info and isinstance(info['img_info'], dict):
        rel_img = info['img_info'].get('filename') or info['img_info'].get('file_name')
    elif 'img' in info:
        rel_img = info.get('img')

    rel_ann = None
    if 'ann' in info and isinstance(info['ann'], dict):
        rel_ann = info['ann'].get('seg_map')
    elif 'ann_info' in info and isinstance(info['ann_info'], dict):
        rel_ann = info['ann_info'].get('seg_map')

    return rel_img, rel_ann, info

def _get_img_ann_paths(dataset, idx):
    """Resolve absolute-ish paths for input image and annotation if present."""
    rel_img, rel_ann, _ = _get_img_rel_anns(dataset, idx)
    img_base = getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    ann_base = getattr(dataset, 'ann_dir', None) or getattr(dataset, 'seg_prefix', None)
    data_root = getattr(dataset, 'data_root', None)

    img_path = _resolve_path(img_base, rel_img, data_root)
    ann_path = _resolve_path(ann_base, rel_ann, data_root)

    # If constructed path missing but rel exists on disk, use rel
    if img_path and not osp.exists(img_path) and rel_img and osp.exists(rel_img):
        img_path = rel_img
    if ann_path and not osp.exists(ann_path) and rel_ann and osp.exists(rel_ann):
        ann_path = rel_ann

    return img_path, ann_path

def _read_npz_image(path):
    data = np.load(path, allow_pickle=False)
    img = data.get('image')
    if img is None:
        raise KeyError(f"'image' not found in {path}")
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 1:   # (1,H,W)
        img = img[0]
    if img.ndim == 3 and img.shape[-1] == 1:  # (H,W,1)
        img = img[..., 0]
    return img

def _read_h5_image_label(path):
    """Return (img, label_or_None). img: (H,W) or (D,H,W)."""
    import h5py
    with h5py.File(path, 'r') as f:
        if 'image' not in f:
            raise KeyError(f"'image' dataset not found in {path}")
        img = np.asarray(f['image'])
        lab = None
        for k in ('label', 'labels', 'seg', 'mask', 'gt', 'gt_seg', 'annotation'):
            if k in f:
                lab = np.asarray(f[k]); break
    return img, lab

def _read_nii_slice_bgr(path, slice_index=None, target_hw=None, win=None):
    import nibabel as nib
    vol = nib.load(path).get_fdata()
    vol = np.asarray(vol)
    if vol.ndim == 3:
        s = slice_index if slice_index is not None else vol.shape[-1] // 2
        s = max(0, min(s, vol.shape[-1] - 1))
        img2d = vol[..., s]
    elif vol.ndim == 2:
        img2d = vol
    else:
        raise ValueError(f'Unexpected image shape {vol.shape} for {path}')
    return _to_bgr(img2d, target_hw=target_hw, win=win)

def _to_gray01(img2d, win=None):
    x = np.asarray(img2d).astype(np.float32)
    if win is not None and len(win) == 2:
        lo, hi = float(win[0]), float(win[1])
        x = np.clip(x, lo, hi)
    vmin, vmax = float(x.min()), float(x.max())
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return x

def _to_bgr(img2d, target_hw=None, win=None):
    g01 = _to_gray01(img2d, win=win)
    img8 = (g01 * 255).astype(np.uint8)
    if target_hw is not None and (img8.shape[0], img8.shape[1]) != tuple(target_hw):
        H, W = int(target_hw[0]), int(target_hw[1])
        img8 = cv2.resize(img8, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

def _colorize(mask, palette, ignore_index=255):
    if mask is None:
        return None
    mask = mask.astype(np.int32)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    K = len(palette)
    for cls_idx in np.unique(mask):
        if cls_idx == ignore_index or cls_idx < 0:
            continue
        color = (0, 0, 0) if cls_idx >= K else palette[cls_idx]
        out[mask == cls_idx] = color
    return out

def _make_palette(n, scheme='bright', seed=0):
    base_bright = [
        (0, 92, 255), (0, 255, 255), (34, 139, 34), (255, 0, 0),
        (255, 0, 255), (255, 105, 180), (147, 20, 255), (60, 179, 113),
        (128, 128, 0), (0, 215, 255), (180, 130, 70), (203, 192, 255),
        (50, 205, 50), (139, 0, 0), (0, 128, 128), (128, 0, 128),
        (255, 255, 255),
    ]
    tab20 = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
        (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
        (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
    ]
    if scheme == 'bright':
        src = base_bright
    elif scheme == 'tab20':
        src = tab20
    elif scheme == 'random':
        rng = np.random.RandomState(seed)
        hsv = np.stack([
            rng.permutation(np.linspace(0, 179, n, endpoint=False)),
            np.full(n, 200),
            np.full(n, 255),
        ], axis=1).astype(np.uint8).reshape(1, n, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
        return [tuple(map(int, c)) for c in bgr]
    else:
        raise ValueError(scheme)
    out = [src[i % len(src)] for i in range(n)]
    return [tuple(map(int, c)) for c in out]

def _legend_strip(width, class_names, palette, max_cols=6, pad=8, bg=(35, 35, 35)):
    if not class_names:
        return np.zeros((1, width, 3), dtype=np.uint8)
    row_h = 26
    rows = (len(class_names) + max_cols - 1) // max_cols
    h = pad * 2 + rows * row_h
    strip = np.zeros((h, width, 3), dtype=np.uint8); strip[:] = bg
    swatch = 18; txt_h = 16
    col_w = max(1, width // max_cols)
    i = 0; y = pad
    for _r in range(rows):
        x = pad
        for _c in range(max_cols):
            if i >= len(class_names): break
            color = palette[i] if i < len(palette) else (200, 200, 200)
            x2 = min(x + swatch, width - pad)
            cv2.rectangle(strip, (x, y), (x2, y + swatch), color, -1)
            cv2.putText(strip, class_names[i][:max(4, col_w // 8)], (x + swatch + 6, y + txt_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            x += col_w; i += 1
        y += row_h
    return strip

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2, ignore_index=255):
    uniq = np.unique(mask)
    for cls in uniq:
        if cls == ignore_index or cls < 0:
            continue
        binm = (mask == cls).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        color = palette[cls] if cls < len(palette) else (255, 255, 255)
        cv2.drawContours(canvas_bgr, cnts, -1, color, thickness, lineType=cv2.LINE_AA)
    return canvas_bgr

def _extract_slice_hint(info, img_path):
    """Try to get the 3D volume slice index that this dataset item corresponds to."""
    # 1) direct keys on info
    for k in ("slice_index", "slice", "slice_id", "z_index", "slice_num"):
        if k in info:
            try:
                return int(info[k])
            except Exception:
                pass
    # 2) filename pattern
    def _find_in_text(txt):
        if not txt:
            return None
        m = re.search(r"(?:slice[_-]?|s)(\d{1,4})", str(txt))
        if m:
            return int(m.group(1))
        m2 = re.search(r"case\d+[_-]?(?:\D)?(\d{1,4})", str(txt))  # loose fallback
        if m2:
            return int(m2.group(1))
        return None
    cand = _find_in_text(info.get("filename") if isinstance(info, dict) else None)
    if cand is not None:
        return cand
    return _find_in_text(img_path)

# ---------------------- main ----------------------
def parse_args():
    p = argparse.ArgumentParser("Visualize Synapse predictions (DenseCLIP/mmseg)")
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint (.pth)')
    p.add_argument('--show-dir', required=True, help='dir to save triptychs')
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    p.add_argument('--opacity', type=float, default=0.6)
    p.add_argument('--palette', default='dataset', choices=['dataset', 'bright', 'tab20', 'random'])
    p.add_argument('--legend', action='store_true')
    p.add_argument('--legend-pos', default='outside-top',
                   choices=['outside-top', 'outside-bottom', 'inside'])
    p.add_argument('--legend-cols', type=int, default=6)
    p.add_argument('--outline', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mask-ignore', type=int, default=255)
    p.add_argument('--slice-index', type=int, default=None,
                   help='fallback slice index when input is 3D and no hint exists (default: center)')
    p.add_argument('--win', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='CT window for grayscale rendering (e.g., --win -200 300)')
    p.add_argument('--max-vis', type=int, default=0, help='>0 to limit saved items')
    p.add_argument('--eval', type=str, nargs='+', help='evaluate metrics (optional)')
    p.add_argument('--out', help='save raw outputs (.pkl/.pickle)')
    p.add_argument('--options', nargs='+', action=DictAction)
    p.add_argument('--eval-options', nargs='+', action=DictAction)
    return p.parse_args()

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # dataset / loader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=getattr(cfg.data, 'workers_per_gpu', 2),
        dist=distributed,
        shuffle=False
    )

    # model
    cfg.model.train_cfg = None
    if isinstance(cfg.model, dict):
        cfg.model.setdefault('class_names', list(getattr(dataset, 'CLASSES', SYNAPSE_CLASSES)))
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # classes & palette
    if not hasattr(model, 'CLASSES') or model.CLASSES is None:
        model.CLASSES = getattr(dataset, 'CLASSES', SYNAPSE_CLASSES)
    if not hasattr(model, 'PALETTE') or model.PALETTE is None:
        model.PALETTE = getattr(dataset, 'PALETTE', None)

    # inference
    torch.cuda.empty_cache()
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    # optional eval / dump
    if args.out:
        if not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a .pkl/.pickle')
        print(f'\n[IO] writing results to {args.out}')
        mmcv.dump(outputs, args.out)

    if args.eval:
        eval_kwargs = {} if args.eval_options is None else args.eval_options
        dataset.evaluate(outputs, args.eval, **eval_kwargs)

    # prepare vis dir
    save_root = osp.abspath(args.show_dir)
    mmcv.mkdir_or_exist(save_root)

    # palette & classes
    palette = getattr(dataset, 'PALETTE', None) or getattr(model, 'PALETTE', None)
    if args.palette != 'dataset' or palette is None:
        num_classes = len(getattr(dataset, 'CLASSES', []) or SYNAPSE_CLASSES)
        scheme = 'tab20' if args.palette == 'tab20' else ('random' if args.palette == 'random' else 'bright')
        palette = _make_palette(num_classes, scheme=scheme, seed=args.seed)
    class_names = list(getattr(dataset, 'CLASSES', []) or SYNAPSE_CLASSES)

    N = len(dataset)
    limit = args.max_vis if args.max_vis and args.max_vis > 0 else N
    saved = 0

    for i in range(N):
        if saved >= limit:
            break

        # prediction (H,W)
        pred = outputs[i]
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = np.asarray(pred).astype(np.uint8)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        assert pred.ndim == 2, f'Unexpected pred shape: {pred.shape}'
        H, W = pred.shape

        # paths & info
        img_path, _ = _get_img_ann_paths(dataset, i)
        _, _, info = _get_img_rel_anns(dataset, i)
        slice_hint = _extract_slice_hint(info, img_path)

        # load a grayscale input aligned to the SAME slice as dataset item
        img_bgr = None
        if img_path:
            low = img_path.lower()
            try:
                if low.endswith('.npz'):
                    img = _read_npz_image(img_path)
                    if img.ndim == 3:  # could be (D,H,W) or (H,W,C)
                        if img.shape[-1] not in (1, 3):  # likely (D,H,W)
                            s = slice_hint if slice_hint is not None else \
                                (args.slice_index if args.slice_index is not None else img.shape[0] // 2)
                            s = max(0, min(s, img.shape[0] - 1))
                            img2d = img[s]
                        else:
                            img2d = img[..., 0] if img.shape[-1] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        img2d = img
                    img_bgr = _to_bgr(img2d, target_hw=(H, W), win=args.win)

                elif low.endswith('.h5') or low.endswith('.hdf5'):
                    img, lab = _read_h5_image_label(img_path)
                    if img.ndim == 3:   # (D,H,W)
                        s = slice_hint if slice_hint is not None else \
                            (args.slice_index if args.slice_index is not None else img.shape[0] // 2)
                        s = max(0, min(s, img.shape[0] - 1))
                        img2d = img[s]
                        file_gt = lab[s] if (lab is not None and lab.ndim == 3 and lab.shape[0] > s) else None
                    else:
                        img2d = img
                        file_gt = lab if (lab is not None and lab.ndim == 2) else None
                    img_bgr = _to_bgr(img2d, target_hw=(H, W), win=args.win)

                elif low.endswith('.nii') or low.endswith('.nii.gz'):
                    img_bgr = _read_nii_slice_bgr(img_path, slice_index=slice_hint if slice_hint is not None else args.slice_index,
                                                  target_hw=(H, W), win=args.win)
                else:
                    if osp.exists(img_path):
                        raw = mmcv.imread(img_path)
                        if raw.shape[:2] != (H, W):
                            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
                        img_bgr = raw
            except Exception as e:
                print(f"[WARN] failed to load input image from {img_path}: {e}")

        if img_bgr is None:
            img_bgr = np.zeros((H, W, 3), np.uint8)

        # ground truth (prefer dataset)
        gt = None
        try:
            gt = dataset.get_gt_seg_map_by_idx(i)  # assume dataset already applied any reduce_zero_label if needed
            if gt.shape != (H, W):
                gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        except Exception:
            gt = None

        # if dataset had no GT but .h5 carried labels
        if gt is None and (img_path and img_path.lower().endswith(('.h5', '.hdf5'))):
            try:
                img, lab = _read_h5_image_label(img_path)
                if lab is not None:
                    if lab.ndim == 3:
                        s = slice_hint if slice_hint is not None else \
                            (args.slice_index if args.slice_index is not None else (lab.shape[0] // 2))
                        s = max(0, min(s, lab.shape[0] - 1))
                        g = lab[s]
                    else:
                        g = lab
                    gt = g.astype(np.uint8)
                    if gt.shape != (H, W):
                        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass

        # colorize + overlay
        pred_color = _colorize(pred, palette, ignore_index=args.mask_ignore)
        alpha = float(args.opacity)
        pred_vis = cv2.addWeighted(img_bgr, 1.0 - alpha, pred_color, alpha, 0)

        if gt is not None:
            gt_color = _colorize(gt, palette, ignore_index=args.mask_ignore)
            gt_vis = cv2.addWeighted(img_bgr, 1.0 - alpha, gt_color, alpha, 0)
        else:
            gt_vis = np.zeros_like(img_bgr)

        # optional contours
        if args.outline:
            pred_vis = _draw_class_contours(pred_vis, pred, palette, thickness=2, ignore_index=args.mask_ignore)
            if gt is not None:
                gt_vis = _draw_class_contours(gt_vis, gt, palette, thickness=2, ignore_index=args.mask_ignore)

        # compose
        left = _add_title(img_bgr, 'Input')
        mid = _add_title(pred_vis, 'Prediction (overlay)')
        trip = cv2.hconcat([left, mid])

        if gt is not None:
            right = _add_title(gt_vis, 'Ground Truth (overlay)')
            trip = cv2.hconcat([trip, right])

        # legend
        if args.legend and class_names:
            if args.legend_pos in ('outside-top', 'outside-bottom'):
                strip = _legend_strip(trip.shape[1], class_names, palette, max_cols=args.legend_cols)
                trip = cv2.vconcat([strip, trip]) if args.legend_pos == 'outside-top' else cv2.vconcat([trip, strip])
            else:
                # small inline legend (corner overlay)
                swatch = 18; pad = 8; txt_h = 16
                cols = min(args.legend_cols, len(class_names))
                rows = int(math.ceil(len(class_names) / cols))
                col_w = 180
                panel_w = cols * col_w + pad * (cols + 1)
                panel_h = rows * (swatch + pad) + pad
                panel = np.full((panel_h, panel_w, 3), 30, np.uint8)
                x = pad; y = pad; j = 0
                for r in range(rows):
                    x = pad
                    for c in range(cols):
                        if j >= len(class_names): break
                        color = palette[j] if j < len(palette) else (200, 200, 200)
                        cv2.rectangle(panel, (x, y), (x + swatch, y + swatch), color, -1)
                        cv2.putText(panel, class_names[j], (x + swatch + 6, y + txt_h),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                        x += col_w; j += 1
                    y += swatch + pad
                Hc, Wc = trip.shape[:2]
                ph, pw = panel.shape[:2]
                x0, y0 = 10, 10
                x1, y1 = min(x0 + pw, Wc), min(y0 + ph, Hc)
                roi = trip[y0:y1, x0:x1]
                panel = panel[:y1 - y0, :x1 - x0]
                trip[y0:y1, x0:x1] = (0.85 * panel + 0.15 * roi).astype(np.uint8)

        # save
        stem = osp.splitext(osp.basename(img_path) if img_path else f'img_{i:06d}')[0]
        out_path = osp.join(save_root, f'{i:06d}_{stem}.png')
        mmcv.imwrite(trip, out_path)
        saved += 1

    print(f'[VIS] saved {saved}/{N} images at: {save_root}')

if __name__ == '__main__':
    main()
