import argparse
import os
import os.path as osp
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

# ------------ ISIC Task1 specific helpers ------------

ISIC_CLASSES = ('background', 'lesion')
ISIC_PALETTE = [(0, 0, 0), (0, 0, 255)]  # BGR for lesion overlay (red)

def binarize_png_mask(path):
    """Read ISIC *_segmentation.png and convert to {0,1}."""
    m = mmcv.imread(path, flag='unchanged')
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    return m

def get_img_ann_paths(dataset, idx):
    import os.path as osp
    info = dataset.img_infos[idx]

    # image path
    rel_img = info.get('img_info', {}).get('filename', info.get('filename'))
    img_base = getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    img_path = rel_img if (rel_img and osp.isabs(rel_img)) else (osp.join(img_base, rel_img) if (img_base and rel_img) else rel_img)

    # ann path (dataset API가 제일 안전)
    ann_path = None
    try:
        ann_info = dataset.get_ann_info(idx)   # {'seg_map': 'ISIC_0000000_segmentation.png'}
        rel_ann = ann_info.get('seg_map') if isinstance(ann_info, dict) else None
        ann_base = getattr(dataset, 'ann_dir', None) or getattr(dataset, 'seg_prefix', None)
        if rel_ann:
            ann_path = rel_ann if osp.isabs(rel_ann) else (osp.join(ann_base, rel_ann) if ann_base else rel_ann)
    except Exception:
        ann_path = None

    return img_path, ann_path

    def _abs_join(base, rel):
        if rel is None:
            return None
        if osp.isabs(rel):
            return rel
        if base:
            path = osp.join(base, rel)
        else:
            path = rel
        if data_root and not osp.isabs(path):
            path = osp.join(data_root, path)
        return path

    img_path = _abs_join(img_base, rel_img)
    ann_path = _abs_join(ann_base, rel_ann)

    # fallbacks if constructed path missing but relative exists
    if img_path and not osp.exists(img_path) and rel_img and osp.exists(rel_img):
        img_path = rel_img
    if ann_path and not osp.exists(ann_path) and rel_ann and osp.exists(rel_ann):
        ann_path = rel_ann

    return img_path, ann_path

def colorize(mask, palette, ignore_index=255):
    """Map 0/1 labels to colors (BGR)."""
    if mask is None:
        return None
    mask = mask.astype(np.int32)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in np.unique(mask):
        if cls_idx == ignore_index or cls_idx < 0:
            continue
        color = (0,0,0) if cls_idx >= len(palette) else palette[cls_idx]
        out[mask == cls_idx] = color
    return out

def add_title(img_bgr, title, font_scale=0.8, thickness=2, pad=6):
    h, w = img_bgr.shape[:2]
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_h = th + base + pad * 2
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    y = pad + th
    cv2.putText(bar, title, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Visualize ISIC Task1 predictions')
    p.add_argument('config', help='config file')
    p.add_argument('checkpoint', help='checkpoint (.pth)')
    p.add_argument('--show-dir', required=True, help='directory to save triptychs')
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    p.add_argument('--opacity', type=float, default=0.6, help='overlay opacity for masks')
    p.add_argument('--max-vis', type=int, default=0, help='>0 to limit number of saved images')
    p.add_argument('--eval', action='store_true', help='run evaluation on test set before visualization')
    p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--test-split', default='test', choices=['val','test'],
                   help='which split from cfg.data to visualize')
    args = p.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # pick dataset split
    ds_cfg = cfg.data.test if args.test_split == 'test' else cfg.data.val
    ds_cfg.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    dataset = build_dataset(ds_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu if hasattr(cfg, 'data') else 2,
        dist=distributed,
        shuffle=False
    )

    # Build model
    cfg.model.train_cfg = None
    # Set class meta for model (for some heads that read CLASSES/PALETTE)
    if isinstance(cfg.model, dict):
        cfg.model.setdefault('class_names', list(getattr(dataset, 'CLASSES', ISIC_CLASSES)))

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # fallback CLASSES/PALETTE
    if not hasattr(model, 'CLASSES') or model.CLASSES is None:
        model.CLASSES = getattr(dataset, 'CLASSES', ISIC_CLASSES)
    if not hasattr(model, 'PALETTE') or model.PALETTE is None:
        model.PALETTE = getattr(dataset, 'PALETTE', ISIC_PALETTE)

    torch.cuda.empty_cache()

    # inference
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    save_root = osp.abspath(args.show_dir)
    mmcv.mkdir_or_exist(save_root)

    CLASSES = list(getattr(dataset, 'CLASSES', ISIC_CLASSES))
    PALETTE = list(getattr(dataset, 'PALETTE', ISIC_PALETTE))
    ignore_index = 255

    N = len(dataset)
    limit = args.max_vis if args.max_vis and args.max_vis > 0 else N
    saved = 0

    for i in range(N):
        if saved >= limit:
            break

        # prediction (mmseg returns 2D label map per sample)
        pred = outputs[i]
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = np.asarray(pred).astype(np.uint8)  # expected {0,1} with 255 for ignore (from pad)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        assert pred.ndim == 2, f'Unexpected pred shape: {pred.shape}'

        # image & GT
        img_path, ann_path = get_img_ann_paths(dataset, i)

        # load image
        if img_path and osp.exists(img_path):
            img = mmcv.imread(img_path)
        else:
            H, W = pred.shape
            img = np.zeros((H, W, 3), np.uint8)

        # load GT (binarize 0/255 -> 0/1)
        gt = None
        if ann_path and osp.exists(ann_path):
            gt = binarize_png_mask(ann_path)
            if gt is not None and gt.shape != pred.shape:
                gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        # colorize pred/gt
        pred_color = colorize(pred, PALETTE, ignore_index=ignore_index)
        gt_color = colorize(gt, PALETTE, ignore_index=ignore_index) if gt is not None else np.zeros_like(img)

        # resize image to pred if mismatch
        if img.shape[:2] != pred.shape:
            img = cv2.resize(img, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)

        # overlay
        alpha = float(args.opacity)
        pred_vis = cv2.addWeighted(img, 1.0 - alpha, pred_color, alpha, 0)
        gt_vis   = cv2.addWeighted(img, 1.0 - alpha, gt_color,   alpha, 0) if gt is not None else np.zeros_like(img)

        # titles
        left  = add_title(img,      'Input')
        mid   = add_title(pred_vis, 'Prediction (overlay)')
        right = add_title(gt_vis,   'Ground Truth (overlay)')
        trip  = cv2.hconcat([left, mid, right])

        # save
        stem = osp.splitext(osp.basename(img_path) if img_path else f'img_{i:06d}')[0]
        out_path = osp.join(save_root, f'{i:06d}_{stem}.png')
        mmcv.imwrite(trip, out_path)
        saved += 1

    print(f'[VIS] saved {saved}/{N} images at: {save_root}')


if __name__ == '__main__':
    main()
