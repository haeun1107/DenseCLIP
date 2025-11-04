# -*- coding: utf-8 -*-
"""
A0 (with BG) tailored DenseCLIP inference → save as BTCV-style sparse .npz
Stores 14-channel one-hot including BG(0)
If you want to exclude the BG column for evaluation, use --bg-index 0
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint, get_dist_info, init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmseg.apis import single_gpu_test, multi_gpu_test
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from scipy import sparse
from scipy.sparse import save_npz

import denseclip  # register DenseCLIP


def parse_args():
    parser = argparse.ArgumentParser(description='DenseCLIP BTCV A0 with BG NPZ export')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', required=True, help='directory to save .npz files')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--sparse-format', default='csr', choices=['csr', 'coo'],
                        help='sparse matrix format')
    parser.add_argument('--bg-index', type=int, default=None,
                        help='set to 0 to drop BG columns (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='ignore predictions with prob <= threshold')
    parser.add_argument('--verify', action='store_true',
                        help='verify saved npz by reloading')
    args = parser.parse_args()
    return args


def _as_label_map(pred, threshold=None):
    """Convert prediction (prob map or label map) into label2d."""
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    arr = np.asarray(pred)
    # --- softmax map with threshold ---
    if arr.ndim == 3:
        if threshold is not None:
            prob_max = arr.max(axis=0)
            label2d = arr.argmax(axis=0)
            label2d[prob_max <= threshold] = 255  # ignore
            return label2d
        else:
            return arr.argmax(axis=0)
    return arr


def _save_sparse_from_label(path_wo_ext, label2d, num_classes, fmt='csr', bg_index=None):
    """Save label2d (H,W) as BTCV-style sparse one-hot (C, H*W)."""
    H, W = label2d.shape
    HW = H * W
    cls = label2d.reshape(-1).astype(np.int32)
    pix = np.arange(HW, dtype=np.int32)

    # skip ignore label (255)
    valid = (cls >= 0) & (cls < num_classes)
    cls = cls[valid]
    pix = pix[valid]

    if bg_index is not None:
        mask = cls != bg_index
        cls = cls[mask]
        pix = pix[mask]

    data = np.ones_like(cls, dtype=np.int8)
    if fmt == 'csr':
        sp = sparse.csr_matrix((data, (cls, pix)), shape=(num_classes, HW))
    else:
        sp = sparse.coo_matrix((data, (cls, pix)), shape=(num_classes, HW))
    save_npz(path_wo_ext + '.npz', sp)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                                   dist=distributed, shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', dataset.PALETTE)
    model = MMDataParallel(model, device_ids=[0])

    mmcv.mkdir_or_exist(args.out_dir)
    results = single_gpu_test(model, data_loader, show=False)

    num_classes = len(dataset.CLASSES)
    print(f"[INFO] Using num_classes={num_classes}, bg_index={args.bg_index}, threshold={args.threshold}")

    saved = 0
    for i, pred in enumerate(results):
        name = osp.splitext(osp.basename(dataset.img_infos[i]['img_info']['filename']))[0]
        label2d = _as_label_map(pred, threshold=args.threshold)
        out_path = osp.join(args.out_dir, name)
        _save_sparse_from_label(out_path, label2d, num_classes, fmt=args.sparse_format, bg_index=args.bg_index)
        saved += 1

        if args.verify:
            from scipy.sparse import load_npz
            sp = load_npz(out_path + '.npz')
            rec = sp.argmax(axis=0).A.reshape(label2d.shape)
            diff = np.sum(rec != label2d)
            if diff > 0:
                print(f"[WARN] mismatch in {name}.npz ({diff} pixels)")

    print(f"[DONE] Saved {saved} files → {osp.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()

    
# python segmentation/generate_npz_btcv.py \
#   segmentation/configs/denseclip_fpn_res50_512x512_80k_btcv_30.py \
#   work_dirs/denseclip_fpn_res50_512x512_80k_btcv_30/iter_64000.pth \
#   --out-dir data/BTCV/pseudo_70 \
#   --bg-index 0 \
#   --verify
