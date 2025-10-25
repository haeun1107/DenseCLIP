# segmentation/generate_npz_A0.py
# MMSegmentation-style test script for DenseCLIP with NPZ saving (dense or sparse, GT-style supported)
import argparse
import os
import os.path as osp
import time
import numpy as np

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from scipy import sparse
from scipy.sparse import load_npz as scipy_load_npz

import denseclip  # noqa: F401  # DenseCLIP modules registration


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model (DenseCLIP, NPZ saving)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='dir to dump eval metrics json')
    parser.add_argument('--aug-test', action='store_true',
                        help='Use Flip and Multi scale aug (if pipeline supports)')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--format-only', action='store_true',
                        help='Format the output results without evaluation')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, e.g., "mIoU"')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='Opacity of painted segmap in (0,1].')
    parser.add_argument('--gpu-collect', action='store_true',
                        help='use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp directory for multi-gpu collection')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override settings in config (key=val)')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # === NPZ saving (dense) ===
    parser.add_argument('--save-npz', action='store_true',
                        help='Save per-image predictions as .npz (compressed)')
    parser.add_argument('--npz-prefix', type=str, default=None,
                        help='Directory for .npz files (e.g., data/BTCV/denseclip_npz)')
    parser.add_argument('--map-zero-to-255', action='store_true',
                        help='Map class 0 → 255 (ignore) before saving (dense only; sparse is recommended for BTCV)')
    parser.add_argument('--npz-dtype', type=str, default='uint8',
                        choices=['uint8', 'uint16', 'int32'],
                        help='dtype of saved dense label maps')

    # === NPZ saving (sparse) ===
    parser.add_argument('--sparse', action='store_true',
                        help='Save label maps as SciPy sparse matrices instead of dense arrays')
    parser.add_argument('--sparse-format', type=str, default='csr',
                        choices=['csr', 'coo'], help='Sparse format (default: csr)')

    # === GT-style 옵션 ===
    parser.add_argument('--bg-gt-dir', type=str, default=None,
                        help='(recommended) GT sparse dir. Columns with no foreground in GT are removed (all-zero).')
    parser.add_argument('--bg-index', type=int, default=None,
                        help='Treat this class index as background when saving (those pixels are removed).')

    # round-trip 검증(저장 후 즉시 복원해서 일치 확인)
    parser.add_argument('--verify', action='store_true',
                        help='After saving, reload npz and verify equality with original labels')

    # === NEW: probability → sparse 저장 시 임계값 ===
    parser.add_argument('--prob-thres', type=float, default=0.5,
                        help='Threshold to keep per-class probabilities [C,H,W] in sparse (values kept as data).')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _as_label_map(pred):
    """Convert prediction to 2D label map (H,W). Handles:
       - (H,W) label map
       - (C,H,W) / (H,W,C) logits/probs → argmax
       - (pred, *meta) tuple → first item
       - dict with 'seg_pred'/'sem_seg'/'pred'/'segmentation'
    """
    if isinstance(pred, tuple) and len(pred) > 0:
        pred = pred[0]
    if isinstance(pred, dict):
        for k in ('seg_pred', 'sem_seg', 'pred', 'segmentation'):
            if k in pred:
                pred = pred[k]
                break
    arr = np.asarray(pred)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # channels-first or channels-last
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:  # small C heuristic
            return arr.argmax(axis=0)
        else:
            return arr.argmax(axis=-1)
    return np.squeeze(arr)


def _extract_array(pred):
    """Return raw ndarray from result item."""
    if isinstance(pred, tuple) and len(pred) > 0:
        pred = pred[0]
    if isinstance(pred, dict):
        for k in ('seg_pred', 'sem_seg', 'pred', 'segmentation'):
            if k in pred:
                pred = pred[k]
                break
    return np.asarray(pred)


def _maybe_map_zero_to_255(x):
    x = x.astype(np.int32, copy=False)
    return np.where(x == 0, 255, x)


def _get_item_name(dataset, idx):
    """Prefer original filename in dataset meta; fallback to index."""
    name = None
    info = None
    for key in ('img_infos', 'data_infos'):
        if hasattr(dataset, key):
            lst = getattr(dataset, key)
            if 0 <= idx < len(lst):
                info = lst[idx]
                break
    if isinstance(info, dict):
        for k in ('filename', 'file_name'):
            if k in info and isinstance(info[k], str) and info[k]:
                name = osp.splitext(osp.basename(info[k]))[0]
                break
        if name is None:
            img_info = info.get('img_info', {})
            fn = img_info.get('filename') or img_info.get('file_name')
            if isinstance(fn, str) and fn:
                name = osp.splitext(osp.basename(fn))[0]
    if name is None:
        name = f'{idx:06d}'
    return name


def _ensure_dir(p):
    if p and not osp.isdir(p):
        os.makedirs(p, exist_ok=True)


def _save_dense_npz(path_wo_ext, label2d, to_dtype, map_zero_to_255=False):
    lab = label2d
    if map_zero_to_255:
        lab = _maybe_map_zero_to_255(lab)
    lab = lab.astype(to_dtype, copy=False)
    np.savez_compressed(path_wo_ext + '.npz', pred=lab)


def _load_fg_mask_from_gt(gt_dir, name, H, W):
    """Load GT sparse (C, H*W) and return foreground-columns mask (H*W,)."""
    path = osp.join(gt_dir, f'{name}.npz')
    sp = scipy_load_npz(path)  # (C, H*W)
    if sp.shape[1] != H * W:
        raise ValueError(f'GT shape mismatch: {sp.shape} vs (C,{H*W}) for {name}')
    col_sums = np.asarray(sp.sum(axis=0)).ravel()
    fg_cols = col_sums > 0
    return fg_cols


def _save_sparse_from_label(path_wo_ext, label2d, num_classes, fmt='csr',
                            fg_cols_mask=None, bg_index=None):
    """Save 2D label map as GT-style sparse one-hot (C, HW)."""
    H, W = label2d.shape
    C = int(num_classes)
    HW = H * W

    pix = np.arange(HW, dtype=np.int32)
    cls = label2d.reshape(-1).astype(np.int32)

    # Optional: keep only GT-foreground columns
    if fg_cols_mask is not None:
        if fg_cols_mask.shape[0] != HW:
            raise ValueError(f'fg_cols_mask shape {fg_cols_mask.shape} != {HW}')
        keep = fg_cols_mask
        pix = pix[keep]
        cls = cls[keep]

    # Optional: drop explicit background class
    if bg_index is not None:
        keep2 = (cls != bg_index)
        pix = pix[keep2]
        cls = cls[keep2]

    valid = (cls >= 0) & (cls < C)
    pix = pix[valid]
    cls = cls[valid]

    data = np.ones_like(pix, dtype=np.int32)
    if fmt == 'csr':
        sp = sparse.csr_matrix((data, (cls, pix)), shape=(C, HW))
    elif fmt == 'coo':
        sp = sparse.coo_matrix((data, (cls, pix)), shape=(C, HW))
    else:
        raise ValueError(f'Unsupported sparse format: {fmt}')

    sparse.save_npz(path_wo_ext + '.npz', sp)


def _save_sparse_from_prob(path_wo_ext, prob3d, thres=0.5, fmt='csr',
                           fg_cols_mask=None):
    """Save [C,H,W] probability/logit as sparse (C, HW) keeping values >= thres.
       Values are stored as data (float32), so argmax after load recovers classes.
       Columns where no class >= thres become all-zero (treated as background later).
    """
    assert prob3d.ndim == 3, f'Expected [C,H,W], got {prob3d.shape}'
    C, H, W = prob3d.shape
    HW = H * W

    # Flatten to (C, HW)
    P = prob3d.reshape(C, HW)

    # Keep entries >= thres
    keep_mask = P >= float(thres)
    rows, cols = np.where(keep_mask)
    vals = P[rows, cols].astype(np.float32)

    # Optional: GT-based foreground column filter
    if fg_cols_mask is not None:
        if fg_cols_mask.shape[0] != HW:
            raise ValueError(f'fg_cols_mask shape {fg_cols_mask.shape} != {HW}')
        keep_cols = fg_cols_mask[cols]
        rows = rows[keep_cols]
        cols = cols[keep_cols]
        vals = vals[keep_cols]

    if fmt == 'csr':
        sp = sparse.csr_matrix((vals, (rows, cols)), shape=(C, HW))
    elif fmt == 'coo':
        sp = sparse.coo_matrix((vals, (rows, cols)), shape=(C, HW))
    else:
        raise ValueError(f'Unsupported sparse format: {fmt}')

    sparse.save_npz(path_wo_ext + '.npz', sp)


def _reload_npz(path, H=None, W=None):
    """Auto-detect saved format and recover (H,W) by argmax.
    - sparse: (H*W, C) or (C, H*W) → (H, W)
    - dense : pred key (H, W)
    """
    try:
        sp = scipy_load_npz(path)  # sparse OK (CSR/COO)
        rows, cols = sp.shape
        if H is None or W is None:
            raise ValueError("Need H,W for sparse reload")
        HW = H * W

        if rows == HW:             # (H*W, C)
            pred = sp.argmax(axis=1).A1
        elif cols == HW:           # (C, H*W)
            pred = sp.argmax(axis=0).A1
        else:
            raise ValueError(f"Unexpected sparse shape {sp.shape}; cannot map to (H,W) with H*W={HW}")

        return pred.reshape(H, W).astype(np.int32)

    except Exception:
        with np.load(path) as f:
            if 'pred' not in f:
                raise ValueError(f'Unknown npz schema: {path}')
            return f['pred'].astype(np.int32)


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.save_npz, \
        'Specify at least one operation: --out/--eval/--format-only/--show/--show-dir/--save-npz'

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be .pkl/.pickle')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # optional aug-test (if pipeline supports)
    if args.aug_test and hasattr(cfg.data.test, 'pipeline') and len(cfg.data.test.pipeline) > 1:
        if isinstance(cfg.data.test.pipeline[1], dict):
            cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
            cfg.data.test.pipeline[1].flip = True

    # test mode
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # distributed
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # work dir
    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(work_dir, f'eval_{timestamp}.json')

    # dataset & dataloader
    dataset = build_dataset(cfg.data.test)

    # DenseCLIP requires class_names
    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(dataset.CLASSES)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=getattr(cfg.data, 'workers_per_gpu', 2),
        dist=distributed,
        shuffle=False
    )

    # model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    torch.cuda.empty_cache()

    # run test
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            model, data_loader, args.show, args.show_dir, False, args.opacity,
            pre_eval=args.eval is not None,
            format_only=args.format_only,
            format_args={} if args.eval_options is None else args.eval_options
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect, False,
            pre_eval=args.eval is not None,
            format_only=args.format_only,
            format_args={} if args.eval_options is None else args.eval_options
        )

    # post (rank 0)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)

        if args.eval:
            eval_kwargs = {} if args.eval_options is None else args.eval_options
            metric = dataset.evaluate(results, metric=args.eval, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if 'json_file' in locals():
                mmcv.dump(metric_dict, json_file, indent=4)

        # === NPZ saving ===
        if args.save_npz:
            if not args.npz_prefix:
                raise ValueError('--save-npz requires --npz-prefix')
            save_dir = args.npz_prefix
            mmcv.mkdir_or_exist(save_dir)

            dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'int32': np.int32}
            to_dtype = dtype_map[args.npz_dtype]

            saved = 0
            mismatches = 0
            for i, pred in enumerate(results):
                arr = _extract_array(pred)          # raw ndarray
                name = _get_item_name(dataset, i)

                # Derive label2d for verification (argmax of arr if needed)
                if arr.ndim == 3:
                    # channels-first assumed if C <= 512 and smallest dim is first
                    if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:
                        label2d = arr.argmax(axis=0)
                        C, H, W = arr.shape
                    else:
                        label2d = arr.argmax(axis=-1)
                        H, W, C = arr.shape
                        arr = np.transpose(arr, (2, 0, 1))  # to [C,H,W]
                    prob3d = arr  # [C,H,W]
                elif arr.ndim == 2:
                    label2d = arr
                    H, W = label2d.shape
                    C = len(dataset.CLASSES)
                    prob3d = None
                else:
                    label2d = _as_label_map(arr)
                    H, W = label2d.shape
                    C = len(dataset.CLASSES)
                    prob3d = None

                out_wo_ext = osp.join(save_dir, name)

                # Optional GT-based foreground mask
                fg_cols = None
                bg_gt_dir = getattr(args, 'bg_gt_dir', None)
                if bg_gt_dir:
                    try:
                        fg_cols = _load_fg_mask_from_gt(bg_gt_dir, name, H, W)  # (H*W,) bool
                    except Exception as e:
                        print(f'[WARN] failed to load GT for {name}: {e} (falling back to no fg mask)')

                if args.sparse:
                    if prob3d is not None:
                        # Save probabilities as sparse with threshold; values kept as data
                        _save_sparse_from_prob(
                            out_wo_ext, prob3d, thres=args.prob_thres,
                            fmt=args.sparse_format, fg_cols_mask=fg_cols
                        )
                        saved_mask = (prob3d >= float(args.prob_thres)).any(axis=0).reshape(-1)  # for verify
                    else:
                        _save_sparse_from_label(
                            out_wo_ext, label2d, num_classes=C,
                            fmt=args.sparse_format, fg_cols_mask=fg_cols, bg_index=args.bg_index
                        )
                        # When saving hard labels, consider all kept columns (after GT/bg filters)
                        HW = H * W
                        kept = np.ones(HW, dtype=bool)
                        if fg_cols is not None:
                            kept &= fg_cols
                        if args.bg_index is not None:
                            kept &= (label2d.reshape(-1) != args.bg_index)
                        saved_mask = kept
                else:
                    # Dense path (rarely used for BTCV training)
                    _save_dense_npz(out_wo_ext, label2d, to_dtype, map_zero_to_255=args.map_zero_to_255)
                    saved_mask = np.ones(H * W, dtype=bool)

                # Optional round-trip verification
                if args.verify:
                    rec = _reload_npz(out_wo_ext + '.npz', H=H, W=W)

                    HW = H * W
                    keep_mask = np.ones(HW, dtype=bool)

                    # Prefer mask of actually saved columns if we did prob-thresholding
                    if args.sparse and prob3d is not None:
                        keep_mask &= saved_mask
                    elif fg_cols is not None:
                        keep_mask &= fg_cols
                    if args.bg_index is not None:
                        keep_mask &= (label2d.reshape(-1) != args.bg_index)

                    if keep_mask.any():
                        ok = (rec.reshape(-1)[keep_mask] == label2d.reshape(-1)[keep_mask]).mean()
                    else:
                        ok = 1.0
                    if ok < 0.9999:
                        mismatches += 1
                        kept_n = int(keep_mask.sum())
                        print(f'[WARN] round-trip partial mismatch @ {name}: kept={kept_n}/{HW}, match={ok:.6f}')

                saved += 1

            mode = 'SPARSE' if args.sparse else 'DENSE'
            print(f'[NPZ:{mode}] Saved {saved} predictions → {osp.abspath(save_dir)}')
            if args.verify:
                print(f'[VERIFY] mismatches: {mismatches} / {saved}')


if __name__ == '__main__':
    main()
