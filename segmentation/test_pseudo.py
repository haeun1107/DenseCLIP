# segmentation/test_denseclip_npz.py
# MMSegmentation-style test script for DenseCLIP with NPZ saving
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
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

    # === NPZ saving ===
    parser.add_argument('--save-npz', action='store_true',
                        help='Save per-image predictions as .npz (compressed)')
    parser.add_argument('--npz-prefix', type=str, default=None,
                        help='Directory/prefix for .npz files (e.g., data/BTCV/denseclip_npz)')
    parser.add_argument('--map-zero-to-255', action='store_true',
                        help='Map class 0 → 255 (ignore) before saving')
    parser.add_argument('--npz-dtype', type=str, default='uint8',
                        choices=['uint8', 'uint16', 'int32'],
                        help='dtype of saved label maps')

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
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 512:  # heuristic: small C
            return arr.argmax(axis=0)
        else:
            return arr.argmax(axis=-1)
    return np.squeeze(arr)


def _maybe_map_zero_to_255(x):
    x = x.astype(np.int32, copy=False)
    return np.where(x == 0, 255, x)


def _get_item_name(dataset, idx):
    """Use filename from dataset meta when possible; else use zero-padded index."""
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

    # set cudnn_benchmark
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

    # distributed init
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # work dir & metric json path
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

    # dataloader
    dataset = build_dataset(cfg.data.test)

    # DenseCLIP은 class_names 필수
    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(dataset.CLASSES)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
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

    # run
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

    # rank 0 post
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)

        if args.eval:
            eval_kwargs = {} if args.eval_options is None else args.eval_options
            metric = dataset.evaluate(results, metric=args.eval, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)

        # === NPZ saving ===
        if args.save_npz:
            if not args.npz_prefix:
                raise ValueError('--save-npz requires --npz-prefix')
            save_dir = args.npz_prefix
            _ensure_dir(save_dir)

            dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'int32': np.int32}
            to_dtype = dtype_map[args.npz_dtype]

            for i, pred in enumerate(results):
                label2d = _as_label_map(pred)
                label2d = np.squeeze(label2d)
                if label2d.ndim != 2:
                    raise ValueError(f'Item {i} prediction shape {label2d.shape} not (H,W)')

                if args.map_zero_to_255:
                    label2d = _maybe_map_zero_to_255(label2d)

                label2d = label2d.astype(to_dtype, copy=False)
                name = _get_item_name(dataset, i)
                np.savez_compressed(osp.join(save_dir, f'{name}.npz'), pred=label2d)

            print(f'[NPZ] Saved {len(results)} predictions → {osp.abspath(save_dir)}')


if __name__ == '__main__':
    main()
