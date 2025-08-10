import argparse
import copy
import os
import os.path as osp
import time
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import denseclip

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--finetune', default=False, action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    datasets = [build_dataset(cfg.data.train)]
    # import os, pprint
    # ds = datasets[0]

    # print("\n[PATH DEBUG] data_root:", getattr(ds, "data_root", None))
    # print("[PATH DEBUG] img_dir  :", getattr(ds, "img_dir", None))   # <-- 이걸 씀
    # print("[PATH DEBUG] ann_dir  :", getattr(ds, "ann_dir", None))

    # def _join(base, rel):
    #     # rel이 절대경로면 그대로, 아니면 base와 조인
    #     return rel if os.path.isabs(rel) else os.path.join(base, rel)

    # for i in range(5):
    #     info = ds.img_infos[i]
    #     # ADE20K는 'filename'과 'ann':{'seg_map'}가 들어있음
    #     img_base = getattr(ds, "img_dir", getattr(ds, "img_prefix", "")) or ""
    #     ann_base = getattr(ds, "ann_dir", "")
    #     img_path = _join(img_base, info["filename"])
    #     ann_path = _join(ann_base, info["ann"]["seg_map"])
    #     print(f"[PATH DEBUG] {i} IMG exists:", os.path.exists(img_path), img_path)
    #     print(f"[PATH DEBUG] {i} ANN exists:", os.path.exists(ann_path), ann_path)

    # print("[PATH DEBUG] first img_info keys:", list(ds.img_infos[0].keys()))
    # print()



    # [DEBUG] Sanity check for shape
    from mmcv.parallel import collate, scatter

    loader_cfg = cfg.data.train_dataloader if 'train_dataloader' in cfg.data else dict(samples_per_gpu=1, workers_per_gpu=1)
    loader_cfg.setdefault('samples_per_gpu', 1)
    loader_cfg.setdefault('workers_per_gpu', 1)

    from mmseg.datasets import build_dataloader
    data_loader = build_dataloader(
        datasets[0],
        samples_per_gpu=loader_cfg['samples_per_gpu'],
        workers_per_gpu=loader_cfg['workers_per_gpu'],
        dist=distributed,
        shuffle=True)

 
    for i in range(3):
        sample = datasets[0][i]
        img = sample['img'].data
        gt = sample['gt_semantic_seg'].data

        if isinstance(img, list):
            img = img[0]
        if isinstance(gt, list):
            gt = gt[0]

        print(f"[DEBUG] Sample {i}: img {img.shape}, gt {gt.shape}")

    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(datasets[0].CLASSES)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    num_parameters = sum([p.numel() for p in model.parameters()])
    logger.info(f'#Params: {num_parameters}')

    model.backbone.init_weights()

    if hasattr(model, 'text_encoder'):
        model.text_encoder.init_weights()

    model.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.backbone)

    logger.info(model)

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    if args.finetune:
        model.eval()
    with torch.autograd.set_detect_anomaly(True):
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
