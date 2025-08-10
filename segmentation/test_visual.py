# segmentation/test.py
import argparse
import os
import os.path as osp
import numpy as np
import cv2
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import denseclip  # noqa: F401


# ---------------------- utils ----------------------
def _ensure_size(img, size_wh):
    """img을 (W,H)로 리사이즈(NEAREST for masks, LINEAR for images)."""
    W, H = size_wh
    interp = cv2.INTER_NEAREST if img.ndim == 2 or img.dtype != np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(img, (W, H), interpolation=interp)

def _add_title(img_bgr, title):
    """이미지 위에 타이틀 바(상단 32px) 붙이고 글자 그리기."""
    h, w = img_bgr.shape[:2]
    bar = np.zeros((32, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _safe_join(base, rel):
    """base와 rel을 안전하게 합친다. rel이 절대경로면 그대로 반환."""
    if rel is None:
        return None
    if not base:
        return rel
    return rel if osp.isabs(rel) else osp.join(base, rel)


def _get_img_ann_paths(dataset, idx):
    """mmseg dataset의 img_infos에서 (이미지경로, 라벨경로) 추출."""
    info = dataset.img_infos[idx]

    # --- 이미지 상대경로 추출 ---
    if 'filename' in info:
        rel_img = info['filename']
    elif 'img_info' in info and isinstance(info['img_info'], dict):
        rel_img = info['img_info'].get('filename')
    elif 'img' in info:
        rel_img = info['img']
    else:
        rel_img = None

    # --- 라벨 상대경로 추출 ---
    rel_ann = None
    if 'ann' in info and isinstance(info['ann'], dict):
        rel_ann = info['ann'].get('seg_map')
    elif 'ann_info' in info and isinstance(info['ann_info'], dict):
        rel_ann = info['ann_info'].get('seg_map')

    img_base = getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    ann_base = getattr(dataset, 'ann_dir', None)

    # 이미 base 접두가 rel에 포함돼 있으면 중복 조인 방지
    def _resolve(base, rel):
        if rel is None:
            return None
        if osp.isabs(rel):
            return rel
        if base and rel.startswith(base):
            return rel
        return _safe_join(base, rel)

    img_path = _resolve(img_base, rel_img)
    ann_path = _resolve(ann_base, rel_ann)
    return img_path, ann_path


def _load_npz_mask(path):
    """npz에서 첫 배열을 uint8로 로드."""
    a = np.load(path)
    if 'arr_0' in a:
        return a['arr_0'].astype(np.uint8)
    for k in a.files:
        return a[k].astype(np.uint8)
    raise ValueError(f'No array inside {path}')


def _to_2d_mask(mask, target_shape=None, img_shape=None):
    """
    다양한 형태의 mask를 (H,W) 2D로 변환.
    - mask: 1D/2D/3D 가능
    - target_shape: (H, W) (예측 mask 크기)
    - img_shape: (H, W, C) (원본 이미지 크기)
    """
    if mask is None:
        return None
    m = np.asarray(mask)

    # squeeze (1,H,W)/(H,W,1)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)

    if m.ndim == 2:
        return m.astype(np.uint8)

    if m.ndim == 1:
        N = m.size
        # 우선 기준 크기 결정
        if target_shape is not None:
            ref_h, ref_w = int(target_shape[0]), int(target_shape[1])
        elif img_shape is not None:
            ref_h, ref_w = int(img_shape[0]), int(img_shape[1])
        else:
            raise AssertionError(f"Cannot reshape 1D mask of length {N} without target/img shape")

        if N == ref_h * ref_w:
            # C-order 시도
            mC = m.reshape((ref_h, ref_w), order='C')
            # F-order 시도
            mF = m.reshape((ref_h, ref_w), order='F')

            # 둘 중 더 ‘덜 줄무늬’인 걸 고르기 (간단한 휴리스틱)
            def _stripe_score(a):
                # 행/열 차분의 절대합이 작을수록 덜 줄무늬
                return np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum()
            return (mC if _stripe_score(mC) < _stripe_score(mF) else mF).astype(np.uint8)

        # 크기가 다르면 근사 reshape → 리사이즈 (여긴 기존 로직 유지)
        ratio = ref_w / ref_h
        h_guess = max(1, int(round(np.sqrt(N / max(ratio, 1e-6)))))
        found = False
        for dh in range(0, min(4096, h_guess + 2048)):
            for cand_h in (h_guess - dh, h_guess + dh):
                if cand_h >= 1 and N % cand_h == 0:
                    cand_w = N // cand_h
                    m2 = m.reshape(cand_h, cand_w, order='C')
                    found = True
                    break
            if found:
                break
        if not found:
            cand_h = int(round(np.sqrt(N)))
            cand_w = max(1, N // max(1, cand_h))
            if cand_h * cand_w != N:
                pad = cand_h * cand_w - N
                if pad > 0:
                    m = np.pad(m, (0, pad), mode='edge')
            m2 = m.reshape(cand_h, cand_w, order='C')

        m2 = cv2.resize(m2.astype(np.uint8), (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        return m2.astype(np.uint8)

    raise AssertionError(f"mask must be 1D/2D/3D, got shape {m.shape}")


def _colorize(mask, palette, ignore_index=255):
    """(H,W) 마스크를 팔레트로 색칠된 BGR 이미지로 변환."""
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
# ---------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file (.pth)')
    parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--format-only', action='store_true',
                        help='Format the output results without perform evaluation')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, e.g., mIoU, mDice')
    parser.add_argument('--show', action='store_true', help='show results (not used here)')
    parser.add_argument('--show-dir', help='directory where triptych images will be saved')
    parser.add_argument('--gpu-collect', action='store_true', help='use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for collecting results from multiple workers')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='eval options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--opacity', type=float, default=0.6,
                        help='Opacity of prediction overlay (0~1]')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--vis-mode', choices=['overlay', 'mask'], default='overlay',
                    help='How to visualize prediction: overlay on image or color mask.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # 적어도 하나 요청
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Please specify at least one of --out / --eval / --format-only / --show / --show-dir'

    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot be both specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a .pkl/.pickle')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        # MultiScaleFlipAug가 pipeline[1]에 있다고 가정 (DenseCLIP 기본과 동일)
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # dist init
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # data & loader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # model
    cfg.model.train_cfg = None
    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(dataset.CLASSES)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # palette/CLASSES 메타 반영
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    efficient_test = eval_kwargs.get('efficient_test', False)

    # inference
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, None, efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, efficient_test)

    print('test done')
    rank, _ = get_dist_info()

    # rank 0: 저장/평가/시각화
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        if args.format_only:
            dataset.format_results(outputs, **eval_kwargs)

        if args.eval:
            dataset.evaluate(outputs, args.eval, **eval_kwargs)

        # -------- Triptych save (Input | Prediction overlay | GT) --------
        if args.show_dir:
            save_root = osp.abspath(args.show_dir)
            mmcv.mkdir_or_exist(save_root)
            print(f'[VIS] saving triptych to: {save_root}')

            palette = getattr(dataset, 'PALETTE', None)
            if palette is None:
                palette = getattr(model, 'PALETTE', None) or getattr(getattr(model, 'module', model), 'PALETTE', None)
            assert palette is not None, 'PALETTE is required for visualization.'

            N = len(dataset)
            assert isinstance(outputs, list) and len(outputs) == N, \
                f'Unexpected outputs length: {len(outputs)} vs dataset: {N}'

            saved = 0
            for i in range(N):
                img_path, ann_path = _get_img_ann_paths(dataset, i)
                if img_path is None or not osp.exists(img_path):
                    # print(f'[VIS][skip] missing img: {img_path}')
                    continue

                # 1) Input
                img = mmcv.imread(img_path)          # BGR
                baseH, baseW = img.shape[:2]

                # 2) Prediction
                pred = outputs[i]
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred = np.asarray(pred).astype(np.uint8)
                if pred.ndim == 3 and pred.shape[0] == 1:  # (1,H,W) → (H,W)
                    pred = pred[0]
                if pred.ndim == 1:
                    # 안전하게 이미지 크기로 reshape
                    pred = _to_2d_mask(pred, target_shape=(baseH, baseW), img_shape=img.shape)
                elif pred.ndim == 2 and (pred.shape[0] != baseH or pred.shape[1] != baseW):
                    pred = _ensure_size(pred, (baseW, baseH))

                pred_color = _colorize(pred, palette)           # (H,W,3)
                if args.vis_mode == 'overlay':
                    pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0)
                else:
                    pred_vis = pred_color

                # 3) Ground Truth
                gt_mask = None
                if ann_path and osp.exists(ann_path):
                    if ann_path.endswith('.npz'):
                        gt_mask = _load_npz_mask(ann_path)
                    else:
                        gt_mask = mmcv.imread(ann_path, flag='grayscale')
                        if gt_mask.ndim == 3:
                            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                        gt_mask = gt_mask.astype(np.uint8)

                gt_mask = dataset.get_gt_seg_map_by_idx(i)  # (H, W) np.uint8, 클래스 인덱스

                # 사이즈 맞추기 (원본 이미지 크기 기준)
                if gt_mask.shape != (baseH, baseW):
                    gt_mask = cv2.resize(gt_mask, (baseW, baseH), interpolation=cv2.INTER_NEAREST)

                gt_color = _colorize(gt_mask, palette)
                right = _add_title(cv2.addWeighted(img, 1.0 - float(args.opacity),
                                                gt_color, float(args.opacity), 0),
                                'Ground Truth (overlay)')
                # 라벨(제목) 추가 + 순서: Input | Prediction | Ground Truth
                left  = _add_title(img, 'Input')
                mid   = _add_title(pred_vis, 'Prediction (overlay)')
                #right = gt_vis  # 위에서 만든 GT overlay
                
                triptych = cv2.hconcat([left, mid, right])

                stem = osp.splitext(osp.basename(img_path))[0]
                out_path = osp.join(save_root, f'{i:06d}_{stem}.png')
                mmcv.imwrite(triptych, out_path)
                saved += 1

            print(f'[VIS] saved {saved}/{N} images at: {save_root}')

        # ---------------------------------------------------------------


if __name__ == '__main__':
    main()
