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

import math

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

def _stripe_score(a):
    return np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum()

def _best_hw_from_N(N, target_shape):
    Ht, Wt = int(target_shape[0]), int(target_shape[1])
    ratio = Wt / max(1.0, float(Ht))
    best = None
    limit = int(math.sqrt(N)) + 1
    for h in range(1, limit):
        if N % h: 
            continue
        w = N // h
        diff = abs((w / float(h)) - ratio)
        if best is None or diff < best[2]:
            best = (h, w, diff)
    if best is None:
        h = int(round(math.sqrt(N)))
        w = max(1, N // max(1, h))
        return h, w
    return best[0], best[1]

def _onehot_to_index(arr, class_axis=0, ignore_index=255):
    if class_axis != 0:
        arr = np.moveaxis(arr, class_axis, 0)  # (C,H,W)
    C, H, W = arr.shape
    s = arr.sum(axis=0)
    idx = arr.argmax(axis=0).astype(np.uint8)
    idx[s == 0] = ignore_index
    return idx

def _load_npz_mask(path, num_classes=None, ignore_index=255, target_shape=None,
                   treat_zero_as_ignore=False):
    a = np.load(path)
    key = 'arr_0' if 'arr_0' in a else a.files[0]
    arr = a[key]

    # 2D 인덱스(H,W)
    if arr.ndim == 2 and (num_classes is None or (arr.shape[0] != num_classes and arr.shape[1] != num_classes)):
        m = arr.astype(np.uint8)

    # 3D 원-핫
    elif arr.ndim == 3:
        if num_classes is not None:
            if arr.shape[0] == num_classes:
                m = _onehot_to_index(arr, class_axis=0, ignore_index=ignore_index)
            elif arr.shape[-1] == num_classes:
                m = _onehot_to_index(arr, class_axis=-1, ignore_index=ignore_index)
            else:
                m = _onehot_to_index(arr, class_axis=int(np.argmin(arr.shape)), ignore_index=ignore_index)
        else:
            m = _onehot_to_index(arr, class_axis=int(np.argmin(arr.shape)), ignore_index=ignore_index)

    # 2D 평탄 원-핫 (C,N) 또는 (N,C)
    elif arr.ndim == 2 and num_classes is not None and (arr.shape[0] == num_classes or arr.shape[1] == num_classes):
        if target_shape is None:
            raise ValueError(f'Need target_shape to reshape flat one-hot: {path}')
        # (C,N)으로 맞추기
        flat = arr if arr.shape[0] == num_classes else arr.T  # (C,N)
        C, N = flat.shape
        h0, w0 = _best_hw_from_N(N, target_shape)
        # C/F 두 가지로 복원해 보고 덜 줄무늬 선택
        arrC = flat.reshape(C, h0, w0, order='C')
        arrF = flat.reshape(C, h0, w0, order='F')
        scoreC = _stripe_score(arrC.sum(axis=0))
        scoreF = _stripe_score(arrF.sum(axis=0))
        arr3 = arrC if scoreC <= scoreF else arrF
        m = _onehot_to_index(arr3, class_axis=0, ignore_index=ignore_index)

    # 1D 인덱스 (H*W,)
    elif arr.ndim == 1:
        N = arr.size
        if target_shape is None:
            raise ValueError(f'Need target_shape to reshape flat label of length {N}: {path}')
        h0, w0 = _best_hw_from_N(N, target_shape)
        mC = arr.reshape(h0, w0, order='C')
        mF = arr.reshape(h0, w0, order='F')
        m  = mC if _stripe_score(mC) < _stripe_score(mF) else mF
        Ht, Wt = int(target_shape[0]), int(target_shape[1])
        if (m.shape[0], m.shape[1]) != (Ht, Wt):
            m = cv2.resize(m.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        m = m.astype(np.uint8)

    else:
        raise ValueError(f'Unexpected npz label shape {arr.shape} for {path}')

    # 필요 시: 0→ignore, 1..C→0..C-1
    if treat_zero_as_ignore:
        mm = m.astype(np.int32)
        mm[mm == 0] = ignore_index
        if num_classes is not None:
            mm = np.where(mm == ignore_index, ignore_index, mm - 1)
        m = mm.astype(np.uint8)

    return m

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
        # 기준 크기 결정
        if target_shape is not None:
            ref_h, ref_w = int(target_shape[0]), int(target_shape[1])
        elif img_shape is not None:
            ref_h, ref_w = int(img_shape[0]), int(img_shape[1])
        else:
            raise AssertionError(f"Cannot reshape 1D mask of length {N} without target/img shape")

        if N == ref_h * ref_w:
            # C/F-order 둘 다 시도
            mC = m.reshape((ref_h, ref_w), order='C')
            mF = m.reshape((ref_h, ref_w), order='F')
            def _stripe_score(a):
                return np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum()
            return (mC if _stripe_score(mC) < _stripe_score(mF) else mF).astype(np.uint8)

        # 근사 reshape → 리사이즈
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

# -------- 팔레트/범례/윤곽선 도우미 --------
def _make_palette(n, scheme='bright', seed=0):
    """길이 n의 BGR 색상 리스트 반환."""
    base_bright = [
        (  0,  92, 255), (  0, 255, 255), ( 34, 139,  34), (255,   0,   0),
        (255,   0, 255), (255, 105, 180), (147,  20, 255), ( 60, 179, 113),
        (128, 128,   0), (  0, 215, 255), (180, 130,  70), (203, 192, 255),
        ( 50, 205,  50), (139,   0,   0), (  0, 128, 128), (128,   0, 128),
        (255, 255, 255),
    ]
    tab20 = [
        ( 31, 119, 180), (255, 127,  14), ( 44, 160,  44), (214,  39,  40),
        (148, 103, 189), (140,  86,  75), (227, 119, 194), (127, 127, 127),
        (188, 189,  34), ( 23, 190, 207), (174, 199, 232), (255, 187, 120),
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
            np.full(n, 200),  # S
            np.full(n, 255),  # V
        ], axis=1).astype(np.uint8).reshape(1, n, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
        return [tuple(map(int, c)) for c in bgr]
    else:
        raise ValueError(scheme)
    out = [src[i % len(src)] for i in range(n)]
    return [tuple(map(int, c)) for c in out]

def _draw_legend(canvas_bgr, class_names, palette, max_cols=4):
    """패널 내부 좌상단에 범례(클래스 명+색) 패널."""
    if not class_names: return canvas_bgr
    swatch = 18; pad = 8; txt_h = 16
    cols = min(max_cols, max(1, int(np.ceil(len(class_names) / np.ceil(len(class_names)/max_cols)))))
    rows = int(np.ceil(len(class_names) / cols))
    col_w = 180
    panel_w = cols * col_w + pad * (cols + 1)
    panel_h = rows * (swatch + pad) + pad
    panel = np.full((panel_h, panel_w, 3), 30, np.uint8)
    x = pad; y = pad; i = 0
    for r in range(rows):
        x = pad
        for c in range(cols):
            if i >= len(class_names): break
            color = palette[i] if i < len(palette) else (200, 200, 200)
            cv2.rectangle(panel, (x, y), (x + swatch, y + swatch), color, -1)
            cv2.putText(panel, class_names[i], (x + swatch + 6, y + txt_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
            x += col_w; i += 1
        y += swatch + pad
    H, W = canvas_bgr.shape[:2]
    ph, pw = panel.shape[:2]
    x0, y0 = 10, 10
    x1, y1 = min(x0+pw, W), min(y0+ph, H)
    roi = canvas_bgr[y0:y1, x0:x1]
    panel = panel[:y1-y0, :x1-x0]
    alpha = 0.85
    canvas_bgr[y0:y1, x0:x1] = (alpha*panel + (1-alpha)*roi).astype(np.uint8)
    return canvas_bgr

def _legend_strip(width, class_names, palette, max_cols=4, pad=8, bg=(35,35,35)):
    """패널 '밖'에 붙이는 가로 레전드 띠."""
    if not class_names:
        return np.zeros((1, width, 3), dtype=np.uint8)
    # 한 줄 높이: 스와치 18 + 패딩
    row_h = 26
    rows = (len(class_names) + max_cols - 1) // max_cols
    h = pad*2 + rows*row_h
    strip = np.zeros((h, width, 3), dtype=np.uint8)
    strip[:] = bg
    # 내부 그리기
    swatch = 18; txt_h = 16
    col_w = width // max_cols
    i = 0
    y = pad
    for r in range(rows):
        x = pad
        for c in range(max_cols):
            if i >= len(class_names): break
            color = palette[i] if i < len(palette) else (200,200,200)
            # 좌표가 strip 너비 넘지 않도록 클램프
            x2 = min(x + swatch, width - pad)
            cv2.rectangle(strip, (x, y), (x2, y + swatch), color, -1)
            cv2.putText(strip, class_names[i][:max(4, col_w//8)], (x + swatch + 6, y + txt_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
            x += col_w
            i += 1
        y += row_h
    return strip

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2, ignore_index=255):
    """클래스별 외곽선 그리기."""
    uniq = np.unique(mask)
    for cls in uniq:
        if cls == ignore_index or cls < 0:
            continue
        binm = (mask == cls).astype(np.uint8)*255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        color = palette[cls] if cls < len(palette) else (255,255,255)
        cv2.drawContours(canvas_bgr, cnts, -1, color, thickness, lineType=cv2.LINE_AA)
    return canvas_bgr
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
    # ---- 추가 옵션 ----
    parser.add_argument('--palette', default='bright',
                        choices=['bright', 'tab20', 'random', 'dataset'],
                        help='마스크 색상 팔레트.')
    parser.add_argument('--legend', action='store_true',
                        help='클래스 범례를 표시합니다.')
    parser.add_argument('--legend-pos', default='outside-top',
                        choices=['outside-top', 'outside-bottom', 'inside'],
                        help='레전드 위치')
    parser.add_argument('--legend-cols', type=int, default=6,
                        help='레전드 한 줄당 아이템 수')
    parser.add_argument('--outline', action='store_true',
                        help='클래스 경계선(윤곽선)을 그립니다.')
    parser.add_argument('--seed', type=int, default=0,
                        help='--palette=random 일 때 사용할 시드.')
    # -------------------
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

            # 팔레트 선택: 데이터셋/모델 메타 → 사용자 지정 팔레트
            palette = getattr(dataset, 'PALETTE', None)
            if palette is None:
                palette = getattr(model, 'PALETTE', None) or getattr(getattr(model, 'module', model), 'PALETTE', None)
            # 사용자 팔레트로 덮어쓰기
            if args.palette != 'dataset':
                num_classes = len(getattr(dataset, 'CLASSES', [])) or (len(palette) if palette else 0)
                palette = _make_palette(num_classes, scheme=args.palette, seed=args.seed)
            assert palette is not None, 'PALETTE is required for visualization.'

            class_names = list(getattr(dataset, 'CLASSES', []))
            K = len(class_names)

            N = len(dataset)
            assert isinstance(outputs, list) and len(outputs) == N, \
                f'Unexpected outputs length: {len(outputs)} vs dataset: {N}'

            saved = 0
            for i in range(N):
                img_path, ann_path = _get_img_ann_paths(dataset, i)
                if img_path is None or not osp.exists(img_path):
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
                    pred = _to_2d_mask(pred, target_shape=(baseH, baseW), img_shape=img.shape)
                elif pred.ndim == 2 and (pred.shape[0] != baseH or pred.shape[1] != baseW):
                    pred = _ensure_size(pred, (baseW, baseH))

                pred_color = _colorize(pred, palette)           # (H,W,3)
                if args.vis_mode == 'overlay':
                    pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0)
                else:
                    pred_vis = pred_color

                # 3) Ground Truth  — 항상 dataset API로 (내부가 형식을 제대로 처리함)
                gt_mask = dataset.get_gt_seg_map_by_idx(i)  # (H, W) np.uint8

                # 크기 맞추기
                if gt_mask.shape != (baseH, baseW):
                    gt_mask = cv2.resize(gt_mask, (baseW, baseH), interpolation=cv2.INTER_NEAREST)

                # 시각화용 색
                gt_color   = _colorize(gt_mask, palette)
                gt_overlay = cv2.addWeighted(img, 1.0 - float(args.opacity), gt_color, float(args.opacity), 0)

                # --- Prediction 배경 마스킹 (val/gt 있을 때만) ---
                organ_mask = (gt_mask != 255)
                pred_color[~organ_mask] = 0
                if args.vis_mode == 'overlay':
                    pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0)
                else:
                    pred_vis = pred_color
    
                if gt_mask.shape != (baseH, baseW):
                    gt_mask = cv2.resize(gt_mask, (baseW, baseH), interpolation=cv2.INTER_NEAREST)

                gt_color = _colorize(gt_mask, palette)
                gt_overlay = cv2.addWeighted(img, 1.0 - float(args.opacity), gt_color, float(args.opacity), 0)

                # 윤곽선 옵션 적용
                if args.outline:
                    pred_vis = _draw_class_contours(pred_vis, pred, palette, thickness=2)
                    gt_overlay = _draw_class_contours(gt_overlay, gt_mask, palette, thickness=2)

                # 라벨(제목) 추가 + 순서: Input | Prediction | Ground Truth
                left  = _add_title(img, 'Input')
                mid   = _add_title(pred_vis, 'Prediction (overlay)')
                right = _add_title(gt_overlay, 'Ground Truth (overlay)')

                triptych = cv2.hconcat([left, mid, right])

                # --- 레전드: 패널 밖에 상단/하단 띠로 ---
                if args.legend:
                    if args.legend_pos in ('outside-top', 'outside-bottom'):
                        strip = _legend_strip(triptych.shape[1], class_names, palette,
                                              max_cols=args.legend_cols)
                        if args.legend_pos == 'outside-top':
                            triptych = cv2.vconcat([strip, triptych])
                        else:
                            triptych = cv2.vconcat([triptych, strip])
                    else:
                        triptych = _draw_legend(triptych, class_names, palette, max_cols=args.legend_cols)

                stem = osp.splitext(osp.basename(img_path))[0]
                out_path = osp.join(save_root, f'{i:06d}_{stem}.png')
                mmcv.imwrite(triptych, out_path)
                saved += 1

            print(f'[VIS] saved {saved}/{N} images at: {save_root}')
        # ---------------------------------------------------------------


if __name__ == '__main__':
    main()
