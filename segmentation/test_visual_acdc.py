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
import nibabel as nib
import math

import denseclip  # noqa: F401


# ---------------------- utils ----------------------

def _reduce_zero_label(seg, ignore_index=255):
    seg = seg.astype(np.uint8).copy()
    seg[seg == 0] = ignore_index
    seg = seg - 1
    seg[seg == 254] = ignore_index
    return seg

def _ensure_size(img, size_wh):
    W, H = size_wh
    interp = cv2.INTER_NEAREST if img.ndim == 2 or img.dtype != np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(img, (W, H), interpolation=interp)

def _add_title(img_bgr, title, font_scale=0.9, thickness=2, pad=8):
    h, w = img_bgr.shape[:2]
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_h = th + base + pad * 2
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    y = pad + th   # 베이스라인 고려
    cv2.putText(bar, title, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _safe_join(base, rel):
    if rel is None:
        return None
    if not base:
        return rel
    return rel if osp.isabs(rel) else osp.join(base, rel)

def _resolve_path(base, rel):
    if rel is None:
        return None
    if osp.isabs(rel):
        return rel
    if base:
        try:
            if osp.commonpath([osp.abspath(base), osp.abspath(rel)]) == osp.abspath(base):
                return rel
        except Exception:
            pass
        return osp.join(base, rel)
    return rel

def _get_img_ann_paths(dataset, idx):
    info = dataset.img_infos[idx]
    # image
    rel_img = None
    if 'filename' in info:
        rel_img = info['filename']
    elif 'img_info' in info and isinstance(info['img_info'], dict):
        rel_img = info['img_info'].get('filename')
    elif 'img' in info:
        rel_img = info.get('img')
    # ann
    rel_ann = None
    if 'ann' in info and isinstance(info['ann'], dict):
        rel_ann = info['ann'].get('seg_map')
    elif 'ann_info' in info and isinstance(info['ann_info'], dict):
        rel_ann = info['ann_info'].get('seg_map')

    img_base = getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    ann_base = getattr(dataset, 'ann_dir', None) or getattr(dataset, 'seg_prefix', None)
    data_root = getattr(dataset, 'data_root', None)

    img_path = _resolve_path(img_base, rel_img)
    ann_path = _resolve_path(ann_base, rel_ann)
    if img_path and not osp.isabs(img_path) and data_root:
        img_path = osp.join(data_root, img_path)
    if ann_path and not osp.isabs(ann_path) and data_root:
        ann_path = osp.join(data_root, ann_path)

    if img_path and not osp.exists(img_path) and rel_img and osp.exists(rel_img):
        img_path = rel_img
    if ann_path and not osp.exists(ann_path) and rel_ann and osp.exists(rel_ann):
        ann_path = rel_ann

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

def _to_2d_mask(mask, target_shape=None, img_shape=None):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)
    if m.ndim == 2:
        return m.astype(np.uint8)
    if m.ndim == 1:
        N = m.size
        if target_shape is not None:
            ref_h, ref_w = int(target_shape[0]), int(target_shape[1])
        elif img_shape is not None:
            ref_h, ref_w = int(img_shape[0]), int(img_shape[1])
        else:
            raise AssertionError(f"Cannot reshape 1D mask of length {N} without target/img shape")
        if N == ref_h * ref_w:
            mC = m.reshape((ref_h, ref_w), order='C')
            mF = m.reshape((ref_h, ref_w), order='F')
            return (mC if _stripe_score(mC) < _stripe_score(mF) else mF).astype(np.uint8)
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
            np.full(n, 200),
            np.full(n, 255),
        ], axis=1).astype(np.uint8).reshape(1, n, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
        return [tuple(map(int, c)) for c in bgr]
    else:
        raise ValueError(scheme)
    out = [src[i % len(src)] for i in range(n)]
    return [tuple(map(int, c)) for c in out]

def _draw_legend(canvas_bgr, class_names, palette, max_cols=4):
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
    if not class_names:
        return np.zeros((1, width, 3), dtype=np.uint8)
    row_h = 26
    rows = (len(class_names) + max_cols - 1) // max_cols
    h = pad*2 + rows*row_h
    strip = np.zeros((h, width, 3), dtype=np.uint8)
    strip[:] = bg
    swatch = 18; txt_h = 16
    col_w = width // max_cols
    i = 0
    y = pad
    for r in range(rows):
        x = pad
        for c in range(max_cols):
            if i >= len(class_names): break
            color = palette[i] if i < len(palette) else (200,200,200)
            x2 = min(x + swatch, width - pad)
            cv2.rectangle(strip, (x, y), (x2, y + swatch), color, -1)
            cv2.putText(strip, class_names[i][:max(4, col_w//8)], (x + swatch + 6, y + txt_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
            x += col_w
            i += 1
        y += row_h
    return strip

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2, ignore_index=255):
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

def _load_nii_slice_bgr(path, slice_index=None, target_hw=None):
    vol = nib.load(path).get_fdata()
    vol = np.asarray(vol)
    if vol.ndim == 3:
        s = slice_index if slice_index is not None else vol.shape[-1] // 2
        s = max(0, min(s, vol.shape[-1]-1))
        img2d = vol[..., s]
    elif vol.ndim == 2:
        img2d = vol
    else:
        raise ValueError(f'Unexpected image shape {vol.shape} for {path}')
    vmin, vmax = np.percentile(img2d, [1, 99])
    if vmax <= vmin:
        vmin, vmax = float(img2d.min()), float(img2d.max())
    if vmax > vmin:
        norm = (img2d - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(img2d, dtype=np.float32)
    img8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    if target_hw is not None and (img8.shape[0], img8.shape[1]) != tuple(target_hw):
        H, W = int(target_hw[0]), int(target_hw[1])
        img8 = cv2.resize(img8, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
# ---------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='DenseCLIP ACDC test/eval/visualize')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint (.pth)')
    parser.add_argument('--aug-test', action='store_true')
    parser.add_argument('--out', help='save raw outputs (.pkl/.pickle)')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--eval', type=str, nargs='+')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir', help='dir to save triptychs')
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none','pytorch','slurm','mpi'], default='none')
    parser.add_argument('--opacity', type=float, default=0.6)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--vis-mode', choices=['overlay','mask'], default='overlay')
    parser.add_argument('--palette', default='dataset', choices=['dataset','bright','tab20','random'])
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--legend-pos', default='outside-top',
                        choices=['outside-top','outside-bottom','inside'])
    parser.add_argument('--legend-cols', type=int, default=6)
    parser.add_argument('--outline', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mask-by-gt', action='store_true')
    parser.add_argument('--mask-ignore', type=int, default=255,   # ★ ACDC 배경=255
                        help='GT ignore value (ACDC/Cityscapes: 0 or 255 depending on setup)')
    parser.add_argument('--slice-index', type=int, default=None,
                        help='.nii(.gz)에서 사용할 슬라이스 인덱스 (없으면 중앙)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Specify at least one of --out / --eval / --format-only / --show / --show-dir'

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a .pkl/.pickle')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        for step in cfg.data.test.pipeline:
            if step.get('type') == 'MultiScaleFlipAug':
                step.setdefault('flip', True)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    cfg.model.train_cfg = None
    if isinstance(cfg.model, dict) and 'type' in cfg.model and 'DenseCLIP' in cfg.model['type']:
        cfg.model.class_names = list(getattr(dataset, 'CLASSES', []))
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    ckpt_meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    if not hasattr(model, 'CLASSES') or model.CLASSES is None:
        model.CLASSES = ckpt_meta.get('CLASSES', getattr(dataset, 'CLASSES', None))
    if not hasattr(model, 'PALETTE') or model.PALETTE is None:
        model.PALETTE = ckpt_meta.get('PALETTE', getattr(dataset, 'PALETTE', None))

    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    efficient_test = eval_kwargs.get('efficient_test', False)

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

    rank, _ = get_dist_info()
    if rank != 0:
        return

    print('test done')

    if args.out:
        print(f'\n[IO] writing results to {args.out}')
        mmcv.dump(outputs, args.out)
    if args.format_only:
        dataset.format_results(outputs, **eval_kwargs)
    if args.eval:
        dataset.evaluate(outputs, args.eval, **eval_kwargs)

    if not args.show_dir:
        return
    save_root = osp.abspath(args.show_dir)
    mmcv.mkdir_or_exist(save_root)
    print(f'[VIS] saving triptychs to: {save_root}')

    palette = getattr(dataset, 'PALETTE', None) or getattr(model, 'PALETTE', None)
    if args.palette != 'dataset' or palette is None:
        num_classes = len(getattr(dataset, 'CLASSES', []) or [])
        scheme = 'tab20' if args.palette == 'tab20' else ('random' if args.palette == 'random' else 'bright')
        palette = _make_palette(num_classes, scheme=scheme, seed=args.seed)
    class_names = list(getattr(dataset, 'CLASSES', []) or [])
    N = len(dataset)

    saved = 0
    for i in range(N):
        img_path, _ = _get_img_ann_paths(dataset, i)

        # ---- 1) pred/gt로 우선 H,W 결정 ----
        pred = outputs[i]
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = np.asarray(pred).astype(np.uint8)
        Hp = Wp = None
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 2:
            Hp, Wp = pred.shape

        gt = None
        Hg = Wg = None
        try:
            gt = dataset.get_gt_seg_map_by_idx(i)   # 0~3
            if getattr(dataset, 'reduce_zero_label', False):
                gt = _reduce_zero_label(gt, ignore_index=args.mask_ignore)  # 0~2 + 255
            if gt.ndim == 2:
                Hg, Wg = gt.shape
        except Exception:
            gt = None

        if Hp and Wp:
            H, W = Hp, Wp
        elif Hg and Wg:
            H, W = Hg, Wg
        else:
            H, W = 512, 512

        # pred 크기 맞추기
        if pred.ndim == 1:
            pred = _to_2d_mask(pred, target_shape=(H, W))
        elif pred.ndim == 2 and pred.shape != (H, W):
            pred = _ensure_size(pred, (W, H))

        # ---- 2) 원본 이미지 로드 (.nii면 nib 사용) ----
        img = None
        if img_path:
            lower = img_path.lower()
            if lower.endswith('.nii') or lower.endswith('.nii.gz'):
                try:
                    img = _load_nii_slice_bgr(img_path, slice_index=args.slice_index, target_hw=(H, W))
                except Exception:
                    img = None
            else:
                if osp.exists(img_path):
                    img = mmcv.imread(img_path)
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)

        # gt 크기 보정
        if gt is not None and gt.shape != (H, W):
            gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        # 색칠
        pred_color = _colorize(pred, palette, ignore_index=args.mask_ignore)
        gt_color   = _colorize(gt,   palette, ignore_index=args.mask_ignore) if gt is not None else np.zeros_like(img)

        # overlay
        if args.vis_mode == 'overlay':
            pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0)
            gt_vis   = cv2.addWeighted(img, 1.0 - float(args.opacity), gt_color,   float(args.opacity), 0) \
                       if gt is not None else np.zeros_like(img)
        else:
            pred_vis = pred_color
            gt_vis   = gt_color if gt is not None else np.zeros_like(img)

        # (옵션) GT로 Prediction 배경 마스킹
        pred_for_contour = pred.copy()
        if args.mask_by_gt and gt is not None:
            valid = (gt != args.mask_ignore)
            pc2 = pred_color.copy()
            pc2[~valid] = 0
            pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pc2, float(args.opacity), 0) \
                       if args.vis_mode == 'overlay' else pc2
            pred_for_contour[~valid] = args.mask_ignore

        # 윤곽선
        if args.outline:
            pred_vis = _draw_class_contours(pred_vis, pred_for_contour, palette,
                                            thickness=2, ignore_index=args.mask_ignore)
            if gt is not None:
                gt_vis = _draw_class_contours(gt_vis, gt, palette,
                                              thickness=2, ignore_index=args.mask_ignore)

        # 합치기
        left  = _add_title(img,       'Input')
        mid   = _add_title(pred_vis,  'Prediction' + (' (overlay)' if args.vis_mode=='overlay' else ''))
        right = _add_title(gt_vis,    'Ground Truth' + (' (overlay)' if args.vis_mode=='overlay' else ''))
        trip = cv2.hconcat([left, mid, right])

        # legend
        if args.legend and class_names:
            if args.legend_pos in ('outside-top', 'outside-bottom'):
                strip = _legend_strip(trip.shape[1], class_names, palette, max_cols=args.legend_cols)
                trip = cv2.vconcat([strip, trip]) if args.legend_pos == 'outside-top' else cv2.vconcat([trip, strip])
            else:
                trip = _draw_legend(trip, class_names, palette, max_cols=args.legend_cols)

        stem = osp.splitext(osp.basename(img_path) if img_path else f'img_{i:06d}')[0]
        mmcv.imwrite(trip, osp.join(save_root, f'{i:06d}_{stem}.png'))
        saved += 1

    print(f'[VIS] saved {saved}/{N} images at: {save_root}')


if __name__ == '__main__':
    main()
