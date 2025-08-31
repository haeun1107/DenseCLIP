# segmentation/test_visual_braTS.py
import argparse, os, os.path as osp, numpy as np, cv2, mmcv, torch, nibabel as nib
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import denseclip  # noqa: F401


# ---------------------- helpers ----------------------
def _make_palette(n, scheme='bright', seed=0):
    base_bright = [(0,92,255),(0,255,255),(34,139,34),(255,0,0),(255,0,255),(255,105,180),
                   (147,20,255),(60,179,113),(128,128,0),(0,215,255),(180,130,70),(203,192,255),
                   (50,205,50),(139,0,0),(0,128,128),(128,0,128),(255,255,255)]
    if scheme=='bright':
        src = base_bright
        return [tuple(map(int, src[i%len(src)])) for i in range(n)]
    elif scheme=='tab20':
        src=[(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),
             (227,119,194),(127,127,127),(188,189,34),(23,190,207),(174,199,232),(255,187,120),
             (152,223,138),(255,152,150),(197,176,213),(196,156,148),(247,182,210),(199,199,199),
             (219,219,141),(158,218,229)]
        return [tuple(map(int, src[i%len(src)])) for i in range(n)]
    else:
        rng = np.random.RandomState(seed)
        hsv = np.stack([rng.permutation(np.linspace(0,179,n,endpoint=False)),
                        np.full(n,200), np.full(n,255)], 1).astype(np.uint8)[None]
        return [tuple(map(int,c)) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]

def _load_nii_slice_bgr(nii_path, z, target_hw=None):
    vol = np.asarray(nib.load(nii_path).get_fdata())  # (H,W,S)
    z = int(np.clip(z, 0, vol.shape[-1]-1))
    sl = vol[..., z]
    vmin, vmax = np.percentile(sl, [1,99])
    if vmax <= vmin: vmin, vmax = float(sl.min()), float(sl.max())
    norm = (sl - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(sl, dtype=np.float32)
    img8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    if target_hw is not None and img8.shape != tuple(target_hw):
        H, W = target_hw
        img8 = cv2.resize(img8, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

def _load_nii_label_slice(nii_path, z):
    vol = np.asarray(nib.load(nii_path).get_fdata()).astype(np.int32)  # (H,W,S)
    z = int(np.clip(z, 0, vol.shape[-1]-1))
    sl = vol[..., z]
    sl[sl == 4] = 3  # BraTS 표준: 4(ET) -> 3
    return sl.astype(np.uint8)

def _colorize(mask, palette):
    if mask is None: return None
    h, w = mask.shape
    out = np.zeros((h, w, 3), np.uint8)
    for cls_idx in np.unique(mask):
        if cls_idx < 0: continue
        color = palette[cls_idx] if cls_idx < len(palette) else (255,255,255)
        out[mask == cls_idx] = color
    return out

def _add_title(img_bgr, title):
    h, w = img_bgr.shape[:2]
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    bar = np.zeros((th + base + 16, w, 3), np.uint8)
    cv2.putText(bar, title, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2):
    uniq = np.unique(mask)
    for cls in uniq:
        if cls <= 0:  # background 제외
            continue
        binm = (mask == cls).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        color = palette[cls] if cls < len(palette) else (255,255,255)
        cv2.drawContours(canvas_bgr, cnts, -1, color, thickness, lineType=cv2.LINE_AA)
    return canvas_bgr


# ---------------------- args ----------------------
def parse_args():
    p = argparse.ArgumentParser(description='DenseCLIP BraTS test/eval/visualize')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint (.pth)')
    p.add_argument('--aug-test', action='store_true')
    p.add_argument('--out', help='save raw outputs (.pkl/.pickle)')
    p.add_argument('--format-only', action='store_true')
    p.add_argument('--eval', type=str, nargs='+')  # e.g., mIoU mDice
    p.add_argument('--show-dir', help='dir to save visualizations')
    p.add_argument('--launcher', choices=['none','pytorch','slurm','mpi'], default='none')
    p.add_argument('--opacity', type=float, default=0.6)
    p.add_argument('--vis-mode', choices=['overlay','mask'], default='overlay')
    p.add_argument('--legend', action='store_true')
    p.add_argument('--legend-cols', type=int, default=4)
    p.add_argument('--palette', default='dataset', choices=['dataset','bright','tab20','random'])
    p.add_argument('--keep-labeled-only', action='store_true',
                   help='GT 또는 Pred에 양성 픽셀이 있는 슬라이스만 저장')
    p.add_argument('--min-pixels', type=int, default=50,
                   help='양성 픽셀 최소 개수(그 미만이면 스킵)')
    p.add_argument('--outline', action='store_true', help='클래스 경계 윤곽선 그리기')
    p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


# ---------------------- main ----------------------
def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show_dir, \
        'Specify at least one of --out / --eval / --format-only / --show-dir'

    cfg = mmcv.Config.fromfile(args.config)

    # aug-test flip 켜기
    if args.aug_test:
        for step in cfg.data.test.pipeline:
            if step.get('type') == 'MultiScaleFlipAug':
                step.setdefault('flip', True)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # Dataset & DataLoader (→ dataset.CLASSES / PALETTE 사용)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )

    # DenseCLIP이 class_names 필수인 경우 대비: 없으면 dataset.CLASSES 주입
    if isinstance(cfg.model, dict) and 'class_names' not in cfg.model:
        ds_classes = getattr(dataset, 'CLASSES', None)
        if ds_classes:
            cfg.model['class_names'] = tuple(ds_classes)

    # 모델
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # CLASSES / PALETTE
    ckpt_meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
    classes = getattr(dataset, 'CLASSES', None) or ckpt_meta.get('CLASSES', [])
    palette = getattr(dataset, 'PALETTE', None) or ckpt_meta.get('PALETTE', None)
    if palette is None or args.palette != 'dataset':
        scheme = 'tab20' if args.palette=='tab20' else ('random' if args.palette=='random' else 'bright')
        palette = _make_palette(len(classes) if classes else 4, scheme=scheme, seed=args.seed)

    # 추론
    torch.cuda.empty_cache()
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False, out_dir=None,
                                  efficient_test=False, opacity=args.opacity)
    else:
        model = MMDistributedDataParallel(model.cuda(),
                                          device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, tmpdir=None,
                                 gpu_collect=False, efficient_test=False)

    rank, _ = get_dist_info()
    if rank != 0:
        return

    if args.out:
        mmcv.dump(outputs, args.out)
    if args.format_only:
        dataset.format_results(outputs)
    if args.eval:
        dataset.evaluate(outputs, args.eval)

    if not args.show_dir:
        print('test done (no visualization dir).'); return

    save_root = osp.abspath(args.show_dir)
    mmcv.mkdir_or_exist(save_root)
    print(f'[VIS] saving to: {save_root}')

    # 저장 루프
    N = len(dataset)
    saved = 0
    for i in range(N):
        info = dataset.img_infos[i]
        img_path = info['img_info']['filename']   # 절대 경로(.nii/.nii.gz)
        z_idx    = info['img_info']['z_index']
        ann      = info.get('ann_info', {})
        seg_path = ann.get('seg_map', None)

        # pred
        pred = outputs[i]
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = np.asarray(pred).astype(np.uint8)
        if pred.ndim == 3 and pred.shape[0] == 1: pred = pred[0]
        assert pred.ndim == 2, f'pred must be 2D, got {pred.shape}'
        H, W = pred.shape

        # input slice
        img = _load_nii_slice_bgr(img_path, z_idx, target_hw=(H, W))

        # GT (있을 때만)
        gt = None
        if seg_path and osp.exists(seg_path):
            gt = _load_nii_label_slice(seg_path, z_idx)
            if gt.shape != (H, W):
                gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        # 라벨 슬라이스만 저장 (옵션)
        pos_pred = int((pred > 0).sum())
        pos_gt   = int((gt > 0).sum()) if gt is not None else 0
        if args.keep_labeled_only and (pos_pred < args.min_pixels and pos_gt < args.min_pixels):
            continue

        # colorize / overlay
        pred_color = _colorize(pred, palette)
        pred_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), pred_color, float(args.opacity), 0) \
                   if args.vis_mode=='overlay' else pred_color

        if gt is not None:
            gt_color = _colorize(gt, palette)
            gt_vis = cv2.addWeighted(img, 1.0 - float(args.opacity), gt_color, float(args.opacity), 0) \
                     if args.vis_mode=='overlay' else gt_color
        else:
            gt_vis = np.zeros_like(img)

        # 윤곽선(옵션)
        if args.outline:
            pred_vis = _draw_class_contours(pred_vis, pred, palette, thickness=2)
            if gt is not None:
                gt_vis = _draw_class_contours(gt_vis, gt, palette, thickness=2)

        # triptych: Input | Pred | GT
        left  = _add_title(img,      f'Input (FLAIR) z={z_idx}')
        mid   = _add_title(pred_vis, 'Prediction')
        right = _add_title(gt_vis,   'Ground Truth' if gt is not None else 'Ground Truth (N/A)')
        trip  = cv2.hconcat([left, mid, right])

        # legend (옵션)
        if args.legend and classes:
            strip_h = 30
            rows = int(np.ceil(len(classes) / float(max(1, args.legend_cols))))
            strip = np.full((strip_h*rows, trip.shape[1], 3), 35, np.uint8)
            sw, pad = 18, 8
            k = 0
            for r in range(rows):
                x = pad; y = r*strip_h + pad
                col_w = trip.shape[1] // max(1, args.legend_cols)
                for c in range(max(1, args.legend_cols)):
                    if k >= len(classes): break
                    color = palette[k] if k < len(palette) else (200,200,200)
                    cv2.rectangle(strip, (x, y), (x+sw, y+sw), color, -1)
                    cv2.putText(strip, classes[k], (x+sw+6, y+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
                    x += col_w; k += 1
            trip = cv2.vconcat([strip, trip])

        stem = osp.splitext(osp.basename(img_path))[0].replace('_flair', '')
        out_path = osp.join(save_root, f'{i:06d}_{stem}_z{z_idx}.png')
        mmcv.imwrite(trip, out_path)
        saved += 1

    print(f'[VIS] saved {saved}/{N} images at: {save_root}')


if __name__ == '__main__':
    main()
