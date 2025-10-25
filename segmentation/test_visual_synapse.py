# segmentation/test_visual_synapse.py
#!/usr/bin/env python3
import argparse, os, os.path as osp
import numpy as np, cv2, mmcv, torch, nibabel as nib
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.datasets.synapse import SynapseNiftiDataset
import denseclip  # noqa

def _windowing(x, wmin=-350.0, wmax=350.0):
    x = np.clip(x, wmin, wmax); x = (x - wmin) / (wmax - wmin)
    return (x * 255.0).astype(np.uint8)

def _load_nii_slice_bgr(path, z):
    vol = np.asarray(nib.load(path).get_fdata())
    z = np.clip(z, 0, vol.shape[-1]-1)
    sl = _windowing(vol[..., z].astype(np.float32))
    return cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)

def _colorize(mask, palette, ignore_index=255):
    mask = np.asarray(mask).astype(np.int32)
    h, w = mask.shape; out = np.zeros((h, w, 3), np.uint8); K = len(palette)
    for cls in np.unique(mask):
        if cls < 0 or cls == ignore_index: continue
        color = (0,0,0) if cls >= K else palette[cls]
        out[mask == cls] = color
    return out

def _add_title(img_bgr, title):
    bar = np.zeros((32, img_bgr.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return cv2.vconcat([bar, img_bgr])

def _make_palette(n, scheme='bright', seed=0):
    if scheme == 'bright':
        base = [
            (  0,  92,255), (  0,255,255), ( 34,139, 34), (255,  0,  0),
            (255,  0,255), (255,105,180), (147, 20,255), ( 60,179,113),
            (128,128,  0), (  0,215,255), (180,130, 70), (203,192,255),
            ( 50,205, 50), (139,  0,  0), (  0,128,128), (128,  0,128),
            (255,255,255),
        ]
        return [base[i % len(base)] for i in range(n)]
    if scheme == 'tab20':
        tab20 = [
            ( 31,119,180),(255,127, 14),( 44,160, 44),(214, 39, 40),
            (148,103,189),(140, 86, 75),(227,119,194),(127,127,127),
            (188,189, 34),( 23,190,207),(174,199,232),(255,187,120),
            (152,223,138),(255,152,150),(197,176,213),(196,156,148),
            (247,182,210),(199,199,199),(219,219,141),(158,218,229),
        ]
        return [tab20[i % len(tab20)] for i in range(n)]
    # random
    rng = np.random.RandomState(seed)
    hsv = np.stack([rng.permutation(np.linspace(0,179,n,endpoint=False)),
                    np.full(n,200), np.full(n,255)], axis=1).astype(np.uint8)[None]
    return [tuple(map(int,c)) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]

def _draw_legend_strip(width, class_names, palette, cols=7, pad=8, bg=(35,35,35)):
    if not class_names: return np.zeros((1,width,3),np.uint8)
    row_h=26; rows=(len(class_names)+cols-1)//cols; h=pad*2+rows*row_h
    strip=np.full((h,width,3),bg,np.uint8); sw=18; col_w=width//cols; i=0; y=pad
    for r in range(rows):
        x=pad
        for c in range(cols):
            if i>=len(class_names): break
            color = palette[i] if i < len(palette) else (200,200,200)
            cv2.rectangle(strip,(x,y),(min(x+sw,width-pad),y+sw),color,-1)
            cv2.putText(strip,class_names[i],(x+sw+6,y+16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(240,240,240),1,cv2.LINE_AA)
            x+=col_w; i+=1
        y+=row_h
    return strip

def _draw_class_contours(canvas_bgr, mask, palette, thickness=2, ignore_index=255):
    for cls in np.unique(mask):
        if cls == ignore_index or cls < 0: continue
        m=(mask==cls).astype(np.uint8)*255
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        color = palette[cls] if cls < len(palette) else (255,255,255)
        cv2.drawContours(canvas_bgr,cnts,-1,color,thickness,lineType=cv2.LINE_AA)
    return canvas_bgr

def parse_args():
    ap = argparse.ArgumentParser("Visualize Synapse inference (color overlay)")
    ap.add_argument('config'); ap.add_argument('checkpoint')
    ap.add_argument('--show-dir', default='vis/synapse')
    ap.add_argument('--opacity', type=float, default=0.65)
    ap.add_argument('--palette', choices=['dataset','bright','tab20','random'], default='bright')
    ap.add_argument('--legend', action='store_true')
    ap.add_argument('--legend-cols', type=int, default=7)
    ap.add_argument('--outline', action='store_true')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained=None
    cfg.data.test.test_mode=True

    # DenseCLIP은 class_names 필수
    if 'DenseCLIP' in cfg.model.get('type','') and not cfg.model.get('class_names'):
        cfg.model['class_names'] = list(SynapseNiftiDataset.CLASSES)
        print(f"[INFO] Injected class_names ({len(cfg.model['class_names'])}).")

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1,
                              workers_per_gpu=cfg.data.workers_per_gpu,
                              shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = ckpt.get('meta',{}).get('CLASSES', getattr(dataset,'CLASSES',None))
    model.PALETTE = ckpt.get('meta',{}).get('PALETTE', getattr(dataset,'PALETTE',None))
    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, loader, show=False)

    save_root = osp.abspath(args.show_dir); mmcv.mkdir_or_exist(save_root)
    print(f"[VIS] Saving to: {save_root}")

    # 팔레트 선정: dataset/denseclip 메타 대신 사용자가 선택한 컬러로 덮어쓰기
    class_names = list(getattr(dataset, 'CLASSES', []))
    if args.palette == 'dataset' and getattr(dataset,'PALETTE',None):
        palette = dataset.PALETTE
    else:
        palette = _make_palette(len(class_names), scheme=args.palette)

    for i, info in enumerate(dataset.img_infos):
        img_path = info['img_info']['filename']
        ann_path = info['ann_info']['seg_map']
        z        = info['img_info']['z_index']

        img = _load_nii_slice_bgr(img_path, z)
        pred = outputs[i][0] if isinstance(outputs[i], (list, tuple)) else outputs[i]
        pred = np.asarray(pred).astype(np.uint8)
        gt   = np.asarray(nib.load(ann_path).get_fdata()).astype(np.int32)[..., z].astype(np.uint8)

        H,W = img.shape[:2]
        if pred.ndim==2 and pred.shape!=(H,W): pred = cv2.resize(pred,(W,H),interpolation=cv2.INTER_NEAREST)
        if gt.shape!=(H,W): gt = cv2.resize(gt,(W,H),interpolation=cv2.INTER_NEAREST)

        pred_c = _colorize(pred, palette); gt_c = _colorize(gt, palette)
        pred_o = cv2.addWeighted(img, 1.0-args.opacity, pred_c, args.opacity, 0)
        gt_o   = cv2.addWeighted(img, 1.0-args.opacity, gt_c,   args.opacity, 0)

        if args.outline:
            pred_o = _draw_class_contours(pred_o, pred, palette)
            gt_o   = _draw_class_contours(gt_o,   gt,   palette)

        trip = cv2.hconcat([_add_title(img,'Input'),
                            _add_title(pred_o,'Prediction'),
                            _add_title(gt_o,'Ground Truth')])

        if args.legend and class_names:
            strip = _draw_legend_strip(trip.shape[1], class_names, palette, cols=args.legend_cols)
            trip  = cv2.vconcat([strip, trip])

        out_path = osp.join(save_root, f'{i:06d}_z{z}.png')
        cv2.imwrite(out_path, trip)

    print(f"[DONE] Saved {len(dataset)} slices to: {save_root}")

if __name__ == "__main__":
    main()
