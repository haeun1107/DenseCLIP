# segmentation/test_visual_btcv_pseudo.py
# Visualize NPZ predictions as colorful overlays on original images (no model build needed).
import argparse, os, os.path as osp, numpy as np, cv2, mmcv, torch
from mmcv.utils import DictAction
from mmseg.datasets import build_dataset

IGNORE = 255

# ---------- Color helpers ----------
def make_palette_vivid(n: int):
    """evenly spaced hues, max saturation/value -> vivid, easily distinguishable."""
    if n <= 0: return []
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.linspace(0, 179, n, endpoint=False)).astype(np.uint8)  # H
    hsv[:, 0, 1] = 255  # S
    hsv[:, 0, 2] = 255  # V
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]
    return [tuple(map(int, c)) for c in bgr]

def make_palette_random(n: int, seed: int = 0):
    if n <= 0: return []
    rng = np.random.RandomState(seed)
    hues = rng.permutation(np.linspace(0, 179, n, endpoint=False)).astype(np.uint8)
    hsv = np.stack([hues, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], axis=1).reshape(n,1,3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:,0,:]
    return [tuple(map(int, c)) for c in bgr]

def colorize(mask, palette):
    m = mask.astype(np.int32)
    out = np.zeros((*m.shape, 3), np.uint8)
    K = len(palette)
    for c in np.unique(m):
        if c == IGNORE or c < 0: continue
        out[m == c] = palette[c] if c < K else (255,255,255)
    return out
# -----------------------------------

def ensure_size(arr, wh):
    W, H = wh
    inter = cv2.INTER_NEAREST if arr.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(arr, (W, H), interpolation=inter)

def load_npz_label(p):
    a = np.load(p)
    k = 'pred' if 'pred' in a.files else (a.files[0] if a.files else 'arr_0')
    x = np.asarray(a[k])
    if x.ndim == 2: return x.astype(np.uint8)
    if x.ndim == 3 and 1 in x.shape: return np.squeeze(x).astype(np.uint8)
    if x.ndim == 3:
        # one-hot/logits -> argmax
        return (x.argmax(0) if x.shape[0] <= x.shape[-1] and x.shape[0] < 1024
                else x.argmax(-1)).astype(np.uint8)
    raise ValueError(f'unexpected npz shape {x.shape} @ {p}')

def stem(p): return osp.splitext(osp.basename(p))[0]

def get_img_path(dataset, i):
    info = dataset.img_infos[i]
    img_info = info.get('img_info', {})
    rel = img_info.get('filename') or info.get('filename')
    pref = img_info.get('img_prefix') or getattr(dataset, 'img_dir', None) or getattr(dataset, 'img_prefix', None)
    if rel is None: return None
    return rel if osp.isabs(rel) or not pref else osp.join(pref, rel)

def draw_outlines(canvas_bgr, lab, palette, thick=2):
    uniq = np.unique(lab)
    for cls in uniq:
        if cls == IGNORE or cls < 0: continue
        binm = (lab == cls).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        col = palette[cls] if cls < len(palette) else (255,255,255)
        cv2.drawContours(canvas_bgr, cnts, -1, col, thickness=max(1, int(thick)), lineType=cv2.LINE_AA)
    return canvas_bgr

# ===== 루프 안에서, overlay 만들기 직전 추가 =====
if args.mask_by_gt:
    gt = dataset.get_gt_seg_map_by_idx(i).astype(np.uint8)  # (H,W)
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    # GT 배경(255)인 곳은 예측도 255(IGNORE)로 만들어서 색 안 칠함
    lab = lab.copy()
    lab[gt == 255] = IGNORE
    
def parse_args():
    p = argparse.ArgumentParser('Overlay NPZ predictions (colorful) on original images')
    p.add_argument('config', help='mmseg test config (for dataset paths)')
    p.add_argument('--checkpoint', required=True, help='to read CLASSES/PALETTE meta only')
    p.add_argument('--npz-dir', required=True, help='per-image .npz dir (key=pred/arr_0)')
    p.add_argument('--save-dir', required=True, help='output dir for overlays')
    p.add_argument('--opacity', type=float, default=0.7, help='mask weight (0~1)')
    p.add_argument('--map-zero-to-255', action='store_true', help='treat label 0 as ignore(255) for display')
    p.add_argument('--limit', type=int, default=0, help='visualize first N only (0=all)')
    p.add_argument('--cfg-options', nargs='+', action=DictAction, help='override cfg values')

    # colorful options
    p.add_argument('--palette', default='vivid', choices=['dataset', 'vivid', 'random'])
    p.add_argument('--seed', type=int, default=0, help='seed for --palette=random')
    p.add_argument('--bg-dim', type=float, default=0.35, help='darken background (0~0.95)')
    p.add_argument('--outline', action='store_true')
    p.add_argument('--outline-thick', type=int, default=3)
    # ===== 추가: argparse 옵션 =====
    p.add_argument('--mask-by-gt', action='store_true',
               help='GT가 255인 배경을 색칠하지 않도록 예측에 마스킹')

    return p.parse_args()


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options: cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # ---- Fetch class names & palette from checkpoint meta (no model build) ----
    class_names = list(getattr(dataset, 'CLASSES', []))
    palette = getattr(dataset, 'PALETTE', None)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    meta = ckpt.get('meta', {})
    class_names = meta.get('CLASSES', class_names)
    palette = meta.get('PALETTE', palette)

    # ---- Palette selection / override ----
    if args.palette == 'dataset' and palette is None:
        # fall back to vivid if dataset palette is missing
        palette = make_palette_vivid(len(class_names))
    elif args.palette == 'vivid':
        palette = make_palette_vivid(len(class_names))
    elif args.palette == 'random':
        palette = make_palette_random(len(class_names), seed=args.seed)

    assert palette is not None, 'PALETTE not found or constructed.'

    os.makedirs(args.save_dir, exist_ok=True)

    # index npz by stem
    npz_map = {stem(fn): osp.join(args.npz_dir, fn)
               for fn in os.listdir(args.npz_dir)
               if fn.lower().endswith('.npz')}

    N = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    saved = 0
    for i in range(N):
        ipath = get_img_path(dataset, i)
        if not ipath or not osp.exists(ipath): continue
        s = stem(ipath)
        npath = npz_map.get(s)
        if npath is None:
            # try prefix match (rare naming differences)
            cands = [p for k,p in npz_map.items() if k.startswith(s)]
            npath = cands[0] if cands else None
        if npath is None:
            print(f'[WARN] npz not found for {s}, skip'); continue

        img = mmcv.imread(ipath)  # BGR
        H, W = img.shape[:2]

        # slightly darken the background to pop colors
        if 0.0 < args.bg_dim < 0.95:
            img = (img.astype(np.float32) * (1.0 - float(args.bg_dim))).clip(0, 255).astype(np.uint8)

        lab = load_npz_label(npath)
        if lab.shape != (H, W):
            lab = ensure_size(lab, (W, H)).astype(np.uint8)

        if args.map_zero_to_255:
            x = lab.astype(np.int32); x[x == 0] = IGNORE; lab = x.astype(np.uint8)

        mask_rgb = colorize(lab, palette)
        overlay = cv2.addWeighted(img, 1.0 - float(args.opacity), mask_rgb, float(args.opacity), 0)

        if args.outline:
            overlay = draw_outlines(overlay, lab, palette, thick=args.outline_thick)

        out = osp.join(args.save_dir, f'{i:06d}_{s}.png')
        mmcv.imwrite(overlay, out)
        saved += 1

    print(f'[OVERLAY] saved {saved} images → {osp.abspath(args.save_dir)}')

if __name__ == '__main__':
    main()
