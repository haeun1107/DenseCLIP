#!/usr/bin/env python3
"""
Generate pseudo labels as NIfTI (.nii/.nii.gz) using a DenseCLIP A0 checkpoint.
- Keeps the SAME relative path & filename as the input image
  e.g., data/ACDC/testing/patient101/patient101_frame01.nii.gz
  -> out_dir/patient101/patient101_frame01_gt.nii.gz

Supports confidence thresholding:
  --threshold 0.5 --ignore-index 255
will set any pixel whose max-class score < 0.5 to ignore_index (e.g., 255).
"""
import argparse, os, os.path as osp, numpy as np, nibabel as nib
import mmcv, torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
import denseclip  # noqa


def parse_args():
    p = argparse.ArgumentParser("DenseCLIP A0 → NIfTI pseudo writer")
    p.add_argument('config', help='config .py')
    p.add_argument('checkpoint', help='.pth')
    p.add_argument('--out-dir', required=True, help='root dir to save NIfTI pseudos')
    p.add_argument(
        '--slice-index',
        type=int,
        default=None,
        help='if LoadNiftiImageFromFile sees 3D, pick this slice (default: center)'
    )
    p.add_argument(
        '--dtype',
        default='uint8',
        choices=['uint8', 'uint16', 'int16', 'int32'],
        help='NIfTI integer dtype for labels'
    )
    p.add_argument(
        '--keep-prob',
        action='store_true',
        help='(ignored for NIfTI) kept for API compatibility'
    )
    # 🔹 추가: threshold & ignore-index
    p.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='if set, pixels with max score < threshold are set to ignore_index'
    )
    p.add_argument(
        '--ignore-index',
        type=int,
        default=255,
        help='label value used to mark ignored pixels (e.g., 255 for mmseg)'
    )
    return p.parse_args()


# --- helpers ---------------------------------------------------------
def _ensure_dir(path):  # create parent
    os.makedirs(path, exist_ok=True)


def _argmax_to_label_and_prob(arr):
    """Accepts (C,H,W)/(H,W,C) or (H,W). Returns (label2d, prob3d or None, H,W,C).

    prob3d is always shaped (C,H,W) if not None.
    """
    a = np.asarray(arr)
    # Already 2D label map
    if a.ndim == 2:
        H, W = a.shape
        return a.astype(np.int32), None, H, W, 1

    if a.ndim == 3:
        # (C,H,W)
        if a.shape[0] < a.shape[-1]:
            lab = a.argmax(0)
            C, H, W = a.shape
            prob3d = a  # (C,H,W)
        else:  # (H,W,C)
            lab = a.argmax(-1)
            H, W, C = a.shape
            prob3d = np.transpose(a, (2, 0, 1))  # (C,H,W)
        return lab.astype(np.int32), prob3d, H, W, C

    # tuple/dict 등은 mmseg가 이미 ndarray로 정리해 줌
    a = np.squeeze(a)
    assert a.ndim == 2, f"Unexpected pred shape {a.shape}"
    H, W = a.shape
    return a.astype(np.int32), None, H, W, 1


def _rel_img_path(dataset, idx):
    """Return relative image path WITH extension like 'patient101/patient101_frame01.nii.gz'."""
    info = None
    for key in ('img_infos', 'data_infos'):
        if hasattr(dataset, key):
            lst = getattr(dataset, key)
            if 0 <= idx < len(lst):
                info = lst[idx]
                break
    rel = None
    if isinstance(info, dict):
        rel = (info.get('img_info', {}) or {}).get('filename') \
              or info.get('filename') or info.get('file_name')
    if not rel:
        return f'{idx:06d}.nii.gz'
    return rel.replace('\\', '/')


def _abs_img_path(dataset, idx, rel_path):
    """Build absolute path to source image to copy affine/header."""
    # Prefer dataset.img_prefix / img_dir if available
    img_prefix = getattr(dataset, 'img_prefix', None) or getattr(dataset, 'img_dir', None)
    if img_prefix and not osp.isabs(rel_path):
        return osp.join(img_prefix, rel_path)
    return rel_path


def _save_as_nifti_like(src_img_path, label2d, out_path, dtype='uint8'):
    """Write 2D label map as NIfTI using src's affine+header."""
    # load src for affine/header
    try:
        nsrc = nib.load(src_img_path)
        affine = nsrc.affine
        header = nsrc.header.copy()
    except Exception:
        # fallback
        affine = np.eye(4)
        header = nib.Nifti1Header()
    # enforce integer label dtype
    np_dtype = np.dtype(dtype)
    data = np.asarray(label2d, dtype=np_dtype)

    # NIfTI expects at least 2D (H,W)
    nim = nib.Nifti1Image(data, affine, header)
    _ensure_dir(osp.dirname(out_path))
    nib.save(nim, out_path)


# --- main ------------------------------------------------------------
def main():
    args = parse_args()
    out_root = args.out_dir
    mmcv.mkdir_or_exist(out_root)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # override slice index if given
    if args.slice_index is not None:
        for t in cfg.data.test.pipeline:
            if isinstance(t, dict) and t.get('type') == 'LoadNiftiImageFromFile':
                t['slice_index'] = args.slice_index

    # dataset
    dataset = build_dataset(cfg.data.test)

    # DenseCLIP needs class_names at build-time
    if 'class_names' not in cfg.model or cfg.model['class_names'] is None:
        acdc_default = ['background', 'right_ventricle', 'myocardium', 'left_ventricle']
        cfg.model['class_names'] = list(getattr(dataset, 'CLASSES', acdc_default))

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=getattr(cfg.data, 'workers_per_gpu', 2),
        dist=False,
        shuffle=False
    )

    # model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    # infer
    results = single_gpu_test(model, data_loader, show=False)

    # save loop
    saved = 0
    warned_no_prob = False

    for i, pred in enumerate(results):
        label2d, prob3d, H, W, C = _argmax_to_label_and_prob(pred)

        # 🔹 threshold + ignore 적용
        if args.threshold is not None:
            if prob3d is not None:
                # prob3d: (C,H,W) 형태의 score/확률 맵이라고 가정
                conf_map = prob3d.max(axis=0)  # (H,W)
                low_conf_mask = conf_map < args.threshold
                # background(0) 대신 ignore-index로 채우기
                label2d[low_conf_mask] = args.ignore_index
            else:
                if not warned_no_prob:
                    print(
                        "[WARN] --threshold is set but model outputs "
                        "no per-class scores (only label map). "
                        "Threshold will have no effect for these outputs."
                    )
                    warned_no_prob = True

        rel_img = _rel_img_path(dataset, i)  # 'patient101/patient101_frame01.nii.gz'
        src_abs = _abs_img_path(dataset, i, rel_img)

        # === 🔹 _gt 붙이기 ===
        base, ext = osp.splitext(rel_img)
        if ext == ".gz":  # handle .nii.gz
            base, _ = osp.splitext(base)
            ext = ".nii.gz"
        out_rel = base + "_gt" + ext               # e.g., patient101_frame01_gt.nii.gz
        out_path = osp.join(out_root, out_rel)

        _save_as_nifti_like(src_abs, label2d, out_path, dtype=args.dtype)
        saved += 1

    print(f"[NIfTI] Saved {saved} files to: {osp.abspath(out_root)}")


if __name__ == '__main__':
    main()

python 
  segmentation/generate_nifti_A0.py \
  work_dirs/denseclip_acdc/latest.pth \
  --out-dir pseudos_acdc \
  --threshold 0.5 \
  --ignore-index 255