#!/usr/bin/env python3
"""
Generate pseudo labels as NIfTI volumes (3D) from a DenseCLIP checkpoint.
- Groups per-volume slice predictions and writes one NIfTI per case:
    data/synapse/{train|val}/GT/label0001.nii.gz  ->  out_dir/{train|val}/GT/label0001.nii.gz
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
    p = argparse.ArgumentParser("DenseCLIP → NIfTI(3D) pseudo writer")
    p.add_argument('config', help='config .py')
    p.add_argument('checkpoint', help='.pth')
    p.add_argument('--out-dir', required=True, help='root dir to save NIfTI pseudos')
    p.add_argument('--dtype', default='uint8',
                  choices=['uint8','uint16','int16','int32'],
                  help='NIfTI integer dtype for labels')
    p.add_argument('--split-tag', default=None,
                  help='Optional subdir name to insert before GT (e.g., train/ or val/). '
                       'If omitted, we infer from original seg_map path.')
    return p.parse_args()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _infer_split_and_rel_gt(seg_path):
    """
    From an absolute seg_map like:
        .../data/synapse/train/GT/label0001.nii.gz
    return:
        split='train', rel_in_gt='label0001.nii.gz'
    """
    # normalize
    sp = seg_path.replace('\\', '/')
    parts = sp.split('/')
    # find ".../<split>/GT/<file>"
    if 'GT' in parts:
        gt_idx = parts.index('GT')
        if gt_idx >= 1:
            split = parts[gt_idx - 1]  # train or val
        else:
            split = None
        rel_in_gt = '/'.join(parts[gt_idx + 1:])  # file under GT/
        return split, rel_in_gt
    # fallback
    return None, osp.basename(seg_path)


def _alloc_volume_like(src_nifti_path, dtype):
    """Allocate empty 3D volume using shape/affine/header of the source image."""
    nimg = nib.load(src_nifti_path)
    # expect (H, W, S)
    shape = nimg.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape={shape} at {src_nifti_path}")
    vol = np.zeros(shape, dtype=np.dtype(dtype))  # background=0
    return vol, nimg.affine, nimg.header.copy()


def _save_volume(out_path, vol, affine, header):
    _ensure_dir(osp.dirname(out_path))
    nim = nib.Nifti1Image(vol, affine, header)
    nib.save(nim, out_path)


def main():
    args = parse_args()
    out_root = args.out_dir
    mmcv.mkdir_or_exist(out_root)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # dataset & dataloader
    dataset = build_dataset(cfg.data.test)

    # DenseCLIP needs class_names at build-time
    if 'class_names' not in cfg.model or cfg.model['class_names'] is None:
        cfg.model['class_names'] = list(getattr(dataset, 'CLASSES', [])) or \
            ['background']  # minimal default

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

    # infer (slice-wise predictions)
    results = single_gpu_test(model, data_loader, show=False)

    # --- group by volume (keyed by ann_info['seg_map']) ---
    groups = {}  # seg_path -> dict(z -> pred2d)
    meta = {}    # seg_path -> dict(src_img=..., seg=..., split=..., rel_in_gt=...)
    for i, pred in enumerate(results):
        info = dataset.img_infos[i]
        z = info['img_info']['z_index']
        img_path = info['img_info']['filename']   # .../CT/img0001.nii.gz
        seg_path = info['ann_info']['seg_map']    # .../GT/label0001.nii.gz

        # argmax to class id map (H,W). DenseCLIP returns per-pixel classes already or logits.
        arr = np.asarray(pred)
        if arr.ndim == 3:  # (C,H,W) or (H,W,C)
            if arr.shape[0] < arr.shape[-1]:
                lab2d = arr.argmax(0)
            else:
                lab2d = arr.argmax(-1)
        else:
            lab2d = arr.astype(np.int64)  # already (H,W)

        # map back to Synapse label IDs: 0..12 -> 1..13
        lab2d = (lab2d + 1).astype(np.int32)

        if seg_path not in groups:
            groups[seg_path] = {}
            sp_split, rel_gt = _infer_split_and_rel_gt(seg_path)
            if args.split_tag is not None:
                sp_split = args.split_tag  # manual override
            meta[seg_path] = dict(src_img=img_path, seg=seg_path,
                                  split=sp_split, rel_in_gt=rel_gt)

        groups[seg_path][z] = lab2d

    # --- save each volume once ---
    saved = 0
    for seg_path, zdict in groups.items():
        m = meta[seg_path]
        split = m['split'] or 'train'  # default
        rel_in_gt = m['rel_in_gt']     # e.g., 'label0001.nii.gz'

        # allocate empty volume like the **image** volume (CT) to inherit affine/header
        # (GT가 없을 수도 있으니 CT를 기준으로 잡는 게 안전)
        ct_path = m['src_img']
        vol, affine, header = _alloc_volume_like(ct_path, dtype=args.dtype)

        # fill predicted slices
        for z, lab2d in zdict.items():
            vol[..., z] = lab2d  # background(0) 유지, organs=1..13

        # build output path: out_root/<split>/GT/labelXXXX.nii.gz
        out_path = osp.join(out_root, split, 'GT', rel_in_gt)
        _save_volume(out_path, vol, affine, header)
        saved += 1

    print(f"[NIfTI] Saved {saved} volumes under: {osp.abspath(out_root)}")


if __name__ == '__main__':
    main()
