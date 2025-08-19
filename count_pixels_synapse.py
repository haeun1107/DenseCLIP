#!/usr/bin/env python3
# print_label_stats_total.py
import os
import argparse
import numpy as np
import nibabel as nib

LABEL_NAMES = {
    0:"background", 1:"spleen", 2:"right_kidney", 3:"left_kidney", 4:"gallbladder",
    5:"esophagus", 6:"liver", 7:"stomach", 8:"aorta", 9:"inferior_vena_cava",
    10:"portal_splenic_vein", 11:"pancreas", 12:"right_adrenal_gland", 13:"left_adrenal_gland"
}

def read_split(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser(description="Count labels over all NIfTI labels listed in a split.")
    ap.add_argument("--split", required=True, help="e.g. data/synapse/splits/train.txt")
    ap.add_argument("--gt_dir", required=True, help="e.g. data/synapse/train/GT or data/synapse/val/GT")
    ap.add_argument("--img_prefix", default="img")
    ap.add_argument("--lbl_prefix", default="label")
    ap.add_argument("--suffix", default=".nii.gz")
    args = ap.parse_args()

    bases = read_split(args.split)
    if len(bases) == 0:
        print("[ERR] empty split list"); return

    total_vox = 0                      # 전체(voxel) 합
    total_nonzero = 0                  # 라벨>0 합
    vox_per_class = {}                 # {label: count}
    vol_mm3_per_class = {}             # {label: mm^3}

    kept = 0
    for base in bases:
        lbl = base.replace(args.img_prefix, args.lbl_prefix) + args.suffix
        path = os.path.join(args.gt_dir, lbl)
        if not os.path.isfile(path):
            # 누락은 조용히 스킵
            continue

        nimg = nib.load(path)
        arr = np.asanyarray(nimg.get_fdata()).astype(np.int64)
        vx_mm3 = float(np.prod(nimg.header.get_zooms()))  # (X,Y,Z) mm

        uniq, cnts = np.unique(arr, return_counts=True)
        for u, c in zip(uniq.tolist(), cnts.tolist()):
            vox_per_class[u] = vox_per_class.get(u, 0) + int(c)
            vol_mm3_per_class[u] = vol_mm3_per_class.get(u, 0.0) + c * vx_mm3

        total_vox += int(arr.size)
        total_nonzero += int((arr > 0).sum())
        kept += 1

    if kept == 0:
        print("[ERR] no labels found from split in given gt_dir"); return

    # ------- 요약 한 번만 출력 -------
    print(f"[SUMMARY] split='{args.split}'  gt_dir='{args.gt_dir}'  "
          f"processed {kept}/{len(bases)} volumes")
    print(f"Total voxels: {total_vox:,}   Nonzero voxels: {total_nonzero:,}\n")

    header = " id | {:>22} | {:>12} | {:>12} | {:>7} | {:>8}".format(
        "name","voxels(sum)","volume(ml)","%all","%nonzero")
    print(header)
    print("-"*len(header))

    for u in sorted(vox_per_class.keys()):
        vox = vox_per_class[u]
        vol_ml = vol_mm3_per_class[u] / 1000.0
        pct_all = 100.0 * vox / max(total_vox, 1)
        pct_nz = 0.0 if u == 0 else 100.0 * vox / max(total_nonzero, 1)
        name = LABEL_NAMES.get(u, f"class_{u}")
        nz_str = f"{pct_nz:>7.2f}" if u != 0 else "   --  "
        print(f"{u:>3} | {name:>22} | {vox:>12,d} | {vol_ml:>12.3f} | {pct_all:>6.2f} | {nz_str}")

if __name__ == "__main__":
    main()

# python count_pixels_synapse.py --split data/synapse/splits/train.txt --gt_dir data/synapse/train/GT