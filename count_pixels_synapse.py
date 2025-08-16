#!/usr/bin/env python3
import os
import csv
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk

# --------- 입력 경로 ----------
LABEL_PATH = "data/synapse/train/GT/label0001.nii.gz"  # <- 네 파일 경로

# (선택) BTCV 클래스 이름 매핑
LABEL_NAMES = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
}

def main():
    assert os.path.exists(LABEL_PATH), f"Not found: {LABEL_PATH}"

    # --- Read NIfTI with SimpleITK ---
    img = sitk.ReadImage(LABEL_PATH)  # labels are usually UInt8/UInt16
    arr = sitk.GetArrayFromImage(img)  # numpy array with shape (Z, Y, X)

    # --- Basic meta info ---
    size_xyz = img.GetSize()           # (X, Y, Z)
    spacing_xyz = img.GetSpacing()     # (X, Y, Z) in mm
    origin_xyz = img.GetOrigin()
    direction = img.GetDirection()
    dtype = arr.dtype
    shape_zyx = arr.shape              # (Z, Y, X)

    print("=== Meta ===")
    print(f"path        : {LABEL_PATH}")
    print(f"shape(Z,Y,X): {shape_zyx}")
    print(f"size (X,Y,Z): {size_xyz}")
    print(f"spacing (mm): {spacing_xyz}   (voxel volume = {np.prod(spacing_xyz):.3f} mm^3)")
    print(f"origin      : {origin_xyz}")
    print(f"dtype       : {dtype}")
    print(f"direction   : {direction}")

    # --- Unique labels & counts ---
    unique, counts = np.unique(arr, return_counts=True)
    voxel_vol_mm3 = float(np.prod(spacing_xyz))

    print("\n=== Label histogram ===")
    rows = []
    total_vox = int(arr.size)
    for val, cnt in zip(unique.tolist(), counts.tolist()):
        name = LABEL_NAMES.get(val, f"class_{val}")
        vol_ml = cnt * voxel_vol_mm3 / 1000.0  # 1 ml = 1000 mm^3
        frac = cnt / total_vox * 100.0
        rows.append(OrderedDict([
            ("label_id", val),
            ("name", name),
            ("voxels", cnt),
            ("volume_ml", round(vol_ml, 3)),
            ("percent", round(frac, 4)),
        ]))
        print(f"{val:>2} ({name:>22}) : vox={cnt:>8}  "
              f"vol={vol_ml:8.3f} ml  ({frac:6.3f} %)")

    # --- Nonzero ROI 범위(라벨 존재하는 Z 구간) ---
    nz = np.any(arr > 0, axis=(1, 2))
    if nz.any():
        z_idx = np.where(nz)[0]
        print(f"\nZ-range with labels : [{z_idx.min()} .. {z_idx.max()}]  "
              f"(thickness ~ {(z_idx.size * spacing_xyz[2]):.1f} mm)")
    else:
        print("\nNo nonzero labels found.")

    # --- Save CSV ---
    out_csv = os.path.join(os.path.dirname(LABEL_PATH), "label0001_stats.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
