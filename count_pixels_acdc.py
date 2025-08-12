import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

CLASS_NAMES_4 = {
    0: "background",
    1: "right_ventricle_cavity",
    2: "myocardium",
    3: "left_ventricle_cavity",
}
CLASS_NAMES_3 = {
    0: "right_ventricle_cavity",
    1: "myocardium",
    2: "left_ventricle_cavity",
}

def read_split_list(split_file):
    with open(split_file, "r") as f:
        bases = [ln.strip() for ln in f if ln.strip()]
    return bases

def print_split_preview(split_file, head=0, title="Split"):
    bases = read_split_list(split_file)
    print(f"\n[{title}] {split_file}  (items: {len(bases)})")
    if head and head > 0:
        show = bases[:head]
        for b in show:
            print("  ", b)
        if len(bases) > head:
            print(f"  ... (+{len(bases) - head} more)")
    else:
        for b in bases:
            print("  ", b)

def load_label_nii(path):
    nimg = nib.load(str(path))
    arr = np.asanyarray(nimg.get_fdata())
    return arr  # (H,W,S) or (H,W)

def reduce_zero_label(seg):
    """0->255(ignore), 1->0, 2->1, 3->2 (ACDC)"""
    seg = seg.astype(np.int32).copy()
    seg[seg == 0] = 255
    seg = seg - 1
    seg[seg == 254] = 255
    return seg

def count_file(seg_path, mode="middle", remap=False, minlength=256):
    arr = load_label_nii(seg_path)
    if arr.ndim == 3:
        if mode == "middle":
            idx = arr.shape[-1] // 2
            seg = arr[..., idx]
        elif mode == "all":
            seg = arr  # all slices
        else:
            raise ValueError(f"Unknown mode: {mode}")
    elif arr.ndim == 2:
        seg = arr
    else:
        raise ValueError(f"Unexpected label shape {arr.shape} for {seg_path}")

    if remap:
        seg = reduce_zero_label(seg)

    flat = seg.reshape(-1) if seg.ndim == 3 else seg.ravel()
    flat = flat.astype(np.int64)
    counts = np.bincount(flat, minlength=minlength)
    return counts

def count_split(data_root, ann_dir, split_file, seg_suffix="_gt.nii.gz",
                mode="middle", remap=False):
    bases = read_split_list(split_file)
    total = np.zeros(256, dtype=np.int64)
    for base in tqdm(bases, desc=f"Counting ({mode}) in {Path(ann_dir).name}", ncols=80):
        seg_path = Path(data_root) / ann_dir / f"{base}{seg_suffix}"
        if not seg_path.exists():
            raise FileNotFoundError(seg_path)
        total += count_file(seg_path, mode=mode, remap=remap)
    return total

def pretty_print(title, total_counts, remap=False):
    if remap:
        keys = [0, 1, 2]
        names = CLASS_NAMES_3
        denom = int(total_counts[0] + total_counts[1] + total_counts[2])
    else:
        keys = [0, 1, 2, 3]
        names = CLASS_NAMES_4
        denom = int(sum(total_counts[k] for k in keys))
    print()
    print(f"[Pixel Counts] {title} | Total pixels: {denom:,}")
    for k in keys:
        cnt = int(total_counts[k])
        pct = (cnt / denom * 100.0) if denom > 0 else 0.0
        print(f"Class {k:1d} ({names[k]:<23}): {cnt:,}  ({pct:5.2f}%)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/ACDC")
    # ⬇ 여러 개 split 파일을 받을 수 있도록 nargs="+"
    ap.add_argument("--train_split", nargs="+",
                    default=["data/ACDC/splits/train.txt", "data/ACDC/splits/train_10.txt"],
                    help="One or more training split files.")
    ap.add_argument("--val_split", nargs="+",
                    default=["data/ACDC/splits/val.txt", "data/ACDC/splits/test.txt"],
                    help="One or more validation, testing split files.")
    ap.add_argument("--img_dir_train", default="training")
    ap.add_argument("--img_dir_val", default="training")
    ap.add_argument("--img_dir_test", default="training")
    ap.add_argument("--seg_suffix", default="_gt.nii.gz")
    ap.add_argument("--mode", choices=["middle", "all"], default="middle",
                    help="middle slice only or all slices")
    ap.add_argument("--remap", action="store_true",
                    help="apply reduce_zero_label and count 3 classes (ignore=255)")
    # ⬇ split 내용 출력 옵션
    ap.add_argument("--show_splits", action="store_true",
                    help="Print entries in each split file.")
    ap.add_argument("--head", type=int, default=0,
                    help="If >0, print only the first N entries per split.")
    args = ap.parse_args()

    # ---- TRAIN splits ----
    for sp in args.train_split:
        if args.show_splits:
            print_split_preview(sp, head=args.head, title="Train split")

        counts = count_split(args.data_root, args.img_dir_train, sp,
                             seg_suffix=args.seg_suffix, mode=args.mode, remap=args.remap)
        pretty_print(f"Train [{args.mode}] ({Path(sp).name})", counts, remap=args.remap)

    # ---- VAL splits ----
    for sp in args.val_split:
        if args.show_splits:
            print_split_preview(sp, head=args.head, title="Val split")

        counts = count_split(args.data_root, args.img_dir_val, sp,
                             seg_suffix=args.seg_suffix, mode=args.mode, remap=args.remap)
        pretty_print(f"Val   [{args.mode}] ({Path(sp).name})", counts, remap=args.remap)

if __name__ == "__main__":
    main()
