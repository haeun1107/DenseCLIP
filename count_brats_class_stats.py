# # check_brats_raw_labels.py
# import os, sys, numpy as np, nibabel as nib
# from tqdm import tqdm

# root = sys.argv[1] if len(sys.argv) > 1 else "data/BraTS"
# dataset_dir = os.path.join(root, "dataset")

# uniq = set()
# counts = {}  # raw label -> voxel count

# cases = [d for d in os.listdir(dataset_dir)
#          if os.path.isdir(os.path.join(dataset_dir, d))]
# for stem in tqdm(sorted(cases), desc="Scanning (raw labels)"):
#     seg_nii = None
#     for ext in (".nii.gz", ".nii"):
#         p = os.path.join(dataset_dir, stem, f"{stem}_seg{ext}")
#         if os.path.exists(p):
#             seg_nii = p; break
#     if seg_nii is None:
#         print(f"[WARN] seg not found: {stem}"); continue

#     seg = nib.load(seg_nii).get_fdata().astype(np.int16)
#     u, c = np.unique(seg, return_counts=True)
#     for v, n in zip(u, c):
#         uniq.add(int(v))
#         counts[int(v)] = counts.get(int(v), 0) + int(n)

# print("\nRAW LABEL VALUES FOUND:", sorted(uniq))
# tot = sum(counts.values())
# print(f"Total voxels counted: {tot:,}\n")
# print(" id | voxels(sum)     | %all")
# print("-------------------------------")
# for k in sorted(counts):
#     pct = 100.0 * counts[k] / tot if tot else 0.0
#     print(f"{k:>3} | {counts[k]:>14,} | {pct:5.3f}%")

# brats_class_stats.py
import os, argparse, csv, sys, glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def nii_path(case_dir, stem, suffix):
    """suffix in ['flair','t1','t1ce','t2','seg'] -> return existing path (.nii or .nii.gz)."""
    cand = [os.path.join(case_dir, f"{stem}_{suffix}.nii"),
            os.path.join(case_dir, f"{stem}_{suffix}.nii.gz")]
    for p in cand:
        if os.path.exists(p):
            return p
    return None

def load_seg(seg_path, remap_4_to_3=True):
    seg = nib.load(seg_path).get_fdata()
    # safety to int
    seg = seg.astype(np.int16)
    if remap_4_to_3:
        seg[seg == 4] = 3
    return seg.astype(np.uint8)

def bincount_labels(seg, n_classes=4):
    """counts for labels {0,1,2,3}; assumes 4 already mapped to 3."""
    bc = np.bincount(seg.flatten(), minlength=n_classes)
    return bc[:n_classes]

def percent(v, tot):
    return (100.0 * v / tot) if tot > 0 else 0.0

def main(args):
    root = args.root  # e.g., data/BraTS
    dataset_dir = os.path.join(root, "dataset")
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # collect case list
    if args.list and os.path.isfile(args.list):
        with open(args.list, 'r') as f:
            stems = [ln.strip() for ln in f if ln.strip()]
    else:
        # infer all case dirs under dataset/
        stems = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d))])

    print(f"[INFO] #cases to scan: {len(stems)}")

    # prepare outputs
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "brats_class_stats.csv")

    total_counts = np.zeros(4, dtype=np.int64)  # [bg,1,2,3]
    non_empty_cases = 0
    rows = []

    for stem in tqdm(stems, desc="Scanning"):
        case_dir = os.path.join(dataset_dir, stem)
        seg_path = nii_path(case_dir, stem, "seg")
        if seg_path is None:
            print(f"[WARN] seg not found for {stem}, skip")
            continue

        seg = load_seg(seg_path, remap_4_to_3=True)
        counts = bincount_labels(seg, n_classes=4)
        voxels = int(seg.size)
        fg = int(counts[1:].sum())
        bg = int(counts[0])

        total_counts += counts
        if fg > 0:
            non_empty_cases += 1

        row = {
            "case": stem,
            "voxels_total": voxels,
            "bg_0": int(counts[0]),
            "ncr_net_1": int(counts[1]),
            "ed_2": int(counts[2]),
            "et_3": int(counts[3]),
            "fg_ratio_%": round(percent(fg, voxels), 4),
            "ncr_net_ratio_%(of_fg)": round(percent(counts[1], max(fg,1)), 4),
            "ed_ratio_%(of_fg)": round(percent(counts[2], max(fg,1)), 4),
            "et_ratio_%(of_fg)": round(percent(counts[3], max(fg,1)), 4),
        }
        rows.append(row)

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["case","voxels_total","bg_0","ncr_net_1","ed_2","et_3",
                                 "fg_ratio_%","ncr_net_ratio_%(of_fg)","ed_ratio_%(of_fg)","et_ratio_%(of_fg)"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    tot_vox = int(total_counts.sum())
    fg_tot = int(total_counts[1:].sum())
    print("\n===== GLOBAL SUMMARY =====")
    print(f"#cases scanned           : {len(rows)}")
    print(f"#cases with tumor (fg>0) : {non_empty_cases}")
    print(f"Total voxels             : {tot_vox:,}")
    print(f"BG (0)                   : {int(total_counts[0]):,} "
          f"({percent(total_counts[0], tot_vox):.3f}%)")
    print(f"NCR/NET (1)              : {int(total_counts[1]):,} "
          f"({percent(total_counts[1], tot_vox):.3f}% | of-fg {percent(total_counts[1], fg_tot):.3f}%)")
    print(f"ED (2)                   : {int(total_counts[2]):,} "
          f"({percent(total_counts[2], tot_vox):.3f}% | of-fg {percent(total_counts[2], fg_tot):.3f}%)")
    print(f"ET (3) [4→3]             : {int(total_counts[3]):,} "
          f"({percent(total_counts[3], tot_vox):.3f}% | of-fg {percent(total_counts[3], fg_tot):.3f}%)")
    print(f"FG total                 : {fg_tot:,} "
          f"({percent(fg_tot, tot_vox):.3f}%)")
    print(f"[SAVED] per-case CSV → {csv_path}")

    # optional: basic intensity stats for FLAIR (helpful when 만들 RGB)
    if args.intensity:
        import json
        q = {"min": [], "p1": [], "p50": [], "p99": [], "max": []}
        for stem in tqdm(stems, desc="FLAIR intensity"):
            case_dir = os.path.join(dataset_dir, stem)
            flair_p = nii_path(case_dir, stem, "flair")
            if flair_p is None: continue
            arr = nib.load(flair_p).get_fdata().astype(np.float32)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0: continue
            q["min"].append(float(np.min(arr)))
            q["p1"].append(float(np.percentile(arr, 1)))
            q["p50"].append(float(np.percentile(arr, 50)))
            q["p99"].append(float(np.percentile(arr, 99)))
            q["max"].append(float(np.max(arr)))
        stats = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k,v in q.items() if v}
        out_json = os.path.join(args.outdir, "flair_intensity_stats.json")
        with open(out_json, "w") as f:
            import json; json.dump(stats, f, indent=2)
        print(f"[SAVED] FLAIR intensity summary → {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS class distribution (0,1,2,4→3)")
    parser.add_argument("--root", type=str, default="data/BraTS",
                        help="dataset root that contains 'dataset/' and optional train/val/test txts")
    parser.add_argument("--list", type=str, default="", help="optional path to train/val/test .txt (one stem per line)")
    parser.add_argument("--outdir", type=str, default="data/BraTS/stats")
    parser.add_argument("--intensity", action="store_true", help="also summarize FLAIR intensity percentiles")
    args = parser.parse_args()
    main(args)
