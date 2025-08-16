# make_splits_all.py
import re, sys
from pathlib import Path

DATA_ROOT = Path("data/synapse")
TRAIN_CT = DATA_ROOT / "train/CT"
TRAIN_GT = DATA_ROOT / "train/GT"
VAL_CT   = DATA_ROOT / "val/CT"
VAL_GT   = DATA_ROOT / "val/GT"
SPLIT_DIR = DATA_ROOT / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def list_bases(ct_dir: Path, gt_dir: Path):
    bases = []
    missing = []
    for p in sorted(ct_dir.glob("img*.nii*"), key=lambda x: natural_key(x.name)):
        base = p.stem.replace(".nii", "")  # drops .gz then .nii => "img0001"
        # check matching GT file exists
        gt = gt_dir / f"{base.replace('img','label')}.nii.gz"
        if not gt.exists():
            gt_alt = gt_dir / f"{base.replace('img','label')}.nii"  # fallback
            if not gt_alt.exists():
                missing.append(base)
        bases.append(base)
    return bases, missing

def write_list(path: Path, items):
    with open(path, "w") as f:
        for b in items:
            f.write(b + "\n")

def main():
    train_bases, miss_train = list_bases(TRAIN_CT, TRAIN_GT)
    test_bases,  miss_test  = list_bases(VAL_CT,   VAL_GT)

    write_list(SPLIT_DIR / "train.txt", train_bases)
    write_list(SPLIT_DIR / "test.txt",  test_bases)

    print(f"[OK] Wrote {len(train_bases)} lines -> {SPLIT_DIR/'train.txt'}")
    print(f"[OK] Wrote {len(test_bases)}  lines -> {SPLIT_DIR/'test.txt'}")

    if miss_train:
        print(f"[WARN] Missing GT for {len(miss_train)} train cases:", miss_train)
    if miss_test:
        print(f"[WARN] Missing GT for {len(miss_test)} test cases:", miss_test)

if __name__ == "__main__":
    main()
