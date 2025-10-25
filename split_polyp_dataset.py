# split_polyp_dataset_sampled.py
import random
from pathlib import Path
from typing import List, Tuple

# ===== 경로/설정 =====
PROJECT_PREFIX = "data"
ROOT = Path(PROJECT_PREFIX) / "Polyp/5dataset"
SPLIT_DIR = ROOT / "splits"
SEED = 2024

TRAIN_N = 1450
TEST_N  = 798

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False

def sort_key(p: Path):
    return (0, int(p.stem)) if is_int(p.stem) else (1, p.stem.lower())

def collect_pairs_anyext(ds_dir: Path) -> List[Tuple[Path, Path]]:
    img_dir, msk_dir = ds_dir / "images", ds_dir / "masks"
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    msks = [p for p in msk_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    imgs.sort(key=sort_key)

    stem2mask = {}
    for m in msks:
        stem2mask.setdefault(m.stem, m)

    pairs = []
    for im in imgs:
        m = stem2mask.get(im.stem)
        if m is not None:
            pairs.append((im, m))
    print(f"[INFO] {ds_dir.name}: {len(pairs)} matched pairs")
    return pairs

def write_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")
    print(f"[OK] {path.name}: {len(lines)} lines (LF)")

def main():
    random.seed(SEED)

    cvc300   = collect_pairs_anyext(ROOT / "CVC-300")
    clinicdb = collect_pairs_anyext(ROOT / "CVC-ClinicDB")
    colondb  = collect_pairs_anyext(ROOT / "CVC-ColonDB")
    etis     = collect_pairs_anyext(ROOT / "ETIS-Larib")
    kvasir   = collect_pairs_anyext(ROOT / "Kvasir-SEG")

    train_pool = clinicdb + kvasir
    if len(train_pool) < TRAIN_N:
        train_pairs = train_pool[:]
    else:
        train_pairs = random.sample(train_pool, TRAIN_N)

    train_set = {p[0] for p in train_pairs}
    remain_pairs = [pm for pm in train_pool if pm[0] not in train_set]
    test_pool = remain_pairs + cvc300 + colondb + etis

    if len(test_pool) < TEST_N:
        test_pairs = test_pool[:]
    else:
        test_pairs = random.sample(test_pool, TEST_N)

    # ---- 저장: 확장자 제외하고 파일명만 ----
    train_lines = [p.stem for p, _ in train_pairs]
    test_lines  = [p.stem for p, _ in test_pairs]

    # ---- 10% 서브셋 생성 ----
    ten_percent_n = max(1, int(len(train_lines) * 0.1))
    train_10_lines = random.sample(train_lines, ten_percent_n)


    print(f"[FINAL] Train={len(train_lines)}  Test={len(test_lines)}  Train_10={len(train_10_lines)}")
    write_lines(train_lines, SPLIT_DIR / "train.txt")
    write_lines(test_lines,  SPLIT_DIR / "test.txt")
    write_lines(train_10_lines, SPLIT_DIR / "train_10.txt")

if __name__ == "__main__":
    main()
