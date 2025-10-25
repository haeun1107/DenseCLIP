# busi_make_splits.py
import os, re, shutil, random
from glob import glob
from pathlib import Path

RAW_ROOT   = Path("data/BUSI")         # 현재 benign/malignant/normal가 있는 곳
OUT_ROOT   = RAW_ROOT                  # 재구성도 같은 위치에 만듦
IMG_DIR    = OUT_ROOT / "images"
MSK_DIR    = OUT_ROOT / "masks"
SPLIT_DIR  = OUT_ROOT / "splits"
CLASSES    = ["benign", "malignant", "normal"]

# split 비율 (ISIC과 동일 7:1:2)
TRAIN_R, VAL_R, TEST_R = 0.7, 0.1, 0.2
SEED = 2025

def _clean_name(stem: str):
    # 공백/괄호 등 통일
    s = re.sub(r"\s+", "_", stem)
    s = s.replace("(", "").replace(")", "")
    s = s.replace("__", "_")
    return s

def _copy_pair(src_img: Path, dst_img: Path, src_msk: Path, dst_msk: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_msk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_msk, dst_msk)

def main():
    random.seed(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MSK_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 모든 (이미지,마스크) 수집 & 표준 이름으로 복사
    pooled = []   # [(rel_img_path, rel_msk_path, cls)]
    for cls in CLASSES:
        src_dir = RAW_ROOT / cls
        imgs = sorted([p for p in src_dir.glob("*.png") if "_mask" not in p.name])
        for img in imgs:
            # 마스크 찾기 (BUSI는 보통 동일 파일명 + '_mask.png')
            msk = img.with_name(img.stem + "_mask.png")
            # 일부 케이스: '(1)_mask_1.png' 같은 변형 처리
            if not msk.exists():
                candidates = list(src_dir.glob(img.stem + "_mask*.png"))
                if candidates:
                    msk = candidates[0]
            if not msk.exists():
                print(f"[WARN] mask not found for {img}")
                continue

            # 표준 파일명: {cls}_{clean_stem}.png
            clean_stem = _clean_name(img.stem)
            out_img = IMG_DIR / f"{clean_stem}.png"
            out_msk = MSK_DIR / f"{clean_stem}_mask.png"

            _copy_pair(img, out_img, msk, out_msk)
            pooled.append((out_img.relative_to(OUT_ROOT).as_posix(),
                           out_msk.relative_to(OUT_ROOT).as_posix(),
                           cls))

    # 2) 클래스별로 70/10/20 분할 (stratified)
    by_cls = {c: [] for c in CLASSES}
    for rel_img, rel_msk, c in pooled:
        by_cls[c].append((rel_img, rel_msk))
    for c in CLASSES:
        random.shuffle(by_cls[c])

    train, val, test = [], [], []
    for c in CLASSES:
        items = by_cls[c]
        n = len(items)
        n_train = int(round(n * TRAIN_R))
        n_val   = int(round(n * VAL_R))
        # 남는 건 test로
        train += items[:n_train]
        val   += items[n_train:n_train+n_val]
        test  += items[n_train+n_val:]

    # 3) txt 저장 (split 파일에는 img 상대경로만 기록 — mmseg 표준)
    def write_split(name, items):
        out = SPLIT_DIR / name
        with open(out, "w") as f:
            for rel_img, _ in items:
                # mmseg는 img_dir 기준 상대경로를 읽음 -> images/ 접두사는 제거
                f.write(Path(rel_img).name + "\n")
        print(f"[OK] wrote {out} ({len(items)})")

    write_split("busi_train.txt", train)
    write_split("busi_val.txt", val)
    write_split("busi_test.txt", test)

    # 4) train의 10%만 모은 split(선택)
    k = max(1, int(round(len(train) * 0.10)))
    train_10 = random.sample(train, k)
    write_split("busi_train_10.txt", train_10)

if __name__ == "__main__":
    main()
