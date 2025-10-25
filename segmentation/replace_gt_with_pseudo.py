#!/usr/bin/env python3
# replace (ACDC)
import argparse, os, os.path as osp, shutil
from collections import Counter

def read_list(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser("Simulate/Apply: replace 90% GT with pseudo in training/")
    ap.add_argument("--data-root", default="data/ACDC")
    ap.add_argument("--train-split", default="splits/train.txt")
    ap.add_argument("--train10-split", default="splits/train_10.txt")
    ap.add_argument("--pseudo-dir", default="pseudo_A0")
    ap.add_argument("--train-dir", default="training")
    ap.add_argument("--backup-dir", default="backup_gt_before_B0")
    ap.add_argument("--apply", action="store_true", help="실제로 교체 수행 (기본은 시뮬레이션)")
    ap.add_argument("--mode", choices=["copy","move"], default="copy",
                    help="백업 시 copy 또는 move (기본 copy)")
    ap.add_argument("--preview", type=int, default=10, help="미리보기 개수")
    args = ap.parse_args()

    data_root   = args.data_root
    train_list  = osp.join(data_root, args.train_split)
    train10     = osp.join(data_root, args.train10_split)
    pseudo_root = osp.join(data_root, args.pseudo_dir)
    train_root  = osp.join(data_root, args.train_dir)
    backup_root = osp.join(data_root, args.backup_dir)

    all_items = read_list(train_list)        # 'patient009/patient009_frame01'
    keep10    = set(read_list(train10))
    dups = [k for k,c in Counter(all_items).items() if c>1]
    if dups: print(f"[WARN] 중복 항목 {len(dups)}개 발견 (예: {dups[:3]})")

    to_replace = [x for x in all_items if x not in keep10]
    to_keep    = [x for x in all_items if x in keep10]

    # 집계
    print(f"\n[PLAN] 총 {len(all_items)}개")
    print(f"  - 원본 유지(10%): {len(to_keep)}개")
    print(f"  - pseudo로 교체(90%): {len(to_replace)}개")
    print(f"  - pseudo 소스 루트: {pseudo_root}")
    print(f"  - training 루트   : {train_root}")
    print(f"  - 백업 루트       : {backup_root}")
    print(f"  - 실행 모드       : {'APPLY' if args.apply else 'DRY-RUN'}\n")

    # 존재 여부 체크 & 미리보기
    miss_pseudo, miss_gt = [], []
    preview_rows = []
    for stem in to_replace:
        patient = osp.dirname(stem)
        base    = osp.basename(stem)
        name    = base + "_gt.nii.gz"
        src_gt  = osp.join(train_root, patient, name)
        src_ps  = osp.join(pseudo_root, patient, name)

        if not osp.isfile(src_ps): miss_pseudo.append(src_ps)
        if not osp.isfile(src_gt): miss_gt.append(src_gt)

        if len(preview_rows) < args.preview:
            preview_rows.append((stem, src_ps, src_gt))

    if preview_rows:
        print("[PREVIEW] 교체 예정 항목 (일부)")
        for stem, src_ps, src_gt in preview_rows:
            print(f"  {stem}:")
            print(f"    pseudo → {src_ps}")
            print(f"    target → {src_gt}")
        if len(to_replace) > args.preview:
            print(f"  ... (총 {len(to_replace)}개 중 {args.preview}개만 표시)")
        print()

    if miss_pseudo:
        print(f"[CHECK] pseudo 파일 누락 {len(miss_pseudo)}개 (예시)")
        for p in miss_pseudo[:5]: print("  -", p)
    if miss_gt:
        print(f"[CHECK] 기존 GT 누락 {len(miss_gt)}개 (예시)")
        for p in miss_gt[:5]: print("  -", p)
    if miss_pseudo or miss_gt:
        print("\n[ABORT] 누락 파일이 있습니다. 경로/생성 여부 확인 후 다시 실행하세요.")
        if not args.apply:
            return
        # apply 모드에서도 누락이 있으면 실제 교체는 해당 항목 건너뜀

    if not args.apply:
        print("\n[DRY-RUN] 시뮬레이션만 수행했습니다. 실제 적용하려면 --apply 플래그를 추가하세요.")
        return

    # === 실제 적용 ===
    ensure_dir(backup_root)
    replaced = kept = skipped = 0

    # 10%: 그대로 두되, 백업은 하지 않음
    kept = len(to_keep)

    # 90%: 백업 후 pseudo로 덮어쓰기
    for stem in to_replace:
        patient = osp.dirname(stem)
        base    = osp.basename(stem)
        name    = base + "_gt.nii.gz"
        target  = osp.join(train_root, patient, name)
        pseudo  = osp.join(pseudo_root, patient, name)

        if not (osp.isfile(target) and osp.isfile(pseudo)):
            skipped += 1
            continue

        # 백업 경로
        bdir = osp.join(backup_root, patient)
        ensure_dir(bdir)
        bdst = osp.join(bdir, name)

        # 백업 (copy/move 선택)
        if args.mode == "move":
            shutil.move(target, bdst)
        else:
            shutil.copy2(target, bdst)

        # 덮어쓰기
        shutil.copy2(pseudo, target)
        replaced += 1

    print(f"\n[DONE]")
    print(f"  - pseudo로 교체: {replaced}개")
    print(f"  - 원본 유지(10%): {kept}개")
    print(f"  - 누락/스킵     : {skipped}개")
    print(f"  - 백업 위치     : {backup_root}\n")

if __name__ == "__main__":
    main()
