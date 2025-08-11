# make_train_10.py
import os, glob, random, argparse, math

def find_bases_from_training(train_dir):
    """training/ 아래의 NIfTI(.nii.gz)들과 _gt.nii.gz가 같이 있는 베이스 경로 수집"""
    imgs = sorted(glob.glob(os.path.join(train_dir, 'patient*', '*_frame*.nii.gz')))
    bases = []
    for p in imgs:
        if p.endswith('_gt.nii.gz'):
            continue
        gt = p.replace('.nii.gz', '_gt.nii.gz')
        if os.path.exists(gt):
            # base: patientXXX/patientXXX_frameYY
            rel = os.path.relpath(p, start=train_dir)
            bases.append(rel[:-7])  # strip ".nii.gz"
    return bases

def read_lines(path):
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

def sample_by_frames(bases, ratio, seed):
    rnd = random.Random(seed)
    items = bases[:]
    rnd.shuffle(items)
    k = max(1, int(round(len(items) * ratio)))
    picked = sorted(items[:k])
    return picked

def sample_by_patients(bases, ratio, seed):
    # patientID = 첫 디렉터리 이름 (e.g., patient101)
    from collections import defaultdict
    by_pt = defaultdict(list)
    for b in bases:
        pt = b.split(os.sep)[0]
        by_pt[pt].append(b)
    pts = sorted(by_pt.keys())
    rnd = random.Random(seed)
    rnd.shuffle(pts)
    kpt = max(1, int(round(len(pts) * ratio)))
    sel_pts = set(pts[:kpt])
    picked = []
    for pt in sorted(sel_pts):
        picked.extend(by_pt[pt])
    picked = sorted(picked)
    return picked

def main(root, split_train, out_name, ratio, seed, by_patient):
    train_dir = os.path.join(root, 'training')
    split_dir = os.path.join(root, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    split_train_path = os.path.join(split_dir, split_train)
    if os.path.exists(split_train_path):
        bases = read_lines(split_train_path)
    else:
        bases = find_bases_from_training(train_dir)

    if not bases:
        raise RuntimeError("No training bases found. Check paths.")

    if by_patient:
        picked = sample_by_patients(bases, ratio, seed)
    else:
        picked = sample_by_frames(bases, ratio, seed)

    out_path = os.path.join(split_dir, out_name)
    write_lines(out_path, picked)

    # 간단 리포트
    # 라벨 파일 존재여부 체크
    missing = []
    for b in picked:
        gt = os.path.join(root, 'training', b + '_gt.nii.gz')
        if not os.path.exists(gt):
            missing.append(b)

    print(f"Found {len(bases)} train items → sampled {len(picked)} ({ratio*100:.1f}%).")
    if by_patient:
        print("Sampling mode: per-patient")
    else:
        print("Sampling mode: per-frame")
    print(f"Wrote: {out_path}")
    if missing:
        print(f"[WARN] {len(missing)} items missing GT (first 5): {missing[:5]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ACDC", help="ACDC data root")
    ap.add_argument("--split-train", default="train.txt", help="existing full train split (if exists)")
    ap.add_argument("--out-name", default="train_10.txt", help="output split filename")
    ap.add_argument("--ratio", type=float, default=0.10, help="fraction to sample (e.g., 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--by-patient", action="store_true",
                    help="sample 10%% of patients (all their frames), not frames")
    args = ap.parse_args()
    main(args.root, args.split_train, args.out_name, args.ratio, args.seed, args.by_patient)
