# save as: find_best_iter.py
import argparse, re, sys
from pathlib import Path

# ---- regex patterns (좀 더 관대하게) ----
RE_TRAIN_ITER = re.compile(r"Iter\s*\[(\d+)\s*/\s*(\d+)\]")                 # Iter [21400/80000]
RE_SAVE_ITER  = re.compile(r"Saving checkpoint at\s+(\d+)\s+iterations")    # Saving checkpoint at 21400 iterations
RE_EVAL_LINE  = re.compile(r"Iter\(val\)\s*\[\d+\].*?mIoU:\s*([0-9.]+)")    # mIoU: 0.9032
RE_SUMMARY_HDR = re.compile(r"^\|\s*aAcc\s*\|\s*mIoU\s*\|\s*mAcc\s*\|\s*mPrec\s*\|\s*mDice\s*\|")
RE_SUMMARY_VAL = re.compile(
    r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
)

def parse_log(path: Path):
    """return list of dicts: [{'iter':21900,'miou':90.15,'mdice':94.79}, ...]"""
    recs = []
    last_iter = None           # 직전 학습 이터
    pending_idx = None         # 방금 추가한 evaluation 레코드 인덱스
    want_summary_values = False

    with path.open('r', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip("\n")

            # 1) 학습 iteration 갱신
            m = RE_TRAIN_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                continue
            m = RE_SAVE_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                continue

            # 2) 평가 라인 (mIoU 0~1 범위) → %로
            m = RE_EVAL_LINE.search(line)
            if m:
                miou_pct = float(m.group(1)) * 100.0
                recs.append({'iter': last_iter, 'miou': miou_pct, 'mdice': None})
                pending_idx = len(recs) - 1
                # Summary 값도 곧 나올 가능성이 높으니 대기 상태 진입
                # (헤더를 보지 못하더라도 값 줄만 잡으면 매칭)
                continue

            # 3-a) Summary 헤더를 보면 값 줄을 기다린다
            if RE_SUMMARY_HDR.match(line):
                want_summary_values = True
                continue

            # 3-b) Summary 값 줄을 찾을 때까지 계속 대기 (구분선/공백은 스킵)
            if want_summary_values:
                mv = RE_SUMMARY_VAL.match(line.strip())
                if mv:
                    # 순서: aAcc, mIoU, mAcc, mPrec, mDice
                    miou_pct = float(mv.group(2))
                    mdice_pct = float(mv.group(5))
                    if pending_idx is not None:
                        # 앞서 읽은 Iter(val) 레코드에 채워넣기
                        recs[pending_idx]['miou'] = recs[pending_idx]['miou'] or miou_pct
                        recs[pending_idx]['mdice'] = mdice_pct
                        pending_idx = None
                    else:
                        # 혹시 Iter(val) 라인이 없었으면 새로 추가
                        recs.append({'iter': last_iter, 'miou': miou_pct, 'mdice': mdice_pct})
                    want_summary_values = False
                # 값 줄이 아니면 계속 기다림 (구분선/빈 줄 등)
                continue

    return recs

def scan(target: Path):
    files = []
    if target.is_dir():
        files = [p for p in target.rglob("*") if p.is_file() and p.suffix in {".log", ".txt"}]
    else:
        files = [target]

    overall = {'miou': (-1.0, None, None), 'mdice': (-1.0, None, None)}  # (score, iter, file)

    for f in sorted(files):
        recs = parse_log(f)
        if not recs:
            print(f"\n=== {f} ===\n(no eval records found)")
            continue

        best_miou  = max(
                        (r for r in recs if r['miou'] is not None),
                        key=lambda x: (x['miou'], x['iter'] if x['iter'] is not None else -1),
                        default=None,
                    )
        best_mdice = max(
            (r for r in recs if r['mdice'] is not None),
            key=lambda x: (x['mdice'], x['iter'] if x['iter'] is not None else -1),
            default=None,
        )
        print(f"\n=== {f} ===")
    
    def _better(curr_tuple, new_score, new_iter):
        """curr_tuple = (score, iter, file). 점수 동점이면 더 큰 iter가 better."""
        cur_score, cur_iter, _ = curr_tuple
        if new_score > cur_score:
            return True
        if abs(new_score - cur_score) < 1e-12 and (new_iter or -1) > (cur_iter or -1):
            return True
        return False

    if best_miou and _better(overall['miou'], best_miou['miou'], best_miou['iter']):
        overall['miou'] = (best_miou['miou'], best_miou['iter'], f)

    if best_mdice and _better(overall['mdice'], best_mdice['mdice'], best_mdice['iter']):
        overall['mdice'] = (best_mdice['mdice'], best_mdice['iter'], f)

    print("\n=== OVERALL BEST ===")
    miou_score, miou_iter, miou_file = overall['miou']
    mdice_score, mdice_iter, mdice_file = overall['mdice']
    print(f"mIoU : {miou_score:.2f}% @ iter {miou_iter} (file: {miou_file})" if miou_iter is not None else "mIoU : not found")
    print(f"mDice: {mdice_score:.2f}% @ iter {mdice_iter} (file: {mdice_file})" if mdice_iter is not None else "mDice: not found")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="로그 파일(.log/.txt) 또는 디렉터리")
    args = ap.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr); sys.exit(1)
    scan(p)
