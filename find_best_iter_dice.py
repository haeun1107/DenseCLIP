# find_best_iter.py
import argparse, re, sys
from pathlib import Path

# e.g. "Iter [2000/80000]" 라인
RE_TRAIN_ITER = re.compile(r"Iter\s*\[(\d+)\s*/\s*(\d+)\]")
# e.g. "Saving checkpoint at 2000 iterations"
RE_SAVE_ITER  = re.compile(r"Saving checkpoint at\s+(\d+)\s+iterations")

# 실제 로그: aAcc, mIoU, mDice, mAcc, mPrec (mdice 위치 주의)
RE_VAL_LINE   = re.compile(
    r"Iter\(val\)\s*\[\d+\].*?aAcc:\s*([0-9.]+),\s*mIoU:\s*([0-9.]+),\s*mDice:\s*([0-9.]+),\s*mAcc:\s*([0-9.]+),\s*mPrec:\s*([0-9.]+)"
)

# 표 헤더도 실제 순서(aAcc | mIoU | mDice | mAcc | mPrec)로 수정
RE_SUMMARY_HDR = re.compile(r"^\|\s*aAcc\s*\|\s*mIoU\s*\|\s*mDice\s*\|\s*mAcc\s*\|\s*mPrec\s*\|")
RE_SUMMARY_VAL = re.compile(r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")

def parse_log(path: Path):
    # iter별 레코드 병합용
    recs_by_iter = {}
    last_iter = None
    wait_summary = False

    def ensure_iter(it):
        if it not in recs_by_iter:
            recs_by_iter[it] = {'iter': it, 'miou': None, 'mdice': None}

    with path.open('r', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = RE_TRAIN_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                wait_summary = False
                continue

            m = RE_SAVE_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                wait_summary = False
                continue

            # Iter(val) 라인: 0~1 값 → %로 변환
            m = RE_VAL_LINE.search(line)
            if m and last_iter is not None:
                _, miou, mdice, _, _ = m.groups()
                ensure_iter(last_iter)
                recs_by_iter[last_iter]['miou']  = float(miou)  * 100.0
                recs_by_iter[last_iter]['mdice'] = float(mdice) * 100.0
                # 표가 뒤따를 수도 있으니 wait_summary는 건드리지 않음
                continue

            if RE_SUMMARY_HDR.match(line):
                wait_summary = True
                continue

            if wait_summary:
                mv = RE_SUMMARY_VAL.match(line.strip())
                if mv and last_iter is not None:
                    # 표는 이미 % 단위
                    miou_pct  = float(mv.group(2))  # mIoU
                    mdice_pct = float(mv.group(3))  # mDice (주의: 3번째 칸)
                    ensure_iter(last_iter)
                    # 표가 더 신뢰도 높다고 가정하고 갱신
                    recs_by_iter[last_iter]['miou']  = miou_pct
                    recs_by_iter[last_iter]['mdice'] = mdice_pct
                    wait_summary = False
                continue

    # None 값 제거
    recs = [v for v in recs_by_iter.values() if v['miou'] is not None or v['mdice'] is not None]
    # 정렬(파일 내 시간순 가독성용, 선택)
    recs.sort(key=lambda r: (r['iter'] if r['iter'] is not None else -1))
    return recs

def better(curr, score, it):
    cs, ci, _ = curr
    if score > cs: return True
    if abs(score - cs) < 1e-9 and (it or -1) > (ci or -1): return True
    return False

def scan(target: Path):
    files = [target] if target.is_file() else [p for p in target.rglob('*') if p.suffix in {'.log', '.txt'}]
    files.sort()

    overall_miou  = (-1.0, None, None)
    overall_mdice = (-1.0, None, None)

    for f in files:
        recs = parse_log(f)
        if not recs:
            print(f"\n=== {f} ===\n(no eval records found)")
            continue

        # 파일 내 최고(동률이면 더 큰 iter)
        best_miou  = max(
            (r for r in recs if r['miou']  is not None),
            key=lambda r: (round(r['miou'], 6),  r['iter'] if r['iter'] is not None else -1),
            default=None
        )
        best_mdice = max(
            (r for r in recs if r['mdice'] is not None),
            key=lambda r: (round(r['mdice'], 6), r['iter'] if r['iter'] is not None else -1),
            default=None
        )

        print(f"\n=== {f} ===")
        if best_miou:
            print(f"[Best mIoU ] {best_miou['miou']:.2f}% @ iter {best_miou['iter']}")
            if better(overall_miou, best_miou['miou'], best_miou['iter']):
                overall_miou = (best_miou['miou'], best_miou['iter'], f)
        else:
            print("[Best mIoU ] not found")

        if best_mdice:
            print(f"[Best mDice] {best_mdice['mdice']:.2f}% @ iter {best_mdice['iter']}")
            if better(overall_mdice, best_mdice['mdice'], best_mdice['iter']):
                overall_mdice = (best_mdice['mdice'], best_mdice['iter'], f)
        else:
            print("[Best mDice] not found")

    print("\n=== OVERALL BEST ===")
    miou_s, miou_i, miou_f = overall_miou
    mdice_s, mdice_i, mdice_f = overall_mdice
    print(f"mIoU : {miou_s:.2f}% @ iter {miou_i} (file: {miou_f})" if miou_i is not None else "mIoU : not found")
    print(f"mDice: {mdice_s:.2f}% @ iter {mdice_i} (file: {mdice_f})" if mdice_i is not None else "mDice: not found")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="로그 파일(.log/.txt) 또는 디렉터리")
    p = Path(ap.parse_args().path)
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr); sys.exit(1)
    scan(p)
