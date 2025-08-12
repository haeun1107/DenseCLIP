# find_best_iter.py
import argparse, re, sys
from pathlib import Path

RE_TRAIN_ITER = re.compile(r"Iter\s*\[(\d+)\s*/\s*(\d+)\]")
RE_SAVE_ITER  = re.compile(r"Saving checkpoint at\s+(\d+)\s+iterations")
RE_VAL_LINE   = re.compile(
    r"Iter\(val\)\s*\[\d+\].*?aAcc:\s*([0-9.]+),\s*mIoU:\s*([0-9.]+),\s*mAcc:\s*([0-9.]+),\s*mPrec:\s*([0-9.]+),\s*mDice:\s*([0-9.]+)"
)
RE_SUMMARY_HDR = re.compile(r"^\|\s*aAcc\s*\|\s*mIoU\s*\|\s*mAcc\s*\|\s*mPrec\s*\|\s*mDice\s*\|")
RE_SUMMARY_VAL = re.compile(r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")

def parse_log(path: Path):
    recs = []
    last_iter = None
    pending_idx = None
    wait_summary = False

    with path.open('r', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip("\n")

            # 학습 이터레이션 → 현재 iter 갱신 + pending 무효화
            m = RE_TRAIN_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                pending_idx = None
                wait_summary = False
                continue
            m = RE_SAVE_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                pending_idx = None
                wait_summary = False
                continue

            # Iter(val) 라인: 즉시 레코드 추가(0~1 → % 변환)
            m = RE_VAL_LINE.search(line)
            if m:
                _, miou, _, _, mdice = m.groups()
                recs.append({
                    'iter': last_iter,
                    'miou': float(miou) * 100.0,
                    'mdice': float(mdice) * 100.0,
                })
                pending_idx = len(recs) - 1
                # 표가 뒤따르면 덮어쓸 준비
                continue

            if RE_SUMMARY_HDR.match(line):
                wait_summary = True
                continue

            if wait_summary:
                mv = RE_SUMMARY_VAL.match(line.strip())
                if mv:
                    # 표는 이미 % 단위
                    miou_pct  = float(mv.group(2))
                    mdice_pct = float(mv.group(5))

                    if (pending_idx is not None and
                        recs[pending_idx]['iter'] == last_iter):
                        # 같은 iter의 표라면 덮어쓰기
                        recs[pending_idx]['miou']  = miou_pct
                        recs[pending_idx]['mdice'] = mdice_pct
                    else:
                        # 다른 iter(혹은 pending 없음)면 새 레코드
                        recs.append({'iter': last_iter, 'miou': miou_pct, 'mdice': mdice_pct})

                    pending_idx = None
                    wait_summary = False
                continue
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
        best_miou  = max(recs, key=lambda r: (round(r['miou'], 6),  r['iter'] if r['iter'] is not None else -1))
        best_mdice = max(recs, key=lambda r: (round(r['mdice'], 6), r['iter'] if r['iter'] is not None else -1))

        print(f"\n=== {f} ===")
        print(f"[Best mIoU ] {best_miou['miou']:.2f}% @ iter {best_miou['iter']}")
        print(f"[Best mDice] {best_mdice['mdice']:.2f}% @ iter {best_mdice['iter']}")

        if better(overall_miou, best_miou['miou'], best_miou['iter']):
            overall_miou = (best_miou['miou'], best_miou['iter'], f)
        if better(overall_mdice, best_mdice['mdice'], best_mdice['iter']):
            overall_mdice = (best_mdice['mdice'], best_mdice['iter'], f)

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
