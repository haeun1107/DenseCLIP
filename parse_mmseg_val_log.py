import re
import csv
import argparse
from collections import OrderedDict

ITER_RE = re.compile(r'Iter \[(\d+)/\d+\]')
CKPT_RE = re.compile(r'Saving checkpoint at (\d+) iterations')
VAL_RE  = re.compile(r'Iter\(val\) \[\d+\].*?mIoU:\s*([0-9.]+),.*?mDice:\s*([0-9.]+)')

def parse_log(log_path):
    results = {}
    last_iter = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_iter = ITER_RE.search(line)
            if m_iter:
                last_iter = int(m_iter.group(1))
                continue

            m_ckpt = CKPT_RE.search(line)
            if m_ckpt:
                last_iter = int(m_ckpt.group(1))
                continue

            m_val = VAL_RE.search(line)
            if m_val and last_iter is not None:
                # ✅ 1000 단위가 아니면 스킵
                if last_iter % 1000 != 0:
                    last_iter = None
                    continue

                mIoU  = float(m_val.group(1)) * 100
                mDice = float(m_val.group(2)) * 100
                iter_k = f"{last_iter // 1000}K"
                results[iter_k] = (mIoU, mDice)
                last_iter = None

    # 정렬
    return OrderedDict(sorted(results.items(), key=lambda x: int(x[0][:-1])))

def save_csv(ordered_results, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iter(K)', 'mIoU(%)', 'mDice(%)'])
        for it, (miou, mdice) in ordered_results.items():
            writer.writerow([it, f'{miou:.2f}', f'{mdice:.2f}'])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('log_path', help='e.g., 20250811_093532.log')
    ap.add_argument('--out', default='iter_metrics_percent_k.csv', help='output CSV path')
    args = ap.parse_args()

    res = parse_log(args.log_path)
    if not res:
        print("No 1K-step (Iter,val) lines found.")
    else:
        save_csv(res, args.out)
        print(f"Saved {len(res)} rows to {args.out}")
