import os, re, glob, argparse

def list_frames(patient_dir):
    pats = sorted(glob.glob(os.path.join(patient_dir, '*_frame*.nii.gz')))
    frames = []
    for p in pats:
        base = os.path.basename(p)
        m = re.search(r'_frame(\d+)\.nii\.gz$', base)
        if m:
            frames.append(int(m.group(1)))
    return sorted(set(frames))

def kth(frames_sorted, k):
    return frames_sorted[k-1] if 1 <= k <= len(frames_sorted) else None

def find_patient_dir(root, pid, search_subdirs):
    for sd in search_subdirs:
        pdir = os.path.join(root, sd, pid)
        if os.path.isdir(pdir):
            return sd, pdir
    return None, None

def convert(root, src_list, dst_list, search_subdirs=('training','testing')):
    with open(src_list, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    out, skipped = [], []
    for ln in lines:
        m = re.match(r'^(patient\d+)_frame(\d+)$', ln)
        if not m:
            skipped.append((ln, 'bad format'))
            continue
        pid, k = m.group(1), int(m.group(2))

        sd, pdir = find_patient_dir(root, pid, search_subdirs)
        if pdir is None:
            skipped.append((ln, f'missing dir in {search_subdirs}'))
            continue

        frames = list_frames(pdir)
        if not frames:
            skipped.append((ln, 'no frames'))
            continue

        real_num = kth(frames, k)
        if real_num is None:
            skipped.append((ln, f'only {len(frames)} frames -> no k={k}'))
            continue

        out.append(f'{pid}/{pid}_frame{real_num:02d}')

    os.makedirs(os.path.dirname(dst_list), exist_ok=True)
    with open(dst_list, 'w') as f:
        f.write('\n'.join(out) + '\n')

    print(f'✅ wrote {len(out)} lines to {dst_list}')
    if skipped:
        print('⚠️ skipped:')
        for s in skipped[:15]:
            print('  -', s)
        if len(skipped) > 15:
            print(f'  ... and {len(skipped)-15} more')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ACDC')
    ap.add_argument('--src-list', required=True)
    ap.add_argument('--dst-list', required=True)
    ap.add_argument('--subdirs', nargs='+', default=['training','testing'])
    args = ap.parse_args()
    convert(args.root, args.src_list, args.dst_list, args.subdirs)
    
# python segmentation/tools/gen_acdc_splits_re.py \
# --root data/ACDC \
#   --src-list data/ACDC/splits/test.list \
#   --dst-list data/ACDC/splits/test.txt \
#   --subdirs training testing