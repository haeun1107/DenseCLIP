# tools/gen_acdc_splits.py
import os, glob, argparse

def list_bases(root_dir):
    paths = sorted(glob.glob(os.path.join(root_dir, 'patient*', '*_frame*.nii.gz')))
    # GT 있는 프레임만
    paths = [p for p in paths if os.path.exists(p.replace('.nii.gz','_gt.nii.gz'))]
    # 예: patient101/patient101_frame01  (확장자 제거)
    return [os.path.relpath(p, start=root_dir)[:-7] for p in paths]

def main(root):
    tr_dir = os.path.join(root, 'training')
    te_dir = os.path.join(root, 'testing')
    os.makedirs(os.path.join(root, 'splits'), exist_ok=True)

    train_bases = list_bases(tr_dir)
    val_bases   = list_bases(te_dir)

    with open(os.path.join(root, 'splits', 'train.txt'), 'w') as f:
        f.write('\n'.join(train_bases) + '\n')
    with open(os.path.join(root, 'splits', 'val.txt'), 'w') as f:
        f.write('\n'.join(val_bases) + '\n')

    print(f'train={len(train_bases)} from training/, val={len(val_bases)} from testing/')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ACDC')
    args = ap.parse_args()
    main(args.root)
