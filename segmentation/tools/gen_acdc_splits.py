import os, argparse, random, glob

def main(root, val_ratio=0.2, seed=42):
    random.seed(seed)
    train_dir = os.path.join(root, 'training')
    test_dir  = os.path.join(root, 'testing')
    split_dir = os.path.join(root, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    # 후보 이미지: *_frameXX.nii.gz (라벨은 *_gt.nii.gz)
    train_imgs = sorted(glob.glob(os.path.join(train_dir, 'patient*', '*_frame*.nii.gz')))
    test_imgs  = sorted(glob.glob(os.path.join(test_dir,  'patient*', '*_frame*.nii.gz')))

    # GT가 있는 프레임만 사용
    train_imgs = [p for p in train_imgs if os.path.exists(p.replace('.nii.gz', '_gt.nii.gz'))]
    test_imgs  = [p for p in test_imgs  if os.path.exists(p.replace('.nii.gz', '_gt.nii.gz'))]

    # 상대경로로 변환 (img_dir='training', ann_dir='training' 기준)
    rel_train = [os.path.relpath(p, start=train_dir) for p in train_imgs]
    rel_test  = [os.path.relpath(p, start=test_dir)  for p in test_imgs]

    # train을 다시 train/val로 나누고, test는 그대로 val로 써도 됨(선호에 따라)
    random.shuffle(rel_train)
    k = int(len(rel_train) * (1 - val_ratio))
    tr, va = rel_train[:k], rel_train[k:]

    # 테스트 슬라이스도 검증에 합치고 싶으면 아래 주석 해제
    # va += [os.path.join('..','testing', p) for p in rel_test]

    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join([os.path.join('patient'+p.split('patient')[-1].split('/')[0], os.path.basename(p)) for p in tr]) + '\n')

    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join([os.path.join('patient'+p.split('patient')[-1].split('/')[0], os.path.basename(p)) for p in va]) + '\n')

    print(f'Wrote {len(tr)} train and {len(va)} val items to {split_dir}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ACDC', help='ACDC data root')
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args.root, args.val_ratio, args.seed)
