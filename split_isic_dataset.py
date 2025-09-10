import os, glob

ROOT = 'data/ISIC'
pairs = [
    ('ISIC2018_Task1-2_Training_Input',   'ISIC2018_Task1_Training_GroundTruth',   'splits/isic_task1_train.txt'),
    ('ISIC2018_Task1-2_Validation_Input', 'ISIC2018_Task1_Validation_GroundTruth', 'splits/isic_task1_val.txt'),
    ('ISIC2018_Task1-2_Test_Input',       'ISIC2018_Task1_Test_GroundTruth',       'splits/isic_task1_test.txt'),
]

def stem(path): return os.path.splitext(os.path.basename(path))[0]

def main():
    os.makedirs(os.path.join(ROOT, 'splits'), exist_ok=True)
    for img_dir, gt_dir, out_txt in pairs:
        imgs = sorted(glob.glob(os.path.join(ROOT, img_dir, '*.jpg')))
        gts  = set(stem(p).replace('_segmentation','') for p in
                   glob.glob(os.path.join(ROOT, gt_dir, '*_segmentation.png')))
        kept = []
        for ip in imgs:
            s = stem(ip)
            if s in gts: kept.append(s)
        with open(os.path.join(ROOT, out_txt), 'w') as f:
            f.write('\n'.join(kept))
        print(f'[OK] {out_txt}: {len(kept)} samples')

if __name__ == '__main__':
    main()
