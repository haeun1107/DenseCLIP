import os
import glob
import random

ROOT = 'data/ISIC'
IMG_DIR = 'ISIC2018_Task1-2_Training_Input'
GT_DIR = 'ISIC2018_Task1_Training_GroundTruth'
SPLIT_DIR = os.path.join(ROOT, 'splits')

# 출력 파일 이름
OUTPUT_FILES = {
    'train': 'isic_task1_train_fix.txt',  # Train 70%에서 다시 10%만 선택
    'val':   'isic_task1_val_fix.txt',    # Validation 10%
    'test':  'isic_task1_test_fix.txt'    # Internal Test 20%
}

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # 1. 이미지와 GT 불러오기
    imgs = sorted(glob.glob(os.path.join(ROOT, IMG_DIR, '*.jpg')))
    gts = set(stem(p).replace('_segmentation', '') for p in
              glob.glob(os.path.join(ROOT, GT_DIR, '*_segmentation.png')))

    # GT와 매칭되는 이미지만 사용
    valid_imgs = [img for img in imgs if stem(img) in gts]
    total = len(valid_imgs)
    print(f"[INFO] Total valid images: {total}")  # 예상: 2594

    # 2. 셔플
    random.seed(42)
    random.shuffle(valid_imgs)

    # 3. 1차 split (70% / 10% / 20%)
    train_end = int(total * 0.7)   # 70% → 1815
    val_end = int(total * 0.8)     # 70% + 10% → 2074

    train_imgs = valid_imgs[:train_end]         # 70%
    val_imgs = valid_imgs[train_end:val_end]    # 10%
    test_imgs = valid_imgs[val_end:]            # 20%

    print(f"[INFO] Split sizes -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # 4. Train에서 다시 10%만 선택
    random.shuffle(train_imgs)  # Train 내부 재셔플
    final_train_count = int(len(train_imgs) * 0.1)  # 10% → 약 182
    final_train_imgs = train_imgs[:final_train_count]

    print(f"[INFO] Final train count (10% of 70%): {len(final_train_imgs)}")

    # 5. 저장 함수
    def save_split(data_list, filename):
        stems = [stem(p) for p in data_list]
        with open(os.path.join(SPLIT_DIR, filename), 'w') as f:
            f.write('\n'.join(stems))
        print(f"[OK] {filename}: {len(stems)} saved")

    # 6. 최종 저장
    save_split(final_train_imgs, OUTPUT_FILES['train'])  # Train(70%) → 10%
    save_split(val_imgs, OUTPUT_FILES['val'])            # Validation(10%)
    save_split(test_imgs, OUTPUT_FILES['test'])          # Test(20%)

if __name__ == '__main__':
    main()
