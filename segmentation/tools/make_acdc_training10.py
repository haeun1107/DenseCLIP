import random

def sample_train_10_percent(src_path, dst_path, seed=42):
    # 원본 train.txt 읽기
    with open(src_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 재현성을 위해 시드 고정
    random.seed(seed)

    # 10% 샘플링
    sample_size = max(1, int(len(lines) * 0.1))
    sampled = random.sample(lines, sample_size)

    # 저장
    with open(dst_path, 'w') as f:
        f.write("\n".join(sampled) + "\n")

    print(f"✅ {sample_size} samples saved to {dst_path}")

if __name__ == "__main__":
    src_file = "data/ACDC/splits/train.txt"      # 원본
    dst_file = "data/ACDC/splits/train_10.txt"   # 저장 경로
    sample_train_10_percent(src_file, dst_file)
