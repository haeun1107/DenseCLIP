# mmseg/datasets/synapse.py
import os.path as osp
import numpy as np
import nibabel as nib

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SynapseNiftiDataset(CustomDataset):
    """Synapse(BTCV) NIfTI 3D 볼륨을 슬라이스 단위 2D 표본으로 전개하는 데이터셋.

    - split 파일에는 'img0001' 같은 베이스 이름만 들어 있음.
    - img_dir: '<root>/train/CT', ann_dir: '<root>/train/GT'
    - suffix: '.nii.gz'
    - 파이프라인에서 LoadNiftiSliceImage / LoadNiftiSliceAnnotations 를 사용한다.
      (여기서는 z_index 만 넘겨주면 됨)

    학습 설계:
      - 배경 포함(BG=0), 장기 클래스 1..13 → 총 14 클래스
      - reduce_zero_label=False 를 권장 (BG를 학습에 포함)
      - 평가 요약에서만 BG 제외 평균(mIoU_fg/mDice_fg 등)은 별도 Eval 로직에서 처리
    """

    # BG 포함: 총 14 classes (index 0이 background)
    CLASSES = [
        'empty space around body',
        'small soft organ near stomach', 'right side kidney shape', 'left side kidney shape', 'small sac below liver', 'food tube to stomach',
        'large organ on right side', 'rounded food pouch area', 'large main body artery', 'large body vein below heart',
        'vein network near liver', 'long soft organ behind stomach', 'small gland above right kidney',
        'small gland above left kidney'
    ]
    PALETTE = (
        [[0, 0, 0]] +                      # background
        [[i * 18, i * 18, i * 18] for i in range(1, 14)]
    )

    def __init__(self, split, **kwargs):
        """kwargs에는 mmseg 표준 인자(ex. img_dir, ann_dir, img_suffix, seg_map_suffix 등)가 들어옵니다.
        배경 포함 셋업이므로 reduce_zero_label=False 로 쓰는 것을 권장합니다.
        """
        super().__init__(split=split, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        """split 목록을 읽고, 각 볼륨에서 '라벨이 존재하는 z'만 샘플로 등록."""
        assert self.split is not None and osp.exists(self.split), f'Invalid split file: {self.split}'
        with open(self.split, 'r') as f:
            bases = [x.strip() for x in f if x.strip()]

        data_infos = []
        for base in bases:
            # 절대/상대 경로 모두 허용. 여기서는 풀 경로를 filename에 넣는다.
            img_path = osp.join(self.img_dir, f'{base}{img_suffix}')
            seg_path = osp.join(self.ann_dir, f'{base.replace("img", "label")}{seg_map_suffix}')

            # 3D 라벨 로드 후, 라벨이 있는 z-슬라이스만 선택
            nlab = nib.load(seg_path)
            lbl3d = np.asanyarray(nlab.get_fdata())  # 기대 형태: (H, W, S)
            has_label = np.any(lbl3d > 0, axis=(0, 1))  # (S,)
            z_indices = np.where(has_label)[0].tolist()

            for z in z_indices:
                data_infos.append(dict(
                    img_info=dict(filename=img_path, z_index=z),
                    ann_info=dict(seg_map=seg_path, z_index=z)
                ))

        print(f'[SynapseNiftiDataset] volumes: {len(bases)}, '
              f'slices(with label): {len(data_infos)}')
        return data_infos

    def get_gt_seg_maps(self, efficient_test=False):
        """GT를 0..13 그대로 반환 (BG=0 포함). unlabeled가 있으면 255를 사용."""
        gts = []
        for i in range(len(self.img_infos)):
            ann = self.get_ann_info(i)
            seg_rel = ann['seg_map']
            z = ann['z_index']

            # seg_map에 이미 프리픽스가 들어있으면 그대로, 아니면 ann_dir를 붙임.
            if osp.isabs(seg_rel) or seg_rel.startswith(self.ann_dir):
                seg_path = seg_rel
            else:
                seg_path = osp.join(self.ann_dir, seg_rel)

            lab3d = np.asanyarray(nib.load(seg_path).get_fdata()).astype(np.int32)
            lab = lab3d[..., z]  # (H, W)
            # 여기서 BG=0, 장기=1..13 그대로 둠. (255는 ignore로만 쓰임)
            gts.append(lab.astype(np.uint8))
        return gts

    # CustomDataset 기본 prepare_* 는 prefix를 붙여 사용할 수 있게 설계되어 있음.
    # 우리는 filename/seg_map에 "완전한 경로"를 넣었으므로 prefix가 끼어들지 않게 오버라이드.
    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'])
        if 'ann_info' in info:
            results['ann_info'] = info['ann_info']
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'])
        if 'ann_info' in info:  # 테스트에서도 GT가 필요할 때(평가) 사용
            results['ann_info'] = info['ann_info']
        return self.pipeline(results)

    # mmseg 호환용 헬퍼
    def get_ann_info(self, idx):
        return self.img_infos[idx].get('ann_info', {})
