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
    """

    CLASSES = [
        'spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus',
        'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_splenic_vein', 'pancreas', 'right_adrenal_gland',
        'left_adrenal_gland'
    ]
    PALETTE = [[i * 20, i * 20, i * 20] for i in range(13)]

    def __init__(self, split, **kwargs):
        # 배경 제외(=reduce_zero_label)은 파이프라인에서 처리함
        super().__init__(split=split, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        """split 목록을 읽고, 각 볼륨에서 '라벨이 존재하는 z'만 샘플로 등록."""
        with open(self.split, 'r') as f:
            bases = [x.strip() for x in f if x.strip()]

        data_infos = []
        for base in bases:
            # 절대/상대 경로 모두 허용. 여기서는 풀 경로를 filename에 넣는다.
            img_path = osp.join(self.img_dir, f'{base}{img_suffix}')
            seg_path = osp.join(self.ann_dir, f'{base.replace("img", "label")}{seg_map_suffix}')

            # 3D 라벨 로드 후, 라벨이 있는 z-슬라이스만 선택
            nlab = nib.load(seg_path)
            lbl3d = np.asanyarray(nlab.get_fdata())  # (H, W, S) 가정
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

    # (선택) mmseg 호환용 헬퍼
    def get_ann_info(self, idx):
        return self.img_infos[idx].get('ann_info', {})
