# segmentation/mmseg/datasets/acdcbg.py
import os
import os.path as osp
import numpy as np
import nibabel as nib

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class ACDCBGDataset(CustomDataset):
    """ACDC 2D slice dataset using .nii.gz files (with background included).

    split 파일의 각 줄은 확장자 없는 'base path'여야 합니다:
      e.g., patient001/patient001_frame01
    그러면 이미지: <img_dir>/<base>.nii.gz
            라벨 : <ann_dir>/<base>_gt.nii.gz

    배경 포함 설정: label 0=background, 1=RV, 2=myocardium, 3=LV
    """

    # 배경 포함 4클래스
    CLASSES = ['background', 'right ventricle cavity', 'myocardium', 'left ventricle cavity']
    PALETTE = [
        [0, 0, 0],      # background
        [0, 0, 255],    # RV
        [255, 0, 0],    # myocardium
        [0, 255, 0],    # LV
    ]

    def __init__(self, split, slice_index=None, **kwargs):
        """kwargs에는 mmseg 표준 인자(ex. img_dir, ann_dir, img_suffix, seg_map_suffix 등)가 들어옵니다.
        reduce_zero_label=False로 사용하는 것이 배경 포함 셋업에 맞습니다.
        """
        self.slice_index = slice_index  # None이면 중앙 슬라이스 사용
        super().__init__(split=split, **kwargs)

    # --- 파일 리스트 로딩 ---
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None, **kwargs):
        assert self.split is not None and osp.exists(self.split), f"Invalid split file: {self.split}"
        with open(self.split, 'r') as f:
            bases = [ln.strip() for ln in f if ln.strip()]

        data_infos = []
        for base in bases:
            img_file = osp.join(img_dir, base + img_suffix)            # .nii.gz
            seg_file = osp.join(ann_dir,  base + seg_map_suffix)       # _gt.nii.gz
            data_infos.append(dict(
                img_info=dict(filename=img_file, img_prefix=None),
                ann_info=dict(seg_map=seg_file, seg_prefix=None)
            ))
        return data_infos

    # --- NIfTI 라벨을 2D 슬라이스로 읽기 ---
    def _read_2d_label(self, seg_map_filename):
        seg_nii = nib.load(seg_map_filename)
        seg = np.asanyarray(seg_nii.get_fdata())  # 기대 형태: (H, W) 또는 (H, W, S)
        if seg.ndim == 3:
            idx = self.slice_index if self.slice_index is not None else seg.shape[-1] // 2
            seg = seg[..., int(idx)]
        elif seg.ndim != 2:
            raise ValueError(f'Unexpected label shape {seg.shape} for {seg_map_filename}')

        # ACDC 라벨은 일반적으로 0..3 (배경 포함) 범위
        seg = seg.astype(np.uint8)
        return seg

    # --- reduce_zero_label 지원(옵션) ---
    def _apply_reduce_zero(self, seg):
        """배경을 무시(255)하고 1..N을 0..N-1로 시프트. 배경 포함 셋업에선 사용하지 않는 것을 권장."""
        seg = seg.astype(np.uint8).copy()
        seg[seg == 0] = 255
        seg = seg - 1
        seg[seg == 254] = 255
        return seg

    def get_gt_seg_map_by_idx(self, index):
        seg_path = self.img_infos[index]['ann_info']['seg_map']
        seg = self._read_2d_label(seg_path)

        return seg

    def get_gt_seg_maps(self, efficient_test=False):
        return [self.get_gt_seg_map_by_idx(i) for i in range(len(self.img_infos))]

    # --- mmseg 파이프라인 연동 ---
    def prepare_test_img(self, idx):
        results = dict(
            img_info=self.img_infos[idx]['img_info'],
            ann_info=self.img_infos[idx].get('ann_info', None)
        )
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx]['img_info'])
        if 'ann_info' in self.img_infos[idx]:
            results['ann_info'] = self.img_infos[idx]['ann_info']
        return self.pipeline(results)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann_info']
