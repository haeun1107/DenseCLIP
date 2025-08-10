# segmentation/mmseg/datasets/acdc.py
import os.path as osp
import numpy as np
import nibabel as nib

from .builder import DATASETS           
from .custom import CustomDataset      

@DATASETS.register_module()
class ACDCDataset(CustomDataset):
    """ACDC 2D slice dataset using .nii.gz files.

    split 파일의 각 줄은 확장자 없는 'base path'여야 합니다:
      e.g., patient001/patient001_frame01
    그러면 이미지: <img_dir>/<base>.nii.gz
            라벨 : <ann_dir>/<base>_gt.nii.gz
    """

    # mmseg 0.x: CLASSES/PALETTE는 bg 포함도 가능. 여기선 참고용으로만 쓰고,
    # 실제 loss/class 수는 config에서 num_classes=4로 맞춰주세요.
    CLASSES = ['background', 'right ventricle cavity', 'myocardium', 'left ventricle cavity']
    PALETTE = [[0,0,0],[0,0,255],[255,0,0],[0,255,0]]

    def __init__(self, split, slice_index=None, **kwargs):
        self.slice_index = slice_index
        super().__init__(split=split, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None, **kwargs):
        # split 파일에서 base 읽고 경로 조합
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

    # 평가 시 GT를 mmseg가 읽을 수 있게 2D로 제공 (학습과 동일 slice 규칙)
    def _read_2d_label(self, seg_map_filename):
        seg = np.asanyarray(nib.load(seg_map_filename).get_fdata())
        if seg.ndim == 3:
            idx = self.slice_index if self.slice_index is not None else seg.shape[-1] // 2
            seg = seg[..., idx]
        elif seg.ndim != 2:
            raise ValueError(f'Unexpected label shape {seg.shape} for {seg_map_filename}')
        return seg.astype(np.uint8)  # 0~3

    def get_gt_seg_map_by_idx(self, index):
        return self._read_2d_label(self.img_infos[index]['ann_info']['seg_map'])

    def get_gt_seg_maps(self, efficient_test=False):
        return [self.get_gt_seg_map_by_idx(i) for i in range(len(self.img_infos))]

    def prepare_test_img(self, idx):
        results = dict(
            img_info=self.img_infos[idx]['img_info'],
            ann_info=self.img_infos[idx]['ann_info'])
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx]['img_info'])
        if 'ann_info' in self.img_infos[idx]:
            results['ann_info'] = self.img_infos[idx]['ann_info']
        return self.pipeline(results)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann_info']
