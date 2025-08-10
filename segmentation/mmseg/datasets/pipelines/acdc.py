# mmseg/datasets/acdc.py
from mmseg.datasets.custom import CustomDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class ACDCDataset(CustomDataset):
    # background 제외 메타
    METAINFO = dict(
        classes=('right ventricle cavity', 'myocardium', 'left ventricle cavity'),
        palette=[[0, 0, 255], [255, 0, 0], [0, 255, 0]]
    )

    def __init__(self, **kwargs):
        # ACDC 파일명 규칙: *_frameXX.nii.gz / *_frameXX_gt.nii.gz
        super().__init__(
            img_suffix='.nii.gz',
            seg_map_suffix='_gt.nii.gz',
            reduce_zero_label=False,   # 배경 0 유지
            **kwargs
        )
