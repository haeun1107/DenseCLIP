from mmcv import Config
from mmseg.datasets import build_dataset
import numpy as np

cfg = Config.fromfile('segmentation/configs/denseclip_fpn_res50_512x512_80k_acdc.py')
dval = build_dataset(cfg.data.val)

print('VAL reduce_zero_label:', dval.reduce_zero_label, 'CLASSES:', dval.CLASSES)

# 몇 개 인덱스 확인
for i in [0, 5, 10]:
    seg = dval.get_gt_seg_map_by_idx(i)
    print(i, 'unique labels:', np.unique(seg))  # 기대: [0 1 2 255] 중 일부
