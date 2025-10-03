# mmseg/datasets/busi.py
import os.path as osp
import mmcv
import numpy as np
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class BUSIDataset(CustomDataset):
    """BUSI (Breast Ultrasound) binary segmentation (background, lesion).
    - images/: <stem>.png
    - masks/:  <stem>_mask.png
    - splits/: lines are filenames with or without extension (robust)
    """
    CLASSES = ('background', 'lesion')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)
        # 평가/시각화에서도 0/255 → 0/1로 읽히도록 GT 로더 교체
        self.gt_seg_map_loader = self._gt_loader_busi

    # ----- annotation list from txt -----
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        assert self.split is not None, 'split txt is required for BUSI'
        with open(self.split, 'r') as f:
            lines = [x.strip() for x in f if x.strip()]

        data_infos = []
        for name in lines:
            # txt에 확장자가 포함되어 있을 수도/없을 수도 있으니 처리
            if name.endswith(img_suffix):
                stem = name[:-len(img_suffix)]
            else:
                stem = name
            data_infos.append(dict(
                img_info=dict(filename=f'{stem}{img_suffix}'),
                ann=dict(seg_map=f'{stem}{seg_map_suffix}')
            ))
        print(f'[BUSIDataset] Loaded {len(data_infos)} samples from {self.split}')
        return data_infos

    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'], ann=info.get('ann', {}))
        results['img_prefix'] = getattr(self, 'img_dir', None) or getattr(self, 'img_prefix', None)
        results['seg_prefix'] = getattr(self, 'ann_dir', None)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'], ann=info.get('ann', {}))
        results['img_prefix'] = getattr(self, 'img_dir', None) or getattr(self, 'img_prefix', None)
        results['seg_prefix'] = getattr(self, 'ann_dir', None)
        return self.pipeline(results)

    # ---------- GT loader: BUSI masks (PNG, 0/255) → binary {0,1} ----------
    def _gt_loader_busi(self, results):
        """results expects keys: 'ann_info': {'seg_map': ...}, and 'seg_prefix'."""
        ann = results.get('ann_info', {})
        seg_map = ann.get('seg_map')
        seg_prefix = results.get('seg_prefix') or getattr(self, 'ann_dir', None)

        seg_path = seg_map if (seg_prefix is None or osp.isabs(seg_map)) \
                   else osp.join(seg_prefix, seg_map)

        mask = mmcv.imread(seg_path, flag='unchanged')
        if mask is None:
            raise FileNotFoundError(seg_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        # BUSI: lesion pixels >0, background 0 → binary
        mask = (mask > 0).astype(np.uint8)

        results['gt_semantic_seg'] = mask.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_semantic_seg')
        return results

    # (optional) direct filename API, same rule
    def get_gt_seg_map_by_filename(self, seg_map_filename):
        res = dict(ann_info=dict(seg_map=seg_map_filename), seg_prefix=None)
        return self._gt_loader_busi(res)['gt_semantic_seg']
