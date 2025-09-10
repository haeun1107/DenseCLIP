import mmcv
import numpy as np
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class ISICDataset(CustomDataset):
    CLASSES = ('background', 'lesion')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)
        # ✅ 평가 시에도 0/255 → 0/1이 되도록 GT 로더를 교체
        self.gt_seg_map_loader = self._gt_loader_isic

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        with open(self.split, 'r') as f:
            stems = [x.strip() for x in f if x.strip()]
        data_infos = []
        for s in stems:
            data_infos.append(dict(
                img_info=dict(filename=f'{s}{img_suffix}'),
                ann=dict(seg_map=f'{s}{seg_map_suffix}')
            ))
        print(f'[ISICDataset] Loaded {len(data_infos)} samples from {self.split}')
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

    # ---------- ✅ 평가에서 사용하는 GT 로더(0/255 → 0/1) ----------
    def _gt_loader_isic(self, results):
        """results: {'ann_info': {'seg_map': ...}, 'seg_prefix': ...} 형태를 기대"""
        ann = results.get('ann_info', {})
        seg_map = ann.get('seg_map')
        seg_prefix = results.get('seg_prefix') or getattr(self, 'ann_dir', None)

        seg_path = seg_map if (seg_prefix is None or osp.isabs(seg_map)) \
                   else osp.join(seg_prefix, seg_map)

        mask = mmcv.imread(seg_path, flag='unchanged')
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)

        results['gt_semantic_seg'] = mask.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_semantic_seg')
        return results

    # (옵션) 파일명으로 직접 불러올 때도 동일 규칙 적용
    def get_gt_seg_map_by_filename(self, seg_map_filename):
        res = dict(ann_info=dict(seg_map=seg_map_filename),
                   seg_prefix=None)
        return self._gt_loader_isic(res)['gt_semantic_seg']
