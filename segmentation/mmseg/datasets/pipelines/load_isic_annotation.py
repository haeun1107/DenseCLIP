# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadISICAnnotations(object):
    """Load ISIC Task1 PNG masks and binarize them to {0,1}."""
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow',
                 reduce_zero_label=False,
                 suppress_labels=None):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.reduce_zero_label = reduce_zero_label
        self.suppress_labels = suppress_labels or []

    def __call__(self, results):
        results.setdefault('seg_fields', [])

        # ✅ test에서 GT가 없을 때 처리
        if results.get('ann') is None or results['ann'].get('seg_map') is None:
            ori_shape = results['ori_shape']
            results['gt_semantic_seg'] = np.ones(ori_shape[:2], dtype=np.int64) * -1
            results['seg_fields'].append('gt_semantic_seg')
            return results

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # ✅ ann 사용
        if results.get('seg_prefix') is not None:
            filename = osp.join(results['seg_prefix'], results['ann']['seg_map'])
        else:
            filename = results['ann']['seg_map']

        img_bytes = self.file_client.get(filename)
        mask = mmcv.imfrombytes(img_bytes, flag='unchanged',
                                backend=self.imdecode_backend).squeeze()

        # 다채널이면 첫 채널만
        if mask.ndim == 3:
            mask = mask[..., 0]

        # 이진화: >0 -> 1
        mask = (mask > 0).astype(np.uint8)

        # (옵션) reduce_zero_label
        if self.reduce_zero_label:
            mask[mask == 0] = 255
            mask = (mask - 1).astype(np.int64)
            mask[mask == 254] = 255

        # (옵션) 특정 라벨 무시
        if self.suppress_labels:
            for cid in self.suppress_labels:
                mask[mask == cid] = -1

        results['gt_semantic_seg'] = mask.astype(np.int64)
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(imdecode_backend={self.imdecode_backend}, '
                f'reduce_zero_label={self.reduce_zero_label})')
