from mmseg.datasets.builder import PIPELINES
from scipy.sparse import load_npz
import numpy as np
import os

@PIPELINES.register_module()
class LoadNpzAnnotations:
    def __init__(self, reduce_zero_label=True, suppress_labels=None):
        self.reduce_zero_label = reduce_zero_label
        self.suppress_labels = suppress_labels or []

    def __call__(self, results):
        seg_path = results['ann_info']['seg_map']
        sparse = load_npz(seg_path)
        seg_array = sparse.toarray()

        if seg_array.shape == (13, 512 * 512):
            seg_array = seg_array.reshape(13, 512, 512)

# True일 때
        # if seg_array.shape[0] == 13:
        #     # Add background channel (zeros)
        #     background = np.zeros_like(seg_array[0:1])  # (1, 512, 512)
        #     seg_array = np.vstack([background, seg_array])  # → (14, 512, 512)

        # if seg_array.shape[0] != 14:
        #     raise ValueError(f"Unexpected shape: {seg_array.shape} in {seg_path}")

        if seg_array.shape[0] != 13:
            raise ValueError(f"Expected shape (13, H, W), got {seg_array.shape} in {seg_path}")
        
        seg = np.argmax(seg_array, axis=0).astype(np.uint8)

        if self.reduce_zero_label:
            zero_mask = (seg == 0)
            seg = seg - 1
            seg[zero_mask] = 255
            seg = seg.astype(np.uint8)

        if self.suppress_labels:
            for cls in self.suppress_labels:
                seg[seg == cls] = 255

        results['gt_semantic_seg'] = seg
        results['seg_fields'] = ['gt_semantic_seg']
        return results