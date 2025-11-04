import numpy as np
from mmseg.datasets.builder import PIPELINES
from scipy.sparse import load_npz
import os

@PIPELINES.register_module()
class LoadNpzAnnotationsA0:
    def __init__(self, reduce_zero_label=False, suppress_labels=None):
        self.reduce_zero_label = reduce_zero_label  # Send class 0 to 255 and shift all others by -1
        self.suppress_labels = suppress_labels or []  # List of class indices to ignore (integers)

    def __call__(self, results):
        npz_path = results['ann_info']['seg_map']

        # Load sparse label (.npz)
        seg_sparse = load_npz(npz_path)
        seg_array = seg_sparse.toarray()

        # BTCV original format is usually (13, 512*512) → reshape to (13, 512, 512)
        if seg_array.shape == (13, 512 * 512):
            seg_array = seg_array.reshape(13, 512, 512)

        if seg_array.shape[0] == 13:
            bg_mask = (seg_array.sum(axis=0) == 0)  # Pixels where all channels sum to 0 → unlabeled → background
            # Original:
            # seg = np.argmax(seg_array, axis=0)  # 0..12
            # seg[bg_mask] = 255

            # Modified: map BG=0, organs=1..13
            seg = np.argmax(seg_array, axis=0)      # 0..12 (organ indices)
            seg = seg + 1                           # 1..13
            seg[bg_mask] = 0                        # BG=0  ← include BG in training
            seg = seg.astype(np.uint8)
        else:
            raise ValueError(f"Unexpected shape: {seg_array.shape}")

        # If reduce_zero_label=True, mask background (0) as 255
        if self.reduce_zero_label:
            seg_zero_mask = (seg == 0)
            seg = seg - 1 
            seg[seg_zero_mask] = 255

        # Optionally suppress specific labels by masking them as 255
        if self.suppress_labels:
            for cls in self.suppress_labels:
                seg[seg == cls] = 255
        
        seg = seg.astype(np.uint8)

        results['gt_semantic_seg'] = seg
        results['seg_fields'] = ['gt_semantic_seg']
        
        return results