# PromptBridge/mmsegmentation/mmseg/datasets/pipelines/load_btcv_ann_B0.py

import numpy as np
from mmseg.datasets.builder import PIPELINES
from scipy.sparse import load_npz

@PIPELINES.register_module()
class LoadNpzAnnotationsB0:
    """
    Pseudo-label loader for B0:
    - Input: A0 prediction npz (recommended: 14-channel one-hot or (H,W) label map 0..13)
    - Output: (H,W) integer label 0..13 (0=BG)  ← ★ keep BG
    """
    def __init__(self, suppress_labels=None):
        self.suppress_labels = suppress_labels or []

    def __call__(self, results):
        npz_path = results['ann_info']['seg_map']

        seg = None
        # 1) Handle dense npz (label map)
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'pred' in data:
                seg = data['pred']            # (H,W) 0..13
            elif 'arr_0' in data and data['arr_0'].ndim == 2:
                seg = data['arr_0']           # (H,W)
            elif 'arr_0' in data and data['arr_0'].ndim == 3:
                oh = data['arr_0']            # (K,H,W)
                seg = np.argmax(oh, axis=0)   # (H,W) 0..K-1
            else:
                raise KeyError
        except KeyError:
            # 2) Sparse one-hot (HW,K) or (K,HW)
            sp = load_npz(npz_path).toarray()
            if sp.shape[0] in (13,14):
                oh = sp.reshape(sp.shape[0], 512, 512)
            elif sp.shape[1] in (13,14):
                oh = sp.T.reshape(sp.shape[1], 512, 512)
            else:
                raise ValueError(f"Unexpected npz shape: {sp.shape}")
            seg = np.argmax(oh, axis=0)       # 0..K-1

        # ★ Do not shift/mask here (keep BG=0)
        seg = seg.astype(np.uint8)

        # (Optional) If there are classes to suppress, mask them to 255
        for c in self.suppress_labels:
            seg[seg == c] = 255

        results['gt_semantic_seg'] = seg
        results['seg_fields'] = ['gt_semantic_seg']
        return results