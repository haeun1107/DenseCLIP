# segmentation/mmseg/datasets/pipelines/load_brats_slice.py

import os.path as osp
import numpy as np
import nibabel as nib
from mmseg.datasets.builder import PIPELINES

def _percentile_uint8_2d(sl: np.ndarray, p1=1, p99=99) -> np.ndarray:
    """2D 슬라이스에 퍼센타일 정규화 적용 → [0,255] uint8.
    통계 계산만 1D(flat)로 하고, 변환은 원본 2D에 적용한다.
    """
    sl = sl.astype(np.float32)
    flat = sl[np.isfinite(sl)]
    if flat.size == 0:
        return np.zeros_like(sl, dtype=np.uint8)

    lo, hi = np.percentile(flat, [p1, p99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(flat.min()), float(flat.max())
        if hi <= lo:
            return np.zeros_like(sl, dtype=np.uint8)

    out = (sl - lo) / (hi - lo + 1e-6)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


@PIPELINES.register_module()
class LoadBraTSSliceImage:
    def __init__(self, use_percentile=True, p1=1, p99=99):
        self.use_percentile = use_percentile
        self.p1, self.p99 = p1, p99

    def __call__(self, results):
        info = results['img_info']
        z = int(info['z_index'])

        vol_path = info['filename']
        img_prefix = results.get('img_prefix')
        if img_prefix and not osp.isabs(vol_path):
            vol_path = osp.join(img_prefix, vol_path)

        vol = nib.load(vol_path).get_fdata()  # (H,W,S)
        sl = vol[..., z]                      # (H,W)

        if self.use_percentile:
            img_u8 = _percentile_uint8_2d(sl, self.p1, self.p99)
        else:
            sl = sl.astype(np.float32)
            lo, hi = float(np.min(sl)), float(np.max(sl))
            if hi > lo:
                img_u8 = ((sl - lo) / (hi - lo) * 255.0).astype(np.uint8)
            else:
                img_u8 = np.zeros_like(sl, dtype=np.uint8)

        img = np.stack([img_u8, img_u8, img_u8], axis=-1)  # (H,W,3)

        results.update(
            filename=vol_path,
            ori_filename=vol_path,
            img=img,
            img_shape=img.shape,  # (H,W,3)
            ori_shape=img.shape,
            pad_shape=img.shape,
            scale_factor=1.0,
            img_fields=['img'],
        )
        return results

@PIPELINES.register_module()
class LoadBraTSSliceAnnotations:
    """
    BraTS 라벨 슬라이스 로더.
    - BraTS: 라벨값 {0,1,2,4} → 4를 3으로 매핑(기본)
    - 배경 포함 학습이므로 reduce_zero_label=False 권장
    Args:
        map_4_to_3 (bool): 라벨 4를 3으로 합치기 (기본 True)
        reduce_zero_label (bool): True면 0→255(ignore), 1..→-1 시프트. (기본 False)
        ignore_index (int): 무시 인덱스 값
    """
    def __init__(self, map_4_to_3=True, reduce_zero_label=False, ignore_index=255):
        self.map_4_to_3 = map_4_to_3
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def __call__(self, results):
        ann = results['ann_info']
        z = int(results['img_info']['z_index'])

        seg_path = ann['seg_map']
        seg_prefix = results.get('seg_prefix')
        if seg_prefix and not osp.isabs(seg_path):
            seg_path = osp.join(seg_prefix, seg_path)

        lab = np.asanyarray(nib.load(seg_path).get_fdata()).astype(np.int32)  # (H,W,S)
        sl = lab[..., z]  # (H,W)

        if self.map_4_to_3:
            sl[sl == 4] = 3  # BraTS 표준

        if self.reduce_zero_label:
            # 배경을 무시로 보낼 때만 사용 (이번 세팅에선 False 권장)
            zero = (sl == 0)
            sl = sl - 1
            sl[zero] = self.ignore_index

        results['gt_semantic_seg'] = sl.astype(np.uint8)
        results['seg_fields'] = ['gt_semantic_seg']
        return results
