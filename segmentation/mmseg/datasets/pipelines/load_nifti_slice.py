# segmentation/mmseg/datasets/pipelines/load_nifti_slice.py
import os.path as osp
import numpy as np
import nibabel as nib
from mmseg.datasets.builder import PIPELINES

def _windowing(x, wmin=-350.0, wmax=350.0):
    x = np.clip(x, wmin, wmax)
    x = (x - wmin) / (wmax - wmin)  # [0,1]
    x = (x * 255.0).astype(np.uint8)  # [0,255]
    return x

@PIPELINES.register_module()
class LoadNiftiSliceImage:
    """NIfTI 3D에서 지정 z-슬라이스 하나를 로드해 3채널로 복제.
    - results['img_info']['filename'] 는 보통 'img0001.nii.gz' (상대)
    - results['img_prefix'] 와 합쳐 실제 경로를 만든다.
    """
    def __init__(self, window_min=-350.0, window_max=350.0, to_rgb=True):
        self.wmin = window_min
        self.wmax = window_max
        self.to_rgb = to_rgb

    def __call__(self, results):
        info = results['img_info']
        z = info['z_index']

        # prefix 존중 (상대/절대 모두 안전)
        vol_path = info['filename']
        img_prefix = results.get('img_prefix')
        if img_prefix and not osp.isabs(vol_path):
            vol_path = osp.join(img_prefix, vol_path)

        nimg = nib.load(vol_path)
        arr = np.asanyarray(nimg.get_fdata())  # (H,W,S)
        sl = arr[..., z].astype(np.float32)
        sl = _windowing(sl, self.wmin, self.wmax)  # (H,W) uint8

        img = np.stack([sl, sl, sl], axis=-1)  # (H,W,3)

        results['filename'] = vol_path
        results['ori_filename'] = vol_path
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadNiftiSliceAnnotations:
    """NIfTI 3D 라벨에서 동일 z-슬라이스 로드.
    - background(0)을 그대로 둘지/무시(255)로 보낼지는 reduce_zero_label로 제어.
    """
    def __init__(self, reduce_zero_label=False, ignore_index=255):
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def __call__(self, results):
        ann = results['ann_info']
        z = results['img_info']['z_index']

        seg_path = ann['seg_map']
        seg_prefix = results.get('seg_prefix')
        if seg_prefix and not osp.isabs(seg_path):
            seg_path = osp.join(seg_prefix, seg_path)

        nlab = nib.load(seg_path)
        lab = np.asanyarray(nlab.get_fdata()).astype(np.int32)  # (H,W,S)
        sl = lab[..., z]  # (H,W)

        if self.reduce_zero_label:
            zero = (sl == 0)
            sl = sl - 1           # 1..13 -> 0..12
            sl[zero] = self.ignore_index  # 0 -> 255(ignore)

        results['gt_semantic_seg'] = sl.astype(np.uint8)
        results['seg_fields'] = ['gt_semantic_seg']
        return results
