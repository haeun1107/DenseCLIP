# segmentation/mmseg/datasets/pipelines/load_nifti_annotation.py
import os
import numpy as np
import nibabel as nib
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadNiftiImageFromFile:
    """Load a .nii/.nii.gz image and convert to 3-channel float32 (by stacking).
    - If image is 3D (H, W, D), pick middle slice unless 'slice_index' provided.
    - If image is 2D (H, W) or (H, W, 1), use as-is.
    Args:
        slice_index (int | None): If not None, use that slice for 3D inputs.
        clip (tuple|None): (min, max) intensity clipping before scaling.
        scale_to_uint8 (bool): If True, min-max to [0,255] then stack to 3ch.
    """
    def __init__(self, slice_index=None, clip=None, scale_to_uint8=True):
        self.slice_index = slice_index
        self.clip = clip
        self.scale_to_uint8 = scale_to_uint8

    def __call__(self, results):
        img_path = (results.get('img_prefix') or '') + results['img_info']['filename'] \
            if results.get('img_prefix') else results['img_info']['filename']
        nimg = nib.load(img_path)
        arr = np.asanyarray(nimg.get_fdata())  # float64

        # choose 2D slice
        if arr.ndim == 3:
            idx = self.slice_index if self.slice_index is not None else arr.shape[-1] // 2
            arr = arr[..., idx]
        elif arr.ndim == 4:  # (H,W,S,C) → C=0 then pick slice
            arr = arr[..., 0]
            idx = self.slice_index if self.slice_index is not None else arr.shape[-1] // 2
            arr = arr[..., idx]
        elif arr.ndim != 2:
            raise ValueError(f'Unexpected image shape {arr.shape} for {img_path}')

        if self.clip is not None:
            lo, hi = self.clip
            arr = np.clip(arr, lo, hi)

        if self.scale_to_uint8:
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = np.zeros_like(arr)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)

        img = np.stack([arr, arr, arr], axis=-1)  # (H,W,3)

        results['filename'] = img_path
        results['ori_filename'] = img_path
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        return results

@PIPELINES.register_module()
class LoadNiftiAnnotations:
    """Load a .nii/.nii.gz label map.
    - Supports 3D: choose middle slice (or 'slice_index').
    - Keeps class indices as uint8. 0=background for ACDC.
    Args:
        reduce_zero_label (bool): keep False for ACDC.
        slice_index (int | None)
    """
    def __init__(self, reduce_zero_label=False, slice_index=None):
        self.reduce_zero_label = reduce_zero_label
        self.slice_index = slice_index

    def __call__(self, results):
        ann_path = (results.get('seg_prefix') or '') + results['ann_info']['seg_map'] \
            if results.get('seg_prefix') else results['ann_info']['seg_map']
        nimg = nib.load(ann_path)
        seg = np.asanyarray(nimg.get_fdata())

        if seg.ndim == 3:
            idx = self.slice_index if self.slice_index is not None else seg.shape[-1] // 2
            seg = seg[..., idx]
        elif seg.ndim != 2:
            raise ValueError(f'Unexpected label shape {seg.shape} for {ann_path}')

        seg = seg.astype(np.uint8)

        if self.reduce_zero_label:
            seg = seg.copy()
            seg[seg == 0] = 255
            seg = seg - 1
            seg[seg == 254] = 255

        results['gt_semantic_seg'] = seg
        results['seg_fields'] = ['gt_semantic_seg']
        return results