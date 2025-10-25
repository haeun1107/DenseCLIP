# ---- Robust loaders for Synapse NPZ/H5 ----
import os
import os.path as osp
import numpy as np
import h5py
from mmseg.datasets.builder import PIPELINES


def _get_path_from_results(results):
    """results에서 이미지 경로를 안전하게 추출."""
    path = results.get('filename') or (results.get('img_info') or {}).get('filename')
    if path is None:
        raise KeyError("Loader: 'filename' not found in results or results['img_info']")
    # img_prefix가 있고 path가 상대경로면 prefix와 합침
    img_prefix = results.get('img_prefix')
    if img_prefix and not osp.isabs(path):
        path = osp.join(img_prefix, path)
    return path


def _ensure_3ch(img, repeat_to_3ch=True):
    """(H,W) 또는 (H,W,1) 이미지를 (H,W,3)로 맞춤."""
    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3 and img.shape[0] == 1:
        # (1,H,W)인 경우 (H,W,1)로 변환
        img = np.transpose(img, (1, 2, 0))
    if repeat_to_3ch and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


@PIPELINES.register_module()
class LoadSynapseNPZ:
    """train_npz/*.npz에서 image(및 선택적 label)를 로드."""
    def __init__(self, with_label=True, repeat_to_3ch=True):
        self.with_label = with_label
        self.repeat_to_3ch = repeat_to_3ch

    def __call__(self, results):
        path = _get_path_from_results(results)
        data = np.load(path, allow_pickle=False)

        # --- image ---
        img = data['image']
        img = _ensure_3ch(img, self.repeat_to_3ch).astype(np.float32)

        results['filename'] = path
        results.setdefault('ori_filename', path)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        # img_fields 누적
        if 'img_fields' in results and results['img_fields']:
            if 'img' not in results['img_fields']:
                results['img_fields'].append('img')
        else:
            results['img_fields'] = ['img']

        # --- label (optional) ---
        if self.with_label and ('label' in data.files):
            seg = data['label']
            # (1,H,W) -> (H,W)
            if seg.ndim == 3 and seg.shape[0] == 1:
                seg = seg[0]
            seg = seg.astype(np.int64)
            results['gt_semantic_seg'] = seg
            if 'seg_fields' in results and results['seg_fields']:
                if 'gt_semantic_seg' not in results['seg_fields']:
                    results['seg_fields'].append('gt_semantic_seg')
            else:
                results['seg_fields'] = ['gt_semantic_seg']

        return results


@PIPELINES.register_module()
class LoadSynapseH5Slice:
    """test_vol_h5/*.h5 에서 특정 slice를 (이미지, 라벨) 로드."""
    def __init__(self, with_label=True, repeat_to_3ch=True):
        self.with_label = with_label
        self.repeat_to_3ch = repeat_to_3ch

    def __call__(self, results):
        path = _get_path_from_results(results)
        if 'slice_idx' not in results:
            raise KeyError("LoadSynapseH5Slice: 'slice_idx' is required in results")
        idx = int(results['slice_idx'])

        with h5py.File(path, 'r') as f:
            # 데이터셋 키 이름은 'image' / 'label'을 기대
            if 'image' not in f:
                raise KeyError(f"'image' dataset not found in {path}")
            img = f['image'][idx]            # (H,W)
            seg = f['label'][idx] if (self.with_label and 'label' in f) else None

        img = _ensure_3ch(img, self.repeat_to_3ch).astype(np.float32)

        results['filename'] = f"{path}::slice{idx}"
        results.setdefault('ori_filename', path)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        # img_fields 누적
        if 'img_fields' in results and results['img_fields']:
            if 'img' not in results['img_fields']:
                results['img_fields'].append('img')
        else:
            results['img_fields'] = ['img']

        if seg is not None:
            seg = seg.astype(np.int64)       # (H,W)
            results['gt_semantic_seg'] = seg
            if 'seg_fields' in results and results['seg_fields']:
                if 'gt_semantic_seg' not in results['seg_fields']:
                    results['seg_fields'].append('gt_semantic_seg')
            else:
                results['seg_fields'] = ['gt_semantic_seg']

        return results
