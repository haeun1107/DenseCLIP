import os, h5py
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class SynapseNPZDataset(CustomDataset):
    """
    train_npz/*.npz 를 그대로 사용.
    split txt에는 파일명만 줄단위로 기록 (예: case0005_slice000.npz)
    """
    CLASSES = [
    "background", "aorta"," gallbladder",
    "left kidney", "right kidney", "liver", "pancreas" ,"spleen", "stomach"
    ]
    
    # CLASSES = (
    #     'background',
    #     'A medical scan showing the main abdominal aorta carrying blood',
    #     'A medical scan showing the small gallbladder beneath the liver',
    #     'A medical scan showing the left kidney filtering blood in the abdomen',
    #     'A medical scan showing the right kidney on the lower right abdomen',
    #     'A medical image showing the large liver in the upper abdomen',
    #     'A medical scan showing the pancreas located behind the stomach',
    #     'A medical scan showing the spleen on the upper left abdomen',
    #     'A medical image showing the stomach used for digestion'
    # )

    PALETTE = None

    def __init__(self, img_suffix='.npz', seg_map_suffix=None, **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        data_infos = []
        with open(split, 'r') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        for name in names:
            img_path = os.path.join(img_dir, name)
            data_infos.append(dict(
                img_info=dict(filename=img_path),
                filename=img_path,           # ★ top-level에도 백필
                ann=dict(seg_map=None)
            ))
        return data_infos

    def prepare_train_img(self, idx):          # ★ NPZ도 오버라이드 (filename 보장)
        info = self.img_infos[idx]
        results = dict(
            img_info=info.get('img_info', {}),
            ann_info=info.get('ann', {}),
        )
        # mmseg 표준 prefix 전파
        for k in ['img_prefix', 'seg_prefix', 'proposal_file', 'proposal_prefix']:
            v = getattr(self, k, None)
            if v is not None:
                results[k] = v

        # filename 백필: img_info/ top-level 모두 보장
        if 'filename' not in results['img_info']:
            if 'filename' in info:
                results['img_info']['filename'] = info['filename']
        if 'filename' not in results and 'filename' in info:
            results['filename'] = info['filename']

        results.setdefault('img_fields', [])
        results.setdefault('seg_fields', [])
        return self.pipeline(results)

    # (선택) 평가 시 NPZ를 쓸 수도 있으니 대비
    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(
            img_info=info.get('img_info', {}),
            ann_info=info.get('ann', {}),
        )
        for k in ['img_prefix', 'seg_prefix', 'proposal_file', 'proposal_prefix']:
            v = getattr(self, k, None)
            if v is not None:
                results[k] = v
        if 'filename' not in results['img_info'] and 'filename' in info:
            results['img_info']['filename'] = info['filename']
        if 'filename' not in results and 'filename' in info:
            results['filename'] = info['filename']
        results.setdefault('img_fields', [])
        results.setdefault('seg_fields', [])
        return self.pipeline(results)

    def get_gt_seg_maps(self, efficient_test=False):
        """평가용 GT 맵을 원본 NPZ에서 직접 읽어 반환."""
        gt_seg_maps = []
        for info in self.img_infos:
            npz_path = info['img_info']['filename']
            data = np.load(npz_path, allow_pickle=False)
            if 'label' not in data.files:
                raise KeyError(f"[SynapseNPZDataset] 'label' not found in {npz_path}")
            seg = data['label']
            # (1,H,W) -> (H,W)
            if seg.ndim == 3 and seg.shape[0] == 1:
                seg = seg[0]
            seg = seg.astype(np.int64)
            gt_seg_maps.append(seg)
        return gt_seg_maps
    
@DATASETS.register_module()
class SynapseH5SliceDataset(CustomDataset):
    """
    test_vol_h5/*.h5 를 slice 단위로 펼쳐서 사용.
    split txt에는 파일명만 기록 (예: case0001.npy.h5)
    → __init__에서 각 h5의 slice 수를 읽어 (path, slice_idx) 리스트 확장
    """
    CLASSES = SynapseNPZDataset.CLASSES
    PALETTE = None

    def __init__(self, img_suffix='.h5', **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=None, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        filelist = []
        with open(split, 'r') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        for name in names:
            path = os.path.join(img_dir, name)
            with h5py.File(path, 'r') as f:
                n_slices = f['image'].shape[0]
            for i in range(n_slices):
                filelist.append((path, i))

        data_infos = []
        for path, idx in filelist:
            data_infos.append(dict(
                img_info=dict(filename=path),
                ann=dict(seg_map=None),
                slice_idx=idx
            ))
        return data_infos

    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(
            img_info=info.get('img_info', {}),
            ann_info=info.get('ann', {}),
            slice_idx=info['slice_idx'],
        )
        for k in ['img_prefix', 'seg_prefix', 'proposal_file', 'proposal_prefix']:
            v = getattr(self, k, None)
            if v is not None:
                results[k] = v
        if 'filename' not in results['img_info'] and 'filename' in info:
            results['img_info']['filename'] = info['filename']
        if 'filename' not in results and 'filename' in info:
            results['filename'] = info['filename']
        results.setdefault('img_fields', [])
        results.setdefault('seg_fields', [])
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(
            img_info=info.get('img_info', {}),
            ann_info=info.get('ann', {}),
            slice_idx=info['slice_idx'],
        )
        for k in ['img_prefix', 'seg_prefix', 'proposal_file', 'proposal_prefix']:
            v = getattr(self, k, None)
            if v is not None:
                results[k] = v
        if 'filename' not in results['img_info'] and 'filename' in info:
            results['img_info']['filename'] = info['filename']
        if 'filename' not in results and 'filename' in info:
            results['filename'] = info['filename']
        results.setdefault('img_fields', [])
        results.setdefault('seg_fields', [])
        return self.pipeline(results)
    
    def get_gt_seg_maps(self, efficient_test=False):
        """평가용 GT 맵을 원본 H5에서 slice 인덱스로 직접 읽어 반환."""
        gt_seg_maps = []
        for info in self.img_infos:
            h5_path = info['img_info']['filename']
            idx = info['slice_idx']
            with h5py.File(h5_path, 'r') as f:
                if 'label' not in f:
                    raise KeyError(f"[SynapseH5SliceDataset] 'label' dataset not found in {h5_path}")
                seg = f['label'][idx]  # (H,W)
            seg = seg.astype(np.int64)
            gt_seg_maps.append(seg)
        return gt_seg_maps