# segmentation/mmseg/datasets/btcv.py
import os.path as osp
import numpy as np
from scipy.sparse import load_npz
from .builder import DATASETS
from .custom import CustomDataset
import os

@DATASETS.register_module()
class BTCVDataset(CustomDataset):
    CLASSES = [
        'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_vein_and_splenic_vein', 'pancreas',
        'adrenal_gland_right', 'adrenal_gland_left'
    ]
    PALETTE = [[i * 20, i * 20, i * 20] for i in range(13)]

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)
        # .npz를 직접 읽도록 교체
        self.gt_seg_map_loader = self._gt_loader_npz

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        with open(self.split, 'r') as f:
            lines = [x.strip() for x in f if x.strip()]

        data_infos = []
        for base in lines:
            # ✅ 상대 경로만 넣는다 (prefix는 나중에 top-level에)
            data_infos.append(dict(
                img_info=dict(filename=f'{base}{img_suffix}'),
                ann_info=dict(seg_map=f'{base}{seg_map_suffix}')
            ))
        print(f"[INFO] Loaded {len(data_infos)} samples.")
        return data_infos

    # ---------- .npz GT 로더 ----------
    def _gt_loader_npz(self, results):
        ann = results.get('ann_info', {})
        seg_map = ann.get('seg_map')
        seg_prefix = results.get('seg_prefix') or getattr(self, 'ann_dir', None)

        seg_path = seg_map if (seg_prefix is None or osp.isabs(seg_map)) \
                   else osp.join(seg_prefix, seg_map)

        sp = load_npz(seg_path)
        dense = sp.toarray()  # (C, H*W) or (C, H, W)
        C = len(self.CLASSES)

        if dense.ndim == 2 and dense.shape[0] == C:
            bg = (dense.sum(axis=0) == 0)
            seg = np.argmax(dense, axis=0)
            seg[bg] = self.ignore_index
            side = int(round(dense.shape[1] ** 0.5))
            seg = seg.reshape(side, side)
        elif dense.ndim == 3 and dense.shape[0] == C:
            bg = (dense.sum(axis=0) == 0)
            seg = np.argmax(dense, axis=0)
            seg[bg] = self.ignore_index
        else:
            raise ValueError(f"Unexpected GT shape {dense.shape} for {seg_path}")

        results['gt_semantic_seg'] = seg.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_semantic_seg')
        return results
    # ----------------------------------

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(
            img_info=info['img_info'],
            ann_info=info.get('ann_info', {})
        )
        # ✅ top-level prefix를 반드시 넣어준다
        results['img_prefix'] = getattr(self, 'img_dir', None) or getattr(self, 'img_prefix', None)
        results['seg_prefix'] = getattr(self, 'ann_dir', None)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'])
        if 'ann_info' in info:
            results['ann_info'] = info['ann_info']
        results['img_prefix'] = getattr(self, 'img_dir', None) or getattr(self, 'img_prefix', None)
        results['seg_prefix'] = getattr(self, 'ann_dir', None)
        return self.pipeline(results)

    # (옵션) 파일명으로 바로 GT 얻기
    def get_gt_seg_map_by_filename(self, seg_map_filename):
        res = dict(ann_info=dict(seg_map=seg_map_filename), seg_prefix=None)
        return self._gt_loader_npz(res)['gt_semantic_seg']

    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx]['img_info'])
        if 'ann_info' in self.img_infos[idx]:
            results['ann_info'] = self.img_infos[idx]['ann_info']
        return self.pipeline(results)
    
    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann_info']

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        if not os.path.exists(imgfile_prefix):
            os.makedirs(imgfile_prefix)

        if indices is None:
            indices = list(range(len(results)))

        saved_paths = []
        
        for i, idx in enumerate(indices):
            result = results[i]  # numpy array of shape (H, W)
            filename = self.img_infos[idx]['img_info']['filename']
            basename = os.path.splitext(os.path.basename(filename))[0]
            out_path = os.path.join(imgfile_prefix, f'{basename}.npz')

            # one-hot encode: [C, H, W]
            num_classes = kwargs.get('num_classes', 13)
            onehot = np.zeros((num_classes, *result.shape), dtype=np.uint8)
            for c in range(num_classes):
                onehot[c] = (result == c).astype(np.uint8)

            # Save as sparse npz
            from scipy import sparse
            sparse_matrix = sparse.csr_matrix(onehot.reshape(num_classes, -1))
            sparse.save_npz(out_path, sparse_matrix)
            saved_paths.append(out_path)

        print(f"[INFO] Saved predictions to {imgfile_prefix}")
        return saved_paths