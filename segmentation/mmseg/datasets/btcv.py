# MaskCLIP/mmseg/datasets/btcv.py
import os.path as osp
import os
import numpy as np
import json
from .builder import DATASETS
from .custom import CustomDataset
from scipy.sparse import load_npz

@DATASETS.register_module()
class BTCVDataset(CustomDataset):
    """BTCV Dataset with .npz label and CLIP feature .npy input."""

    CLASSES = [
        'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_vein_and_splenic_vein', 'pancreas',
        'adrenal_gland_right', 'adrenal_gland_left'
    ]

    PALETTE = [[i * 20, i * 20, i * 20] for i in range(13)]

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs) 
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split=None, **kwargs):
        with open(self.split, 'r') as f:
            lines = f.readlines()

        data_infos = []
        for line in lines:
            base = line.strip()
            data_infos.append(dict(
                img_info=dict(
                    filename=osp.join(img_dir, base + img_suffix),
                    img_prefix=None
                ),
                ann_info=dict(
                    seg_map=osp.join(ann_dir, base + seg_map_suffix),
                    seg_prefix=None
                )
            ))
        #print("[DEBUG] First sample in data_infos:", data_infos[0])
        print(f"[INFO] Loaded {len(data_infos)} of {len(lines)} samples (filtered missing npz).")
        return data_infos

    def get_gt_seg_map_by_filename(self, seg_map_filename):
        sparse = load_npz(seg_map_filename)  # (13, 262144) or (13, 512*512)
        dense = sparse.toarray()

        # (13, H*W) 또는 (13, H, W) 모두 대응
        if dense.ndim == 2 and dense.shape[0] == 13:
            bg_mask = (dense.sum(axis=0) == 0)              # 모든 채널 0이면 배경
            seg = np.argmax(dense, axis=0)                  # 0~12
            seg[bg_mask] = self.ignore_index               # ← 여기! 배경을 255로
            seg = seg.reshape(512, 512)
        elif dense.ndim == 3 and dense.shape[0] == 13:
            bg_mask = (dense.sum(axis=0) == 0)
            seg = np.argmax(dense, axis=0)
            seg[bg_mask] = self.ignore_index
        else:
            raise ValueError(f"Unexpected shape: {dense.shape} in {seg_map_filename}")

        return seg.astype(np.uint8)

        
    def get_gt_seg_map_by_idx(self, index):
        seg_map_path = self.img_infos[index]['ann_info']['seg_map']
        return self.get_gt_seg_map_by_filename(seg_map_path)

    def prepare_test_img(self, idx):
        results = dict(
            img_info=self.img_infos[idx]['img_info'],
            ann_info=self.img_infos[idx]['ann_info']
        )
        #print(f"[DEBUG] test results: {results}")
        return self.pipeline(results)


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