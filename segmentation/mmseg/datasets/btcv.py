# mmseg/datasets/btcv.py
import os.path as osp
import numpy as np
from .builder import DATASETS
from .custom import CustomDataset
from scipy.sparse import load_npz

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
        return data_infos

    def get_gt_seg_map_by_filename(self, seg_map_filename):
        sparse = load_npz(seg_map_filename)
        dense = sparse.toarray()
        if dense.shape == (13, 512 * 512):
            dense = dense.reshape(13, 512, 512)
        if dense.shape[0] == 13:
            background = np.zeros_like(dense[0:1])
            dense = np.vstack([background, dense])
        if dense.shape[0] != 14:
            raise ValueError(f"Unexpected shape: {dense.shape} in {seg_map_filename}")
        seg = np.argmax(dense, axis=0).astype(np.uint8)
        seg[seg == 0] = 255
        seg = seg - 1
        seg[seg == 254] = 255
        return seg

    def get_gt_seg_maps(self, efficient_test=False):
        gt_seg_maps = []
        for i in range(len(self.img_infos)):
            seg_map = self.get_gt_seg_map_by_idx(i)
            gt_seg_maps.append(seg_map)
        return gt_seg_maps

    def get_gt_seg_map_by_idx(self, index):
        seg_map_path = self.img_infos[index]['ann_info']['seg_map']
        return self.get_gt_seg_map_by_filename(seg_map_path)

    def prepare_test_img(self, idx):
        results = dict(
            img_info=self.img_infos[idx]['img_info'],
            ann_info=self.img_infos[idx]['ann_info']
        )
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        results = dict(img_info=self.img_infos[idx]['img_info'])
        if 'ann_info' in self.img_infos[idx]:
            results['ann_info'] = self.img_infos[idx]['ann_info']
        return self.pipeline(results)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann_info']