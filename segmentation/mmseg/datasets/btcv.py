# 1) mmsegmentation/mmseg/datasets/btcv.py -> use 1) code
import os.path as osp
import numpy as np
from .builder import DATASETS
from .custom import CustomDataset
from scipy.sparse import load_npz

@DATASETS.register_module()
class BTCVDataset(CustomDataset):
    
    CLASSES = [
        'background',
        'spleen', 'kidney_right', 'kidney_left', 'gallbladder',
        'esophagus', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
        'portal_vein_and_splenic_vein', 'pancreas',
        'adrenal_gland_right', 'adrenal_gland_left'
    ]
    PALETTE = [[i*20, i*20, i*20] for i in range(14)]  # length 14
    
    # changing for gpt generated prompts
    # CLASSES = [
    #     "the spleen: a soft, fist-sized organ that filters blood and helps fight infections in the immune system.",
    #     "the right kidney: a bean-shaped organ located in the right side of the abdomen, responsible for filtering waste from blood and producing urine.",
    #     "the left kidney: a bean-shaped organ located in the left side of the abdomen that maintains fluid balance and removes toxins from the body.",
    #     "the gallbladder: a small, pear-shaped organ under the liver that stores bile to aid in fat digestion.",
    #     "the esophagus: a muscular tube connecting the throat to the stomach, allowing food and liquids to pass through.",
    #     "the liver: a large reddish-brown organ that detoxifies chemicals, metabolizes drugs, and produces bile for digestion.",
    #     "the stomach: a muscular, hollow organ that breaks down food using digestive acids and enzymes.",
    #     "the aorta: the largest artery in the body that carries oxygen-rich blood from the heart to the rest of the body.",
    #     "the inferior vena cava: a large vein that carries deoxygenated blood from the lower body back to the heart.",
    #     "the portal and splenic veins: blood vessels that transport nutrient-rich blood from the gastrointestinal tract and spleen to the liver.",
    #     "the pancreas: an elongated gland behind the stomach that helps with digestion and regulates blood sugar levels.",
    #     "the right adrenal gland: a small gland sitting above the right kidney that produces hormones like adrenaline and cortisol.",
    #     "the left adrenal gland: a hormone-secreting gland located above the left kidney, involved in stress response and metabolism."
    # ]

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
        dense = sparse.toarray()  # (13, H*W) or (13, H, W)

        if dense.ndim == 2 and dense.shape[0] == 13:
            hw = dense.shape[1]
            side = int(hw ** 0.5)
            assert side * side == hw, f"Non-square map @ {seg_map_filename}"
            dense = dense.reshape(13, side, side)

        if dense.shape[0] != 13:
            raise ValueError(f"Unexpected shape: {dense.shape} in {seg_map_filename}")

        # Apply the same rule as in training
        bg_mask = (dense.sum(axis=0) == 0)
        seg = np.argmax(dense, axis=0).astype(np.uint8)  # 0..12 (organs)
        seg = seg + 1                                    # 1..13
        seg[bg_mask] = 0                                 # BG=0  ← include BG in evaluation as well
        return seg  # 0..13

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