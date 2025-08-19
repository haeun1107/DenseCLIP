# mmseg/datasets/brats.py
import os.path as osp
import numpy as np
import nibabel as nib
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BraTSNiftiDataset(CustomDataset):
    """BraTS NIfTI 3D 볼륨을 2D 슬라이스 표본으로 전개.
       - split 파일에는 케이스 폴더명(BraTS19_xxx_yyy_1) 한 줄씩.
       - img_dir/ann_dir 모두 같은 루트(dataset/)를 가리키되
         파일명은 '<stem>/<stem>_flair.nii(.gz)', '<stem>/<stem>_seg.nii(.gz)'
       - 파이프라인에서 LoadBraTSSliceImage / LoadBraTSSliceAnnotations 사용.
    """

    # background 포함 4클래스
    CLASSES = ('background', 'NCR_NET', 'ED', 'ET')
    PALETTE = [[0,0,0], [255,0,0], [0,255,0], [0,0,255]]

    def __init__(self, split, use_label_only_slices=False, **kwargs):
        self.use_label_only_slices = use_label_only_slices
        super().__init__(split=split, **kwargs)

    @staticmethod
    def _find(path_nii, path_niigz):
        if osp.exists(path_niigz): return path_niigz
        if osp.exists(path_nii):   return path_nii
        return None

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split=None, **kwargs):
        with open(self.split, 'r') as f:
            stems = [x.strip() for x in f if x.strip()]

        infos = []
        for stem in stems:
            case_dir = osp.join(self.img_dir, stem)
            flair_p = self._find(osp.join(case_dir, f'{stem}_flair.nii'),
                                 osp.join(case_dir, f'{stem}_flair.nii.gz'))
            seg_p   = self._find(osp.join(case_dir, f'{stem}_seg.nii'),
                                 osp.join(case_dir, f'{stem}_seg.nii.gz'))
            if flair_p is None or seg_p is None:
                print(f'[WARN] missing flair/seg for {stem}, skip'); continue

            lbl3d = np.asanyarray(nib.load(seg_p).get_fdata())
            # 슬라이스 선택
            if self.use_label_only_slices:
                sel = np.where(np.any(lbl3d > 0, axis=(0,1)))[0]
            else:
                sel = np.arange(lbl3d.shape[2])

            for z in sel.tolist():
                infos.append(dict(
                    img_info=dict(filename=flair_p, z_index=z),
                    ann_info=dict(seg_map=seg_p, z_index=z)
                ))

        print(f'[BraTSNiftiDataset] volumes: {len(stems)}, slices: {len(infos)} '
              f'({"label-only" if self.use_label_only_slices else "all"})')
        return infos

    # mmseg 기본 prepare_*는 prefix 붙이므로, 우리는 절대경로를 넘겨 그대로 사용
    def prepare_train_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'])
        if 'ann_info' in info:
            results['ann_info'] = info['ann_info']
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        results = dict(img_info=info['img_info'])
        if 'ann_info' in info:
            results['ann_info'] = info['ann_info']
        return self.pipeline(results)

    def get_ann_info(self, idx):
        return self.img_infos[idx].get('ann_info', {})
