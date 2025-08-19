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
    
    def get_gt_seg_maps(self, efficient_test=False):
        """평가용 GT 슬라이스 로딩.
        - load_annotations에서 저장한 절대경로(seg_map)를 그대로 사용
        - BraTS 라벨 4 -> 3으로 리맵 (배경 0은 그대로)
        """
        gts = []
        for i in range(len(self.img_infos)):
            info = self.img_infos[i]
            ann = info.get('ann_info', None)
            if ann is None:
                # 평가에 GT가 없는 경우는 보통 없음. 방어적으로 0맵을 넣고 continue
                # (원한다면 raise로 바꿔도 됩니다)
                img_path = info['img_info']['filename']
                z = info['img_info']['z_index']
                shape3d = nib.load(img_path).shape  # (H,W,S)
                H, W = shape3d[0], shape3d[1]
                gts.append(np.zeros((H, W), dtype=np.uint8))
                continue

            seg_path = ann['seg_map']          # 우리는 절대경로로 저장
            z = ann.get('z_index', info['img_info']['z_index'])

            lab3d = np.asanyarray(nib.load(seg_path).get_fdata()).astype(np.int32)
            sl = lab3d[..., int(z)]

            # BraTS 표준: 4(ET) -> 3
            sl[sl == 4] = 3

            gts.append(sl.astype(np.uint8))
        return gts
