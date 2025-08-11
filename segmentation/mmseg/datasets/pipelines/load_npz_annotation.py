import numpy as np
from mmseg.datasets.builder import PIPELINES
from scipy.sparse import load_npz
import os

@PIPELINES.register_module()
class LoadNpzAnnotations:
    def __init__(self, reduce_zero_label=True, suppress_labels=None):
        self.reduce_zero_label = reduce_zero_label # 0번 클래스를 255로 보내고 나머지를 -1 쉬프트
        self.suppress_labels = suppress_labels or [] # 무시하고 싶은 클래스 인덱스 목록(정수 리스트)

    def __call__(self, results):
        # 경로 추론을 유연하게 처리
        if 'ann_info' in results and 'seg_map' in results['ann_info']:
            npz_path = results['ann_info']['seg_map']
        else:
            # 'filename'이 'image/x/XXX.png'이면 'label/XXX.npz'로 추론
            basename = os.path.basename(results['filename']).replace('.png', '.npz')
            npz_path = os.path.join('data/BTCV/label', basename)

        # 희소 라벨 읽기(.npz)
        seg_sparse = load_npz(npz_path)
        seg_array = seg_sparse.toarray()

        # BTCV 원본은 보통 (13, 512*512) → (13, 512, 512)로 재배열 필요
        if seg_array.shape == (13, 512 * 512):
            seg_array = seg_array.reshape(13, 512, 512)

        if seg_array.shape[0] == 13:
            bg_mask = (seg_array.sum(axis=0) == 0) # 모든 채널 합이 0인 픽셀 = 라벨 없음 → 배경(무시) 처리
            seg = np.argmax(seg_array, axis=0) # 각 픽셀에서 원-핫이 가장 강한 채널(클래스) 인덱스
            seg[bg_mask] = 255 # 배경 픽셀은 무시 인덱스(255)
        else:
            raise ValueError(f"Unexpected shape: {seg_array.shape}")

        # reduce_zero_label이 True면 background를 255로 마스킹
        if self.reduce_zero_label:
            seg_zero_mask = (seg == 0)
            seg = seg - 1 
            seg[seg_zero_mask] = 255
            seg = seg.astype(np.uint8)

        seg = seg.astype(np.int16)

        if self.suppress_labels:
            for cls in self.suppress_labels:
                seg[seg == cls] = -1

        results['gt_semantic_seg'] = seg
        results['seg_fields'] = ['gt_semantic_seg']
        
        # unique, counts = np.unique(seg, return_counts=True)
        # print("GT 클래스 분포:", dict(zip(unique, counts)))
        return results
