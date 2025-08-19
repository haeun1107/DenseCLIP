# configs/_base_/datasets/brats.py
dataset_type = 'BraTSNiftiDataset'
data_root = 'data/BraTS' 

# CLIP 통계 그대로 사용 (입력은 0~255 uint8로 변환 후 Normalize)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    # FLAIR만 읽어서 percentile(1,99) 정규화 → uint8 → 3채널 복제
    dict(type='LoadBraTSSliceImage', modality='flair'),
    # seg에서 4→3 리맵, background 포함 (reduce_zero_label=False)
    dict(type='LoadBraTSSliceAnnotations', reduce_zero_label=False, map_4_to_3=True),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadBraTSSliceImage', modality='flair'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset',   
        ann_dir='dataset',
        split='splits/train.txt',
        img_suffix='_flair.nii',    
        seg_map_suffix='_seg.nii',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset',
        ann_dir='dataset',
        split='splits/val.txt',
        img_suffix='_flair.nii',    
        seg_map_suffix='_seg.nii',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset',
        ann_dir='dataset',
        split='splits/test.txt',
        img_suffix='_flair.nii',    
        seg_map_suffix='_seg.nii',
        pipeline=test_pipeline,
    ),
)
