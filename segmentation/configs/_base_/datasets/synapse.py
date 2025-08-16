dataset_type = 'SynapseNiftiDataset'
data_root = 'data/synapse'   # <- 네 폴더에 맞춰 조정

# 이미 파이프라인에서 CT 윈도잉 후 [0,255] uint8로 만들어서
# 여기 Normalize는 CLIP 사전학습 통계(이미 사용 중)를 그대로 사용 가능
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadNiftiSliceImage', window_min=-350.0, window_max=350.0),
    dict(type='LoadNiftiSliceAnnotations', reduce_zero_label=True),  # 0->255 ignore, 1..13->0..12
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
    dict(type='LoadNiftiSliceImage', window_min=-350.0, window_max=350.0),
    dict(type='LoadNiftiSliceAnnotations', reduce_zero_label=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
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
        img_dir='train/CT',
        ann_dir='train/GT',
        split='splits/train.txt',
        img_suffix='.nii.gz',
        seg_map_suffix='.nii.gz',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/CT',
        ann_dir='val/GT',
        split='splits/test.txt',
        img_suffix='.nii.gz',
        seg_map_suffix='.nii.gz',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/CT',
        ann_dir='val/GT',
        split='splits/test.txt',
        img_suffix='.nii.gz',
        seg_map_suffix='.nii.gz',
        pipeline=test_pipeline),
)
