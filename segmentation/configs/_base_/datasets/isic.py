dataset_type = 'ISICDataset'
data_root = 'data/ISIC'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadISICAnnotations'),  # ← 커스텀 로더 사용 (0/255 -> 0/1)
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
    dict(type='LoadImageFromFile'),
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
        img_dir='ISIC2018_Task1-2_Training_Input',
        ann_dir='ISIC2018_Task1_Training_GroundTruth',
        split='splits/isic_task1_train_10.txt',
        img_suffix='.jpg',
        seg_map_suffix='_segmentation.png',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ISIC2018_Task1-2_Validation_Input',
        ann_dir='ISIC2018_Task1_Validation_GroundTruth',
        split='splits/isic_task1_val.txt',
        img_suffix='.jpg',
        seg_map_suffix='_segmentation.png',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ISIC2018_Task1-2_Test_Input',
        ann_dir='ISIC2018_Task1_Test_GroundTruth',  # 로컬에 GT가 있을 때만 지정
        split='splits/isic_task1_test.txt',
        img_suffix='.jpg',
        seg_map_suffix='_segmentation.png',
        pipeline=test_pipeline),
)
