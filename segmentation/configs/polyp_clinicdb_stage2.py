dataset_type = 'POLYPDataset'
data_root = 'data/Polyp_new'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline_gt = [
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

train_pipeline_pseudo = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadISICAnnotations'),
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
    train=[
        dict(  # 10% GT
            type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='masks',
            split='splits/train_clinicdb_10.txt',
            img_suffix='.png',
            seg_map_suffix='.png',
            pipeline=train_pipeline_gt
        ),
        dict(  # 90% pseudo
            type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='pseudo',
            split='splits/train_clinicdb_90.txt',
            img_suffix='.png',
            seg_map_suffix='.png',
            pipeline=train_pipeline_pseudo
        ),
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='splits/test_clinicdb.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks', 
        split='splits/test_clinicdb.txt',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline),
)
