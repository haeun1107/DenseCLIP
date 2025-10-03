# _base_/datasets/busi.py 

dataset_type = 'BUSIDataset'  
data_root = 'data/BUSI'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# BUSI는 352×352가 관례이므로 기본 352로, 필요하면 512로 변경 가능
crop_size = (352, 352)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadISICAnnotations'),   # ← ISIC에서 쓰던 0/255 → 0/1 커스텀 로더 재사용
    dict(type='Resize', img_scale=(352, 352), ratio_range=(0.5, 2.0)),
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
        img_scale=(352, 352),
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
        img_dir='images',
        ann_dir='masks',
        split='splits/busi_train_10.txt',      
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='splits/busi_val.txt',
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',                        # 로컬에 GT 없으면 이 줄/seg_map_suffix 제거 가능
        split='splits/busi_test.txt',
        img_suffix='.png',
        seg_map_suffix='_mask.png',
        pipeline=test_pipeline),
)
