dataset_type_train = 'SynapseNPZDataset'
dataset_type_eval  = 'SynapseH5SliceDataset'
data_root = 'data/Synapse'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadSynapseNPZ', with_label=True),     # 새 로더
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
    dict(type='LoadSynapseH5Slice'),                  # 새 로더
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
        type=dataset_type_train,
        data_root=data_root,
        img_dir='train_npz',   # 절대경로 허용
        ann_dir=None,
        split='splits/train_10.txt',
        img_suffix='.npz',
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type_eval,
        data_root=data_root,
        img_dir='test_vol_h5',
        ann_dir=None,
        split='splits/test.txt',
        img_suffix='.h5',
        ignore_index=0,
        pipeline=test_pipeline
    ),

    test=dict(
        type=dataset_type_eval,
        data_root=data_root,
        img_dir='test_vol_h5',
        ann_dir=None,
        split='splits/test.txt',
        img_suffix='.h5',
        ignore_index=0,
        pipeline=test_pipeline
    ),
)
