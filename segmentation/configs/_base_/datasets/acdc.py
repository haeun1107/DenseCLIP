# configs/_base_/datasets/acdc.py
dataset_type = 'ACDCDataset'
data_root = 'data/ACDC'

img_norm_cfg = dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadNiftiImageFromFile', slice_index=None, clip=None, scale_to_uint8=True),
    dict(type='LoadNiftiAnnotations', reduce_zero_label=True, slice_index=None),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadNiftiImageFromFile', slice_index=None, clip=None, scale_to_uint8=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size, flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_root=data_root,
        img_dir='training', ann_dir='training',
        split='splits/train_10.txt',
        img_suffix='.nii.gz', seg_map_suffix='_gt.nii.gz',
        slice_index=None,
        reduce_zero_label=True,                           
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type, data_root=data_root,
        img_dir='testing', ann_dir='testing',
        split='splits/val.txt',
        img_suffix='.nii.gz', seg_map_suffix='_gt.nii.gz',
        slice_index=None,
        reduce_zero_label=True,  
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type, data_root=data_root,
        img_dir='testing', ann_dir='testing',
        split='splits/val.txt',
        img_suffix='.nii.gz', seg_map_suffix='_gt.nii.gz',
        slice_index=None,
        reduce_zero_label=True,  
        pipeline=test_pipeline
    ),
)
