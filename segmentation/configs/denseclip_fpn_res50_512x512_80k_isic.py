_base_ = [
    '_base_/models/denseclip_r50.py',
    '_base_/datasets/isic.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_80k.py'
]

custom_imports = dict(
    imports=[
        'mmseg.datasets.isic',                       # Dataset class
        'mmseg.datasets.pipelines.load_isic_annotation'  # Custom loader
    ],
    allow_failed_imports=False
)

NUM_CLASSES = 2  # background, lesion

model = dict(
    type='DenseCLIP',
    pretrained='segmentation/pretrained/RN50.pt',
    context_length=12,
    text_head=False,

    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=512,
        style='pytorch'),

    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=16,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),

    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048 + NUM_CLASSES],  # +2 for (bg, lesion)
        out_channels=256,
        num_outs=4),

    decode_head=dict(
        type='FPNHead',
        num_classes=NUM_CLASSES,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear', warmup_iters=1500, warmup_ratio=1e-6)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'text_encoder': dict(lr_mult=0.0),
        'norm': dict(decay_mult=0.)
    }))

data = dict(samples_per_gpu=4)
evaluation = dict(metric=['mIoU', 'mDice'])
device = 'cuda'
