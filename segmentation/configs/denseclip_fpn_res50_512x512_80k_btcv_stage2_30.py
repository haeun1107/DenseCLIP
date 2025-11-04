# DenseCLIP/segmentation/configs/denseclip_fpn_res50_512x512_80k_btcv_stage2.py
_base_ = [
    '_base_/models/denseclip_r50.py',
    '_base_/datasets/btcv_stage2_30.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_40k.py'
]

custom_imports = dict(
    imports=[
        'mmseg.datasets.btcv',                      # BTCVDataset 정의 파일
        'mmseg.datasets.pipelines.load_btcv_ann_A0',
        'mmseg.datasets.pipelines.load_btcv_ann_B0'
    ],
    allow_failed_imports=False
)

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
        in_channels=[256, 512, 1024, 2048 + 14],
        out_channels=256,
        num_outs=4),

    decode_head=dict(
        type='FPNHead',
        num_classes=14,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),

    # DenseCLIP text prompt 14 classes
    class_names=[
        'background',
        'spleen','kidney_right','kidney_left','gallbladder',
        'esophagus','liver','stomach','aorta','inferior_vena_cava',
        'portal_vein_and_splenic_vein','pancreas',
        'adrenal_gland_right','adrenal_gland_left'
    ],
)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone':         dict(lr_mult=1.0),  # ✅ update (image encoder)
            'text_encoder':     dict(lr_mult=0.0),  # ❌ freeze
            'context_decoder':  dict(lr_mult=0.0),  # ❌ freeze
            'neck':             dict(lr_mult=0.0),  # ❌ freeze (FPN)
            'decode_head':      dict(lr_mult=0.0),  # ❌ freeze (Seg head)
            'norm':             dict(decay_mult=0.) # norm no weight decay
        }
    )
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

data = dict(samples_per_gpu=4)
evaluation = dict(
    metric=['mIoU', 'mDice']
)
load_from = '/home/haeun1107/decs_jupyter_lab/DenseCLIP/work_dirs/denseclip_fpn_res50_512x512_80k_btcv_30/iter_40000.pth'
device = 'cuda'