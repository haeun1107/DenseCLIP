_base_ = [
    '_base_/models/denseclip_r50.py',
    '_base_/datasets/isic_B0.py',
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

    # ---- Teacher (DenseCLIP) ----
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
    # transformer decoder = Vision-guided Prompt Adapter(VPA)
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
    # ---- B0 settings for freeze ----
    freeze_teacher=True,
    
    # ---- Student: from MaskCLIP+ ----
    # === MaskCLIP+ 노란블록: dilated ResNet ===
    aux_backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        dilations=(1,1,2,4),
        strides=(1,2,1,1),
        norm_cfg=dict(type='BN', requires_grad=True),
        contract_dilation=True),

    # === ASPP (features만 뽑기): classifier는 사용 안 함 ===
    aux_aspp=dict(
        type='ASPPHeadV2',      # classifier 쓰지 않고 forward_module만 사용
        in_channels=2048,
        in_index=3,
        channels=256,                # V의 채널
        dilations=(6,12,18,24),
        num_classes=NUM_CLASSES,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False
    ),

    # === DenseCLIP decoder 복제본: 입력 2개 [V, S’] ===
    student_head_aspp=dict( # 기존 decode_head_aspp였던 이름 student_head_aspp으로 수정
        type='FPNHead',              # DenseCLIP이 쓰는 decoder와 동일 계열
        in_channels=[256, 256],      # [V, S’]
        feature_strides=(8, 8),
        channels=256,
        num_classes=NUM_CLASSES,
        in_index=[0, 1],
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=0.5)
        ]),
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear', warmup_iters=1500, warmup_ratio=1e-6)


# Optimizer: teacher는 lr=0(동결), 학생 경로만 학습
optimizer = dict(
  type='AdamW', lr=1e-4, weight_decay=1e-4,
  paramwise_cfg=dict(custom_keys={
        'text_encoder':    dict(lr_mult=0.0),
        'decode_head':     dict(lr_mult=0.0),
        'context_decoder': dict(lr_mult=0.0),
        'neck':            dict(lr_mult=0.0),
        # 'backbone':        dict(lr_mult=0.0),
        'contexts':        dict(lr_mult=0.0),
        'gamma':           dict(lr_mult=0.0),
        'norm':            dict(decay_mult=0.0),
  })
)

data = dict(samples_per_gpu=4)
evaluation = dict(metric=['mIoU', 'mDice'])
device = 'cuda'
