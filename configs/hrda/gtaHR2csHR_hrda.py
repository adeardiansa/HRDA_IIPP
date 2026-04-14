# ---------------------------------------------------------------
# Custom config untuk inference gambar bebas
# Berdasarkan gtaHR2csHR_hrda.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 1

model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    scales=[1, 0.5],
    hr_crop_size=[256, 256],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]
    )
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,

    test=dict(
        type='CustomDataset',
        img_dir='data/custom_images',  # <== ganti sesuai folder gambarmu
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    )
)

# Optimizer, checkpoint, dll tidak dipakai saat inference, tapi tetap disertakan
optimizer_config = None
optimizer = dict(lr=6e-05)

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU')
fp16 = dict(loss_scale='dynamic')
