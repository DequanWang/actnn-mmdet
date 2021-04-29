_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
conv_cfg = dict(type='QConv2d')
norm_cfg = dict(type='QBN2d', requires_grad=True)
act_cfg = dict(type='QReLU')
# model settings
model = dict(
    backbone=dict(
        frozen_stages=-1,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        norm_eval=False),
    neck=dict(
        conv_cfg=conv_cfg,
        norm_cfg=None,
        act_cfg=None),
    bbox_head=dict(
        conv_cfg=conv_cfg,
        norm_cfg=None,
        act_cfg=act_cfg),
)
# optimizer
data = dict(samples_per_gpu=16, workers_per_gpu=8)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='actnn',
                project='detection',
                name='retinanet_r50_fpn_1x_coco_1gpu',
            )
        )
    ]
)
