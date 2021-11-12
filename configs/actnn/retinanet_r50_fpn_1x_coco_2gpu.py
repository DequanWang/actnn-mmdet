_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
actnn = True
data = dict(
    samples_per_gpu=8, # 8*2 = 16
    workers_per_gpu=4,
)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(
        mode='agc',
        clip_factor=0.01,
        eps=1e-3,
        norm_type=2.0,
    )
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='detection',
                entity='actnn',
                name='retinanet_r50_fpn_1x_coco_2gpu',
            )
        )
    ]
)