_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
actnn = True
data = dict(
    samples_per_gpu=16, # 16*1 = 16
    workers_per_gpu=8,
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
                name='faster_rcnn_r50_fpn_1x_coco_1gpu',
            )
        )
    ]
)
