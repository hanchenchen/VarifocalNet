_base_ = './vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# DATA
CATE_ID = '4'

classes_dict = {'1': 'visible body', '2': 'full body', '3': 'head', '4': 'vehicle'}
json_pre_dict = {'1': 'person_visible', '2': 'person_full', '3': 'person_head', '4':'vehicle'}

data_root = 'DATA/split_' + json_pre_dict[CATE_ID].split('_')[0] +'_train/'
anno_root = 'DATA/coco_format_json/'

classes = (classes_dict[CATE_ID],)
json_pre = json_pre_dict[CATE_ID]

dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_train.json',
        img_prefix=data_root + 'image_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=anno_root + json_pre + '_val.json',
        img_prefix=data_root + 'image_train',
        pipeline=test_pipeline))
        
# default_runtime
load_from = "./cpt/vfnet_r2_101_dcn_ms_2x_51.1.pth"
resume_from = None

# schedule_2x
# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

