_base_ = '../configs/yolox/yolox_l_8xb8-300e_coco.py'
CUDA_LAUNCH_BLOCKING=1
'''
set CUDA_LAUNCH_BLOCKING=1
python tools/train.py MyModel/yolox_l_8xb8-300e_WAID.py
python tools/analysis_tools/browse_dataset.py MyModel/yolox_l_8xb8-300e_WAID.py
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/yolox_l_8xb8-300e_WAID/20250606_182252/vis_data/20250606_182252.json --keys loss --legend Loss
python tools/test.py MyModel/yolox_l_8xb8-300e_WAID.py work_dirs/yolox_l_8xb8-300e_WAID/epoch_100.pth --out yolox.pkl
python tools/analysis_tools/analyze_results.py MyModel/yolox_l_8xb8-300e_WAID.py yolox.pkl work_dirs/yolox_l_8xb8-300e_WAID/vis --show-score-thr 0.3 
'''
# 使用moco的预训练权重
load_from = 'checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'

# model settings
# TODO: change the backbone, neck, and bbox_head settings as needed
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(
        num_classes=6, 
        in_channels=256, 
        feat_channels=256,
))
# training settings
max_epochs = 50
num_last_epochs = 10
interval = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

base_lr = 0.0001
batch_size = 4


# dataset settings
data_root = 'data/COCO_WAID/'
dataset_type = 'CocoDataset'
metainfo = {
    'classes': ('sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra'),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 60, 100), (0, 80, 100),(0, 0, 230)
    ]
}

# 重写image_scale
img_scale = (640, 640)

# train和test的pipeline不知道是覆盖还是新增
# train_pipeline=[
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='LoadAnnotations', 
#         with_bbox=True,
#         with_mask=False,
#         poly2mask=False),
#     dict(
#         type='RandomFlip',
#         prob=0.5),
#     dict(type='PackDetInputs')
# ]


# test_pipeline = [  
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),  
#     dict(
#         type='PackDetInputs',  
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]



train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)
val_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/valid_coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                scale=img_scale,
                keep_ratio=True
            ),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'annotations/valid_coco.json')
test_evaluator = val_evaluator
