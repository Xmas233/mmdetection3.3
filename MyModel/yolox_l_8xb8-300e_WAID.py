_base_ = '../configs/yolox/yolox_l_8xb8-300e_coco.py'
'''
python tools/train.py MyModel/yolox_l_8xb8-300e_WAID.py
python tools/analysis_tools/browse_dataset.py MyModel/yolox_l_8xb8-300e_WAID.py
'''

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
num_last_epochs = 15
interval = 5


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
    batch_size=2,
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
    batch_size=2,
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
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'annotations/valid_coco.json')
test_evaluator = val_evaluator
