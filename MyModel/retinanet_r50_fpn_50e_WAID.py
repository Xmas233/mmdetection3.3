_base_ = [
    '../configs/_base_/models/retinanet_r50_fpn.py',
    '../configs/_base_/datasets/WAID_detection.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py',
    '../configs/retinanet/retinanet_tta.py'
]

'''
python tools/train.py MyModel/retinanet_r50_fpn_50e_WAID.py --amp
python tools/analysis_tools/browse_dataset.py MyModel/retinanet_r50_fpn_50e_WAID.py
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/retinanet_r50_fpn_50e_WAID/20250608_154657/vis_data/20250608_154657.json --keys loss --legend Loss
python tools/test.py MyModel/retinanet_r50_fpn_50e_WAID.py work_dirs/retinanet_r50_fpn_50e_WAID/epoch_50.pth --out work_dirs/retinanet_r50_fpn_50e_WAID/retinanet_r50_fpn_50e_WAID.pkl --show-dir work_dirs/retinanet_r50_fpn_50e_WAID/vis
python tools/analysis_tools/analyze_results.py MyModel/retinanet_r50_fpn_50e_WAID.py work_dirs/retinanet_r50_fpn_50e_WAID/retinanet_r50_fpn_50e_WAID.pkl --eval bbox --format-only --show-dir work_dirs/retinanet_r50_fpn_50e_WAID/vis work_dirs/yolox_l_8xb8-300e_WAID/vis --show-score-thr 0.3 
python tools/analysis_tools/confusion_matrix.py MyModel/retinanet_r50_fpn_50e_WAID.py work_dirs/retinanet_r50_fpn_50e_WAID/retinanet_r50_fpn_50e_WAID.pkl work_dirs/retinanet_r50_fpn_50e_WAID --show
python demo/large_image_demo.py data/DJI.jpg MyModel/retinanet_r50_fpn_50e_WAID.py work_dirs/retinanet_r50_fpn_50e_WAID/epoch_50.pth
'''

max_epochs = 100
interval = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=interval)


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# model settings
model = dict(
    
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean: [107.720 126.924 131.645]
        # std: [35.672 36.244 38.385]
        mean=[107.720, 126.924, 131.645], #TODO: check if this is correct
        std=[35.672, 36.244, 38.385],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    bbox_head=dict(
        num_classes=6)
    )

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=interval, max_keep_ckpts=3),
)