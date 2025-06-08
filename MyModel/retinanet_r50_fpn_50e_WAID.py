_base_ = [
    '../configs/_base_/models/retinanet_r50_fpn.py',
    '../configs/_base_/datasets/WAID_detection.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py',
    '../configs/retinanet/retinanet_tta.py'
]

'''
python tools/train.py MyModel/retinanet_r50_fpn_50e_WAID.py
python tools/analysis_tools/browse_dataset.py MyModel/retinanet_r50_fpn_50e_WAID.py
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/yolox_l_8xb8-300e_WAID/20250606_182252/vis_data/20250606_182252.json --keys loss --legend Loss
python tools/test.py MyModel/retinanet_r50_fpn_50e_WAID.py work_dirs/yolox_l_8xb8-300e_WAID/epoch_100.pth --out yolox.pkl
python tools/analysis_tools/analyze_results.py MyModel/retinanet_r50_fpn_50e_WAID.py yolox.pkl work_dirs/yolox_l_8xb8-300e_WAID/vis --show-score-thr 0.3 
'''

max_epochs = 50
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
        mean=[123.675, 116.28, 103.53], #TODO: check if this is correct
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    bbox_head=dict(
        num_classes=6)
    )