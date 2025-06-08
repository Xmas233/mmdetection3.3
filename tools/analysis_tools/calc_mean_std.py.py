import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
"""
calc_mean_std.py

功能：统计指定图片文件夹下所有图片的均值(mean)和标准差(std)，用于深度学习数据预处理。

使用方法：
    python calc_mean_std.py <img_dir>
参数说明：
    <img_dir>  图片文件夹路径（支持jpg/png/jpeg）

示例：
    python calc_mean_std.py D:/SDH/Detection/mmdetection3.3/data/COCO_WAID/train
"""



def calc_mean_std(img_dir):
    img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    mean = np.zeros(3)
    std = np.zeros(3)
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path).astype(np.float32)
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
    mean /= len(img_list)
    std /= len(img_list)
    print('mean:', mean)
    print('std:', std)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python calc_mean_std.py <img_dir>")
        sys.exit(1)
    calc_mean_std(sys.argv[1])