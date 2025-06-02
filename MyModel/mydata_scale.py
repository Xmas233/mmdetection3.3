import os
from PIL import Image
from collections import Counter

img_dir = 'data/COCO_WAID/train/'  # 替换为你的图片文件夹路径

sizes = []
for fname in os.listdir(img_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(img_dir, fname)
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception as e:
            print(f'无法读取图片: {img_path}, 错误: {e}')

counter = Counter(sizes)
print('图片分辨率统计（宽,高）：')
for size, count in counter.most_common():
    print(f'{size}: {count} 张')

if sizes:
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    print(f'最大宽: {max(widths)}, 最小宽: {min(widths)}, 平均宽: {sum(widths)//len(widths)}')
    print(f'最大高: {max(heights)}, 最小高: {min(heights)}, 平均高: {sum(heights)//len(heights)}')
else:
    print('未统计到有效图片尺寸。')