import os
import json
from tqdm import tqdm
from PIL import Image

def read_classes(class_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def convert(img_dir, label_dir, class_file, output_json):
    categories = []
    class_names = read_classes(class_file)
    for i, name in enumerate(class_names):
        categories.append({'id': i, 'name': name, 'supercategory': 'none'})

    images, annotations = [], []
    ann_id = 1
    img_id = 1
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.lower().endswith('.jpg'):
            continue
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        # 读取图片宽高
        with Image.open(img_path) as img:
            width, height = img.size
        images.append({
            'file_name': img_name,
            'id': img_id,
            'width': width,
            'height': height
        })
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x, y, w, h = map(float, parts)
                    # 反归一化
                    x_c = x * width
                    y_c = y * height
                    w_pixel = w * width
                    h_pixel = h * height
                    x_min = x_c - w_pixel / 2
                    y_min = y_c - h_pixel / 2
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': int(cls_id),
                        'bbox': [
                            x_min, y_min, w_pixel, h_pixel
                        ],
                        'area': w_pixel * h_pixel,
                        'iscrowd': 0,
                        'segmentation': []
                    })
                    ann_id += 1
        img_id += 1

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # 修改为你的实际路径
    img_dir = '../WAID/WAID/images/test'
    label_dir = '../WAID/WAID/labels/test'
    class_file = '../WAID/WAID/classes.txt'
    output_json = 'data/WAID/test_coco.json'
    convert(img_dir, label_dir, class_file, output_json)