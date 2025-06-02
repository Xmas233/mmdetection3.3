import os
import cv2
import json

from pycocotools.coco import COCO

# 配置路径
img_dir = 'data/COCO_WAID/train/'  # 图片文件夹
ann_file = 'data/COCO_WAID/annotations/train_coco.json'  # 标注文件

# 加载 COCO 标注
coco = COCO(ann_file)
catid2name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图片: {img_path}')
        continue

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)
        cat_name = catid2name[ann['category_id']]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('COCO Visualization', img)
    key = cv2.waitKey(0)
    if key == 27:  # 按ESC退出
        break

cv2.destroyAllWindows()