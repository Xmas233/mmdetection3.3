#!/usr/bin/env python3
# inc_coco_catid_fixed.py
#
# 功能：将 COCO 注释文件中所有 category_id 加 1，
#       同时同步更新 categories 数组里的 id。
# ---------------------------------------------------

import json
from pathlib import Path

# === 1. 写死文件路径（自行修改） =====================
input_path  = Path(r"D:\SDH\Detection\mmdetection3.3\data\COCO_WAID\annotations\train_coco.json")       # 原始 COCO 注释
output_path = Path(r"D:\SDH\Detection\mmdetection3.3\data\COCO_WAID\annotations\train_coco.json")   # 输出文件
# ===================================================

def inc_category_id(coco: dict) -> dict:
    """把 annotations 与 categories 中的 ID 全部 +1"""
    for ann in coco.get("annotations", []):
        if isinstance(ann.get("category_id"), int):
            ann["category_id"] += 1
    for cat in coco.get("categories", []):
        if isinstance(cat.get("id"), int):
            cat["id"] += 1
    return coco

def main():
    # 读取
    with input_path.open('r', encoding='utf-8') as f:
        coco = json.load(f)

    # 处理
    coco = inc_category_id(coco)

    # 保存
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成：{len(coco.get('annotations', []))} 条标注，"
          f"{len(coco.get('categories', []))} 个类别。输出已写入 {output_path}")

if __name__ == "__main__":
    main()
