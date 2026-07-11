"""
将 LabelMe JSON 标注 (harder_label/) 转换为 YOLO segment 格式，
并加入 data/my_dataset/train 训练集。

LabelMe JSON → shapes[i].points = [[x1,y1],[x2,y2],...]
YOLO seg    → class_id x1_norm y1_norm x2_norm y2_norm ...

用法:
  python yolov8/convert_harder_label.py
"""
import json
import shutil
from pathlib import Path

HARDER_DIR = Path('/root/CarND-Advanced-Lane-Lines-master/harder_label')
TRAIN_IMG_DIR = Path('/root/CarND-Advanced-Lane-Lines-master/data/my_dataset/train/images')
TRAIN_LBL_DIR = Path('/root/CarND-Advanced-Lane-Lines-master/data/my_dataset/train/labels')

TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LBL_DIR.mkdir(parents=True, exist_ok=True)

json_files = sorted(HARDER_DIR.glob('*.json'))
print(f"找到 {len(json_files)} 个标注文件")

converted = 0
skipped = 0

for jf in json_files:
    img_name = jf.stem + '.jpg'
    img_path = HARDER_DIR / img_name

    if not img_path.exists():
        print(f"  ⚠ 跳过 {jf.name}: 找不到图片 {img_name}")
        skipped += 1
        continue

    with open(jf, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data.get('imageWidth', 1280)
    img_h = data.get('imageHeight', 720)

    yolo_lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        if len(points) < 3:
            continue

        class_id = 0  # lane

        norm_points = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
            norm_points.append(f"{nx:.6f}")
            norm_points.append(f"{ny:.6f}")

        line = f"{class_id} " + " ".join(norm_points)
        yolo_lines.append(line)

    if not yolo_lines:
        print(f"  ⚠ 跳过 {jf.name}: 没有有效的多边形标注")
        skipped += 1
        continue

    label_path = TRAIN_LBL_DIR / (jf.stem + '.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines) + "\n")

    dst_img = TRAIN_IMG_DIR / img_name
    shutil.copy2(img_path, dst_img)
    converted += 1

print(f"\n✅ 完成！转换了 {converted} 个标注，跳过了 {skipped} 个。")
print(f"   训练集图片目录: {TRAIN_IMG_DIR}")
print(f"   训练集标签目录: {TRAIN_LBL_DIR}")

img_count = len(list(TRAIN_IMG_DIR.glob('*.jpg')))
lbl_count = len(list(TRAIN_LBL_DIR.glob('*.txt')))
print(f"\n训练集总计: {img_count} 张图片, {lbl_count} 个标签文件")