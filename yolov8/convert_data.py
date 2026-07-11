"""
YOLOv8 数据格式转换：将图片 + 标签组织到 my_dataset 训练目录。

用法:
  python yolov8/convert_data.py
"""
import shutil
from pathlib import Path

SRC_IMG_DIR  = Path('data/train_from_video/images')
SRC_LBL_DIR  = Path('data/train_from_video/labels')
DST_IMG_DIR  = Path('data/my_dataset/train/images')
DST_LBL_DIR  = Path('data/my_dataset/train/labels')

DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DST_LBL_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_IMG_DIR.exists():
    print("源目录不存在，跳过。")
else:
    for f in sorted(SRC_IMG_DIR.glob('*.*')):
        shutil.copy2(f, DST_IMG_DIR / f.name)
    for f in sorted(SRC_LBL_DIR.glob('*.txt')):
        shutil.copy2(f, DST_LBL_DIR / f.name)
    print(f"已同步 {len(list(SRC_IMG_DIR.iterdir()))} 张图片到训练集")
