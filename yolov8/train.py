from ultralytics import YOLO

model = YOLO('models/yolov8m-seg.pt')

model.train(
    data='data/my_dataset/dataset.yaml',
    epochs=30,                 # 无早停，跑满30轮
    imgsz=640,
    batch=32,                  # 降低batch，适应显存
    lr0=0.001,
    lrf=0.1,                  # 更平缓的衰减
    optimizer='AdamW',
    weight_decay=0.001,       # 抑制过拟合
    device=0,
    project='runs',
    name='yolov8m_lane',
    exist_ok=True,
    augment=True,
    amp=True,                 # 混合精度，节省显存
    patience=0,               # 关闭早停
    save=True,
    verbose=True,
    workers=16,
    mosaic=0.0,               # 数据少，保持关闭
    # 可额外添加：
    # scale=0.5, translate=0.1, erasing=0.4,  # 这些都是默认值，不需要重复写
)