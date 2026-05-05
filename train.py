"""
YOLOv8s Training Script for Chicken Wings Detection
Single-class detection with optimized hyperparameters.
"""

import os
from ultralytics import YOLO


def main():
    # Resolve absolute path to data.yaml so Ultralytics resolves
    # the relative train/val/test paths correctly (Roboflow default format)
    data_path = os.path.abspath(os.path.join("dataset", "data.yaml"))

    # Load YOLO26s pretrained on COCO (transfer learning)
    model = YOLO("yolo26s.pt")

    # Train with optimized settings for single-class small-object detection
    model.train(
        data=data_path,
        epochs=50,
        imgsz=512,
        batch=16,
        patience=5,                # early stopping after 5 epochs with no improvement
        device=0,                  # GPU 0; use "cpu" if no GPU

        # Optimizer settings
        optimizer="auto",              # automatically choose optimizer (AdamW for small datasets)
        lr0=0.01,                  # initial learning rate
        lrf=0.01,                  # final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights (tuned for single-class detection)
        box=7.5,                   # box loss weight
        cls=0.5,                   # classification loss weight (lower since single class)
        dfl=1.5,                   # distribution focal loss weight

        # Data augmentation
        hsv_h=0.015,               # hue augmentation
        hsv_s=0.7,                 # saturation augmentation
        hsv_v=0.4,                 # value augmentation
        degrees=10.0,              # rotation
        translate=0.1,             # translation
        scale=0.5,                 # scale augmentation
        fliplr=0.5,                # horizontal flip probability
        flipud=0.0,                # vertical flip (off for wings)
        mosaic=1.0,                # mosaic augmentation
        mixup=0.1,                 # mixup augmentation
        copy_paste=0.1,            # copy-paste augmentation

        # Other settings
        cos_lr=True,               # cosine learning rate scheduler
        close_mosaic=10,           # disable mosaic for last 10 epochs
        amp=True,                  # automatic mixed precision
        single_cls=False,          # already single class in data.yaml
        cache="disk",              # cache images to disk for deterministic training
        workers=8,
        seed=42,
        verbose=True,
        plots=True,

        # Project output
        project="runs",
        name="chicken_wings",
        exist_ok=True,
    )

    # Validate
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export best model info
    best_path = "runs/chicken_wings/weights/best.pt"
    print(f"\nBest model saved to: {best_path}")
    print("Copy best.pt to the project root to use with the counting app.")


if __name__ == "__main__":
    main()
