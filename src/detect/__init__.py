"""Custom YOLO detection pipeline — zero framework overhead, direct PyTorch."""

from src.detect.models import (
    DetectionModel,
    load_ultralytics_weights,
    yolo11s,
    yolo26s,
    yolov8s,
)

__all__ = [
    "DetectionModel",
    "load_ultralytics_weights",
    "yolo11s",
    "yolo26s",
    "yolov8s",
]
