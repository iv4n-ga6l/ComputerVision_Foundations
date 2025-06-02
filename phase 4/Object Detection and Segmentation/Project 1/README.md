# Project 1: YOLO Object Detection

## Overview
Implement and train a YOLO (You Only Look Once) object detection model for real-time object detection. This project covers the complete YOLO architecture, including anchor boxes, non-maximum suppression, and multi-scale detection.

## Learning Objectives
- Understand YOLO architecture and design principles
- Implement anchor boxes and grid-based detection
- Learn non-maximum suppression (NMS) algorithms
- Handle multi-scale object detection
- Evaluate detection performance using mAP metrics
- Optimize models for real-time inference

## Key Concepts
- **YOLO Architecture**: Single-stage detector that predicts bounding boxes and class probabilities directly
- **Grid-based Detection**: Divide image into grid cells, each responsible for detecting objects
- **Anchor Boxes**: Pre-defined boxes of different scales and aspect ratios
- **Non-Maximum Suppression**: Remove duplicate detections
- **Multi-scale Training**: Train on images of different sizes for robustness
- **Loss Functions**: Combination of localization, confidence, and classification losses

## Implementation Features
- Complete YOLOv5-style architecture implementation
- Support for multiple datasets (COCO, Pascal VOC, custom)
- Real-time detection with webcam integration
- Model training with data augmentation
- Performance evaluation with mAP metrics
- Model optimization and quantization
- Visualization tools for detections and training progress

## Dataset
- COCO dataset for comprehensive object detection
- Pascal VOC for comparison
- Custom dataset support with annotation tools

## Requirements
See `requirements.txt` for full dependencies.

## Usage
```bash
# Train model
python main.py --mode train --dataset coco --epochs 100

# Detect objects in images
python main.py --mode detect --input images/ --output results/

# Real-time detection
python main.py --mode realtime --source webcam

# Evaluate model
python main.py --mode evaluate --dataset coco --weights best.pt
```

## Expected Results
- mAP@0.5 > 0.4 on COCO validation set
- Real-time inference (>30 FPS) on modern GPUs
- Accurate detection of multiple object classes
- Robust performance across different image scales
