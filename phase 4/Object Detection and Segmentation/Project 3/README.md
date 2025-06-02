# Project 3: Instance Segmentation

## Overview
Implement Mask R-CNN for instance segmentation, combining object detection with pixel-level segmentation to detect and segment individual object instances. This project covers region-based approaches, feature pyramid networks, and mask prediction.

## Learning Objectives
- Understand instance segmentation vs semantic segmentation
- Implement Mask R-CNN architecture
- Learn region proposal networks (RPN)
- Master feature pyramid networks (FPN)
- Handle multi-task learning (detection + segmentation)
- Evaluate instance segmentation metrics

## Key Concepts
- **Instance Segmentation**: Detect and segment individual object instances
- **Mask R-CNN**: Extension of Faster R-CNN with segmentation branch
- **Region Proposal Network**: Generate object proposals
- **Feature Pyramid Network**: Multi-scale feature extraction
- **RoI Align**: Precise feature extraction for regions of interest
- **Multi-task Loss**: Combine classification, regression, and mask losses

## Implementation Features
- Complete Mask R-CNN implementation
- Support for COCO dataset and custom datasets
- Advanced data augmentation for instance segmentation
- Comprehensive evaluation metrics (AP, AR)
- Visualization tools for instances and masks
- Model optimization and inference acceleration

## Dataset
- COCO dataset for comprehensive instance segmentation
- Custom dataset support with annotation tools
- Data augmentation preserving instance relationships

## Requirements
See `requirements.txt` for full dependencies.

## Usage
```bash
# Train Mask R-CNN
python main.py --mode train --dataset coco --epochs 100

# Segment instances in images
python main.py --mode segment --input images/ --output results/

# Evaluate model
python main.py --mode evaluate --dataset coco --weights best.pt

# Create demo dataset
python main.py --mode create_demo --data_dir demo_data
```

## Expected Results
- AP@0.5 > 0.35 on COCO validation set
- High-quality instance masks
- Accurate object detection and segmentation
- Real-time inference on modern hardware
