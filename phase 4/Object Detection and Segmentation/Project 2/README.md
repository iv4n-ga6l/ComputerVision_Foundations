# Project 2: Semantic Segmentation with U-Net

## Overview
Implement U-Net architecture for semantic segmentation, performing pixel-level classification to assign each pixel in an image to a specific class. This project covers encoder-decoder architectures, skip connections, and segmentation evaluation metrics.

## Learning Objectives
- Understand U-Net architecture and design principles
- Implement encoder-decoder networks with skip connections
- Learn semantic segmentation concepts and applications
- Master segmentation loss functions and evaluation metrics
- Handle class imbalance in segmentation tasks
- Visualize and interpret segmentation results

## Key Concepts
- **U-Net Architecture**: Encoder-decoder with skip connections for precise localization
- **Semantic Segmentation**: Assign class labels to every pixel in an image
- **Skip Connections**: Combine low-level and high-level features
- **Dice Loss**: Handle class imbalance in segmentation
- **IoU (Intersection over Union)**: Measure segmentation quality
- **Multi-class Segmentation**: Handle multiple object classes simultaneously

## Implementation Features
- Complete U-Net implementation with ResNet backbone option
- Support for multiple datasets (Cityscapes, Pascal VOC, medical images)
- Advanced loss functions (Dice, Focal, Combo)
- Comprehensive evaluation metrics (IoU, Dice, Pixel Accuracy)
- Data augmentation for segmentation
- Model visualization and interpretation tools
- Multi-scale training and inference

## Dataset
- Cityscapes for urban scene understanding
- Pascal VOC for general object segmentation
- Medical image datasets for specialized applications
- Custom dataset support with annotation tools

## Requirements
See `requirements.txt` for full dependencies.

## Usage
```bash
# Train U-Net model
python main.py --mode train --dataset cityscapes --epochs 100

# Segment images
python main.py --mode segment --input images/ --output results/

# Evaluate model
python main.py --mode evaluate --dataset cityscapes --weights best.pt

# Create demo dataset
python main.py --mode create_demo --data_dir demo_data
```

## Expected Results
- mIoU > 0.7 on Cityscapes validation set
- High-quality pixel-level predictions
- Robust performance across different object scales
- Clear class boundaries in segmentation masks
