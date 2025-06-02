# Project 2: SVM Image Classification with HOG Features

Implement Support Vector Machine (SVM) for image classification using Histogram of Oriented Gradients (HOG) features.

## Objective
Build a robust image classifier that combines traditional feature extraction (HOG) with powerful machine learning (SVM) to recognize different object categories.

## Key Concepts

### HOG (Histogram of Oriented Gradients):
- Captures edge and gradient structure information
- Divides image into small cells and computes gradient orientations
- Robust to lighting changes and small deformations
- Commonly used for object detection (e.g., pedestrian detection)

### SVM (Support Vector Machine):
- Finds optimal hyperplane to separate classes
- Works well with high-dimensional feature spaces
- Effective with small to medium datasets
- Uses kernel functions for non-linear classification

## Features Implemented
- HOG feature extraction with configurable parameters
- Multi-class SVM classification
- Feature scaling and preprocessing
- Model evaluation with confusion matrix
- Support for custom datasets with directory structure

## Dataset Structure
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Expected Results
- Training accuracy: ~85-95% (depending on dataset complexity)
- Clear visualization of HOG features
- Confusion matrix showing per-class performance
- Saved model for future predictions

## Applications
- Object classification in robotics
- Quality control in manufacturing
- Medical image analysis
- Document classification
