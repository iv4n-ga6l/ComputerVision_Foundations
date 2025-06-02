# Project 5: Custom Dataset Classification

Build a complete image classification pipeline for custom datasets with data handling, augmentation, training, and deployment.

## Objective
Create a comprehensive image classification system that can handle custom datasets with automatic data preprocessing, augmentation, model training, and performance evaluation.

## Key Concepts

### Custom Dataset Handling:
- **Dataset Structure**: Organizing images in class folders
- **Data Loading**: Efficient data loading with PyTorch DataLoader
- **Train/Validation Split**: Proper data splitting strategies
- **Data Preprocessing**: Normalization and resizing

### Advanced Data Augmentation:
- **Geometric Transforms**: Rotation, scaling, cropping
- **Color Transforms**: Brightness, contrast, saturation
- **Noise Addition**: Gaussian noise, salt-pepper noise
- **Cutout/Mixup**: Advanced augmentation techniques

### Model Training Pipeline:
- **Transfer Learning**: Using pre-trained models
- **Custom Architectures**: Building task-specific models
- **Hyperparameter Tuning**: Grid search and optimization
- **Model Validation**: Cross-validation and metrics

## Features Implemented
- Automatic dataset analysis and visualization
- Comprehensive data augmentation pipeline
- Multiple model architectures support
- Advanced training techniques (mixup, cutmix)
- Real-time training monitoring
- Model evaluation and interpretation
- Export for deployment (ONNX, TorchScript)

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train on custom dataset
python main.py --data-dir ./my_dataset --model resnet50 --epochs 100

# With advanced augmentation
python main.py --data-dir ./my_dataset --model efficientnet --augmentation advanced

# Export trained model
python main.py --data-dir ./my_dataset --export-model --model-path best_model.pth
```

## Dataset Structure
```
my_dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── class1/
    └── class2/
```

## Expected Results
- High accuracy on custom datasets
- Robust model with good generalization
- Comprehensive analysis of model performance
- Ready-to-deploy model files

## Applications
- Custom image classification projects
- Industry-specific computer vision tasks
- Prototype development for new applications
