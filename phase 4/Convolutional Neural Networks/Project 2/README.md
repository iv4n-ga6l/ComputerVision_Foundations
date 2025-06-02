# Project 2: Image Classification with CNNs

Implement modern CNN architectures using deep learning frameworks for large-scale image classification tasks.

## Objective
Build and train state-of-the-art CNN models on challenging datasets like CIFAR-10/100 and ImageNet, implementing popular architectures and training techniques.

## Key Concepts

### Modern CNN Architectures:
- **LeNet**: Historical foundation of CNNs
- **AlexNet**: Deep CNN with ReLU and dropout
- **VGG**: Very deep networks with small filters
- **ResNet**: Residual connections and skip connections
- **DenseNet**: Dense connectivity patterns

### Training Techniques:
- Data augmentation strategies
- Learning rate scheduling
- Batch normalization
- Regularization methods (Dropout, Weight decay)
- Transfer learning and fine-tuning

## Features Implemented
- Multiple CNN architecture implementations
- Comprehensive data augmentation pipeline
- Advanced training loops with validation
- Model evaluation and performance metrics
- Visualization of training progress and results
- Model saving and loading capabilities

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --model resnet18 --dataset cifar10 --epochs 100
```

## Expected Results
- High accuracy on CIFAR-10 (>95%) and CIFAR-100 (>75%)
- Training convergence visualization
- Comparison of different architectures
- Analysis of learned representations

## Applications
- Production-ready image classification systems
- Baseline models for custom datasets
- Architecture comparison and selection
- Transfer learning foundation models
