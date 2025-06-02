# Project 1: CNN from Scratch

Build a Convolutional Neural Network from scratch to understand the fundamental operations and architecture of CNNs.

## Objective
Implement all core components of a CNN including convolution layers, pooling layers, and fully connected layers without using high-level deep learning frameworks.

## Key Concepts

### CNN Components:
- **Convolution Layer**: Feature extraction using learnable filters
- **Pooling Layer**: Spatial dimension reduction and translation invariance
- **Activation Functions**: Non-linearity introduction (ReLU, Sigmoid)
- **Fully Connected Layer**: Final classification/regression

### Mathematical Operations:
- 2D convolution with stride and padding
- Backpropagation through convolutional layers
- Gradient computation for filters and biases
- Efficient implementation using vectorization

## Features Implemented
- Modular CNN architecture with configurable layers
- Forward propagation through all layer types
- Backpropagation with gradient computation
- Mini-batch training with momentum
- Visualization of learned filters and feature maps
- Training on MNIST or CIFAR-10 datasets

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Expected Results
- Working CNN trained on image classification
- Convergence to competitive accuracy (>95% on MNIST)
- Visualization of learned convolutional filters
- Understanding of CNN computational complexity

## Applications
- Educational foundation for understanding CNNs
- Baseline for custom CNN architectures
- Research into novel convolutional operations
- Performance optimization studies
