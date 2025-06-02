# Project 2: Image Classification with Multi-Layer Perceptron (MLP)

## Overview
This project implements a multi-layer perceptron (MLP) neural network for image classification using the CIFAR-10 dataset. It demonstrates the fundamental concepts of deep learning applied to computer vision tasks, including network architecture design, training procedures, and performance evaluation.

## Key Components

### 1. Multi-Layer Perceptron Architecture
- **Fully connected layers**: Dense neural network structure
- **Activation functions**: ReLU, Sigmoid, Tanh comparisons
- **Output layer**: Softmax for multi-class classification
- **Configurable architecture**: Customizable hidden layer sizes

### 2. CIFAR-10 Dataset Processing
- **Data loading**: Automatic CIFAR-10 dataset handling
- **Preprocessing**: Normalization and flattening for MLP input
- **Train/validation split**: Proper data partitioning
- **Class visualization**: Dataset exploration and analysis

### 3. Training Pipeline
- **Loss function**: Categorical crossentropy for classification
- **Optimization**: Adam optimizer with learning rate scheduling
- **Metrics tracking**: Accuracy, loss, and validation metrics
- **Batch processing**: Efficient mini-batch training

### 4. Performance Analysis
- **Training curves**: Loss and accuracy visualization
- **Confusion matrix**: Class-wise performance analysis
- **Classification report**: Precision, recall, F1-score metrics
- **Model comparison**: Different architectures evaluation

## Technical Features

### Network Design
- **Input layer**: Flattened 32x32x3 CIFAR-10 images (3072 features)
- **Hidden layers**: Configurable dense layers with dropout
- **Activation functions**: ReLU activation with optional variations
- **Output layer**: 10-class softmax for CIFAR-10 categories

### Training Strategies
- **Data augmentation**: Basic image transformations
- **Regularization**: Dropout and weight decay
- **Learning rate scheduling**: Adaptive learning rate
- **Early stopping**: Prevent overfitting with validation monitoring

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Per-class metrics**: Individual class performance analysis
- **Confusion matrix**: Detailed error analysis
- **Training efficiency**: Convergence speed and stability

## Learning Objectives
- Understand MLP architecture for image classification
- Learn proper dataset handling and preprocessing
- Implement training loops with validation
- Analyze model performance with various metrics
- Compare different network configurations

## Applications
- **Image recognition**: Basic computer vision tasks
- **Pattern classification**: General classification problems
- **Feature learning**: Understanding neural network representations
- **Baseline models**: Foundation for more complex architectures

## Key Concepts Demonstrated
- **Forward propagation**: Signal flow through neural networks
- **Backpropagation**: Gradient computation and weight updates
- **Batch processing**: Efficient training with mini-batches
- **Regularization**: Preventing overfitting in neural networks
- **Performance evaluation**: Comprehensive model assessment

This project provides a solid foundation for understanding deep learning principles applied to computer vision, serving as a stepping stone to more advanced architectures like CNNs.
