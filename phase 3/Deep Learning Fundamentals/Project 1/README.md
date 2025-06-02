# Project 1: Neural Network from Scratch

Build and train a complete neural network from scratch without using deep learning frameworks.

## Objective
Implement a multi-layer perceptron (MLP) from the ground up to understand the mathematical foundations of neural networks, including forward propagation, backpropagation, and gradient descent.

## Key Concepts

### Mathematical Foundations:
- Matrix operations for neural network computations
- Forward propagation through multiple layers
- Backpropagation algorithm for gradient computation
- Chain rule application in neural networks

### Implementation Details:
- Modular design with separate classes for layers, activations, and optimizers
- Support for different activation functions (Sigmoid, ReLU, Tanh)
- Multiple loss functions (MSE, Cross-entropy)
- Mini-batch gradient descent implementation

## Features Implemented
- Configurable network architecture (hidden layers and neurons)
- Multiple activation functions with derivatives
- Different loss functions for regression and classification
- Learning rate scheduling and momentum
- Training progress visualization
- Model evaluation and testing

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Expected Results
- Working neural network trained on classification/regression tasks
- Convergence visualization showing loss reduction
- Understanding of gradient flow and backpropagation
- Performance comparable to framework implementations

## Applications
- Educational tool for understanding deep learning
- Baseline for custom neural network architectures
- Research into novel training algorithms
- Prototype development for specialized networks
