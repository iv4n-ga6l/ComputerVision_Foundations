# Project 4: CNN Architectures Comparison

Compare different CNN architectures to understand their strengths, weaknesses, and performance characteristics across various tasks.

## Objective
Implement and compare multiple CNN architectures including LeNet, AlexNet, VGG, ResNet, DenseNet, and EfficientNet to analyze their performance, efficiency, and characteristics.

## Key Concepts

### Classic Architectures:
- **LeNet**: Pioneer of modern CNNs
- **AlexNet**: Deep CNN revolution
- **VGG**: Very deep networks with small kernels
- **ResNet**: Residual connections for very deep networks
- **DenseNet**: Dense connectivity patterns
- **EfficientNet**: Compound scaling methodology

### Comparison Metrics:
- **Accuracy**: Classification performance
- **Parameters**: Model size and complexity
- **FLOPs**: Computational requirements
- **Training Time**: Convergence speed
- **Memory Usage**: Resource requirements
- **Inference Speed**: Real-time performance

## Features Implemented
- Multiple CNN architecture implementations
- Comprehensive benchmarking framework
- Performance profiling and analysis
- Memory and computation tracking
- Visualization of architectural differences
- Trade-off analysis (accuracy vs efficiency)

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Compare all architectures
python main.py --compare-all --dataset cifar10 --epochs 50

# Compare specific models
python main.py --models resnet18 efficientnet_b0 --dataset cifar100 --epochs 30
```

## Expected Results
- Performance comparison across architectures
- Analysis of parameter efficiency
- Training time and convergence comparison
- Memory usage profiling
- Recommendations for different use cases

## Applications
- Architecture selection for new projects
- Understanding CNN design principles
- Performance optimization guidance
