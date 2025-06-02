# Project 3: Transfer Learning

Implement transfer learning using pre-trained models for efficient training on custom datasets.

## Objective
Leverage pre-trained CNN models to achieve high performance on new tasks with limited data, demonstrating the power of transfer learning and fine-tuning strategies.

## Key Concepts

### Transfer Learning Strategies:
- **Feature Extraction**: Freeze pre-trained layers, train only classifier
- **Fine-tuning**: Unfreeze some layers and train with low learning rate
- **Progressive Fine-tuning**: Gradually unfreeze layers during training
- **Domain Adaptation**: Adapt models from different domains

### Pre-trained Models:
- **ResNet**: Deep residual networks
- **EfficientNet**: Efficient scaling of CNNs
- **Vision Transformer**: Attention-based architectures
- **DenseNet**: Dense connectivity patterns

## Features Implemented
- Multiple pre-trained model implementations
- Feature extraction and fine-tuning modes
- Custom dataset handling with data augmentation
- Progressive unfreezing strategies
- Model comparison and performance analysis
- Visualization of learned features

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Feature extraction mode
python main.py --mode extract --model resnet50 --dataset custom --epochs 20

# Fine-tuning mode
python main.py --mode finetune --model efficientnet --dataset custom --epochs 50
```

## Expected Results
- Fast convergence with feature extraction (few epochs)
- High accuracy with limited training data
- Comparison of different transfer learning strategies
- Analysis of which layers to fine-tune

## Applications
- Custom image classification with limited data
- Domain adaptation for specialized datasets
- Rapid prototyping of vision systems
