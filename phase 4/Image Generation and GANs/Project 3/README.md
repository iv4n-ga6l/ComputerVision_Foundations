# Project 3: Conditional GAN for Controlled Image Generation

## Overview
Implement a Conditional Generative Adversarial Network (cGAN) that can generate images based on specific class labels or conditions. This project demonstrates how to control the generation process by conditioning both the generator and discriminator on additional information.

## Learning Objectives
- Understand conditional generation in GANs
- Implement label conditioning mechanisms
- Learn about embedding layers and concatenation strategies
- Apply conditional generation to different datasets
- Evaluate conditional generation quality

## Key Concepts
- **Conditional GANs**: Extension of GANs where generation is conditioned on additional information
- **Label Embedding**: Converting categorical labels into dense vector representations
- **Conditioning Strategies**: Different ways to incorporate conditions (concatenation, projection, etc.)
- **Auxiliary Classifier GANs (AC-GAN)**: Alternative approach with classification loss
- **Evaluation Metrics**: Inception Score, FID for conditional generation

## Project Structure
```
Project 3/
├── README.md
├── main.py                 # Main training and evaluation script
├── models/
│   ├── generator.py        # Conditional generator network
│   ├── discriminator.py    # Conditional discriminator network
│   └── utils.py           # Model utilities
├── data/
│   └── datasets.py        # Dataset handling for conditional generation
├── training/
│   ├── trainer.py         # Training loop with conditioning
│   └── losses.py          # GAN losses with conditioning
├── evaluation/
│   ├── metrics.py         # Conditional generation metrics
│   └── visualization.py   # Visualization tools
└── requirements.txt
```

## Key Features
1. **Multiple Conditioning Strategies**
   - Label embedding and concatenation
   - Projection-based conditioning
   - Auxiliary classifier approach

2. **Dataset Support**
   - MNIST with digit labels
   - CIFAR-10 with class labels
   - Custom attribute-based conditioning

3. **Advanced Architectures**
   - Spectral normalization
   - Self-attention mechanisms
   - Progressive growing

4. **Comprehensive Evaluation**
   - Class-specific generation quality
   - Inception Score per class
   - Interpolation between conditions

## Usage
```bash
# Basic conditional GAN training
python main.py --dataset cifar10 --epochs 100

# With specific conditioning strategy
python main.py --dataset mnist --conditioning projection --epochs 50

# Generate specific classes
python main.py --mode generate --classes 0,1,2 --num_samples 100

# Evaluate model
python main.py --mode evaluate --model_path checkpoints/best_model.pth
```

## Implementation Details
- Uses both embedding and projection-based conditioning
- Implements spectral normalization for training stability
- Supports multiple datasets with different label types
- Includes comprehensive evaluation metrics
- Provides visualization tools for conditional generation
