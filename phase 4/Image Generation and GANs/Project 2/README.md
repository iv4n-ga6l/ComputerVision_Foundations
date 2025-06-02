# Project 2: DCGAN for Image Generation

## Overview
Implement Deep Convolutional GAN (DCGAN) with architectural best practices for stable training and high-quality image generation. This project focuses on convolutional architectures and training techniques for better results.

## Learning Objectives
- Understand DCGAN architecture improvements
- Implement convolutional generator and discriminator
- Learn training stabilization techniques
- Master architectural guidelines for GANs
- Generate high-resolution images
- Compare with basic GAN performance

## Key Concepts
- **Deep Convolutional Architecture**: Use convolutions instead of fully connected layers
- **Batch Normalization**: Stabilize training and improve convergence
- **Leaky ReLU**: Better gradient flow in discriminator
- **Transposed Convolutions**: Upsampling in generator
- **Training Techniques**: Learning rate scheduling, gradient clipping
- **Spectral Normalization**: Control discriminator capacity

## Usage
```bash
# Train DCGAN
python main.py --mode train --dataset cifar10 --epochs 200

# Generate high-quality samples
python main.py --mode generate --model_path best_dcgan.pt --num_samples 100
```
