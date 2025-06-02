# Project 1: Basic GAN Implementation

## Overview
Implement a basic Generative Adversarial Network (GAN) to understand the fundamental concepts of adversarial training, generator and discriminator networks, and the minimax game theory behind GANs.

## Learning Objectives
- Understand GAN architecture and theory
- Implement generator and discriminator networks
- Learn adversarial training procedures
- Handle training instability and mode collapse
- Evaluate generative model quality
- Visualize the training process

## Key Concepts
- **Adversarial Training**: Minimax game between generator and discriminator
- **Generator Network**: Maps random noise to realistic images
- **Discriminator Network**: Distinguishes real from generated images
- **Nash Equilibrium**: Theoretical convergence point
- **Mode Collapse**: When generator produces limited diversity
- **Training Stability**: Balancing generator and discriminator learning

## Implementation Features
- Complete GAN implementation from scratch
- Multiple dataset support (MNIST, CIFAR-10, CelebA)
- Training monitoring and visualization
- Loss curve analysis and metrics
- Generated sample progression tracking
- Hyperparameter tuning utilities

## Dataset
- MNIST for initial experimentation
- CIFAR-10 for more complex generation
- CelebA for face generation (optional)

## Requirements
See `requirements.txt` for full dependencies.

## Usage
```bash
# Train basic GAN on MNIST
python main.py --mode train --dataset mnist --epochs 100

# Generate samples
python main.py --mode generate --model_path best_gan.pt --num_samples 100

# Evaluate model
python main.py --mode evaluate --model_path best_gan.pt

# Monitor training
python main.py --mode visualize --log_dir logs/
```

## Expected Results
- Realistic digit generation for MNIST
- Stable training without mode collapse
- Smooth interpolation in latent space
- FID score < 50 for MNIST dataset
