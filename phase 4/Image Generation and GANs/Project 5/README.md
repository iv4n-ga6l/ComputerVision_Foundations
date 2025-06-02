# Project 5: Variational Autoencoders (VAE)

## Overview
Implement Variational Autoencoders for learning latent representations and generating new images. This project explores probabilistic generative models and the theoretical foundations of variational inference in deep learning.

## Learning Objectives
- Understand variational inference and the Evidence Lower Bound (ELBO)
- Implement reparameterization trick for backpropagation through stochastic nodes
- Learn about latent space interpolation and disentangled representations
- Apply VAEs to different datasets and architectures
- Compare VAEs with other generative models

## Key Concepts
- **Variational Inference**: Approximating intractable posteriors with learnable distributions
- **Evidence Lower Bound (ELBO)**: Objective function combining reconstruction and KL divergence
- **Reparameterization Trick**: Enabling gradient flow through stochastic sampling
- **Latent Space**: Learned continuous representation space
- **β-VAE**: Disentangled representation learning through weighted KL divergence

## Project Structure
```
Project 5/
├── README.md
├── main.py                 # Main VAE implementation and training
├── models/
│   ├── vae.py             # Standard VAE architecture
│   ├── beta_vae.py        # β-VAE for disentanglement
│   ├── cvae.py            # Conditional VAE
│   └── utils.py           # Model utilities
├── training/
│   ├── trainer.py         # VAE training loop
│   ├── losses.py          # VAE loss functions
│   └── metrics.py         # Evaluation metrics
├── evaluation/
│   ├── interpolation.py   # Latent space interpolation
│   ├── reconstruction.py  # Reconstruction quality
│   └── generation.py      # Sample generation
└── requirements.txt
```

## Key Features
1. **Multiple VAE Variants**
   - Standard VAE
   - β-VAE for disentanglement
   - Conditional VAE (CVAE)
   - Hierarchical VAE

2. **Comprehensive Analysis**
   - Latent space visualization
   - Interpolation capabilities
   - Reconstruction quality metrics
   - Disentanglement evaluation

3. **Advanced Features**
   - Normalizing flows for flexible priors
   - Importance weighted autoencoders (IWAE)
   - Vector Quantized VAE (VQ-VAE) components

4. **Interactive Tools**
   - Latent space exploration
   - Attribute manipulation
   - Style transfer in latent space

## Usage
```bash
# Train standard VAE
python main.py --model vae --dataset mnist --epochs 100

# Train β-VAE for disentanglement
python main.py --model beta_vae --beta 4.0 --dataset celeba --epochs 200

# Train conditional VAE
python main.py --model cvae --dataset cifar10 --epochs 150

# Generate new samples
python main.py --mode generate --model_path checkpoints/vae_final.pth --num_samples 100

# Interpolate in latent space
python main.py --mode interpolate --model_path checkpoints/vae_final.pth --image1 img1.jpg --image2 img2.jpg

# Evaluate reconstruction quality
python main.py --mode evaluate --model_path checkpoints/vae_final.pth --test_dataset test_data/
```

## Implementation Details
- Uses both convolutional and fully connected architectures
- Implements proper reparameterization trick
- Includes comprehensive loss functions (reconstruction + KL)
- Supports multiple datasets (MNIST, CIFAR-10, CelebA)
- Provides extensive evaluation and visualization tools
