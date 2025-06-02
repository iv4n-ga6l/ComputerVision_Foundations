"""
Basic GAN Implementation
=======================

A comprehensive implementation of a basic Generative Adversarial Network
to understand the fundamentals of adversarial training and image generation.

Features:
- Complete GAN implementation from scratch
- Multi-dataset support (MNIST, CIFAR-10, CelebA)
- Training monitoring and visualization
- Loss analysis and evaluation metrics
- Sample generation and interpolation

 Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import seaborn as sns
from PIL import Image
import json
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(self, latent_dim=100, img_channels=1, img_size=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate initial size for fully connected layer
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    
    def __init__(self, img_channels=1, img_size=28):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class ImprovedGenerator(nn.Module):
    """Improved generator with better architecture"""
    
    def __init__(self, latent_dim=100, img_channels=1, img_size=28, features_g=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Calculate the size after all upsampling layers
        self.init_size = img_size // 8
        
        # Initial projection
        self.project = nn.Sequential(
            nn.Linear(latent_dim, features_g * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        # Upsampling layers
        self.main = nn.Sequential(
            # First upsampling
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # Second upsampling
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # Third upsampling
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # Output layer
            nn.Conv2d(features_g, img_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        # Project and reshape
        out = self.project(z)
        out = out.view(-1, 64 * 8, self.init_size, self.init_size)
        
        # Apply convolutions
        out = self.main(out)
        
        # Resize to exact target size if needed
        if out.size(-1) != self.img_size:
            out = torch.nn.functional.interpolate(out, size=(self.img_size, self.img_size), 
                                                mode='bilinear', align_corners=False)
        
        return out

class ImprovedDiscriminator(nn.Module):
    """Improved discriminator with better architecture"""
    
    def __init__(self, img_channels=1, img_size=28, features_d=64):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        self.main = nn.Sequential(
            # First layer (no batch norm)
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second layer
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth layer
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply convolutions
        out = self.main(x)
        
        # Flatten
        return out.view(-1, 1).squeeze(1)

class GANTrainer:
    """Training pipeline for GAN"""
    
    def __init__(self, generator, discriminator, device, latent_dim=100):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.latent_dim = latent_dim
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        self.fixed_noise = torch.randn(64, latent_dim, device=device)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_G_loss = 0
        epoch_D_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for i, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Create labels
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
            
            # ========================
            # Train Discriminator
            # ========================
            self.optimizer_D.zero_grad()
            
            # Train with real images
            real_output = self.discriminator(real_images)
            real_loss = self.criterion(real_output, real_labels)
            
            # Train with fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            fake_output = self.discriminator(fake_images.detach())
            fake_loss = self.criterion(fake_output, fake_labels)
            
            # Backward pass
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_D.step()
            
            # ========================
            # Train Generator
            # ========================
            self.optimizer_G.zero_grad()
            
            # Generate fake images and get discriminator output
            fake_output = self.discriminator(fake_images)
            g_loss = self.criterion(fake_output, real_labels)  # We want discriminator to think they're real
            
            # Backward pass
            g_loss.backward()
            self.optimizer_G.step()
            
            # Update losses
            epoch_G_loss += g_loss.item()
            epoch_D_loss += d_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'D(x)': f'{real_output.mean().item():.4f}',
                'D(G(z))': f'{fake_output.mean().item():.4f}'
            })
        
        # Calculate average losses
        avg_G_loss = epoch_G_loss / len(dataloader)
        avg_D_loss = epoch_D_loss / len(dataloader)
        
        self.G_losses.append(avg_G_loss)
        self.D_losses.append(avg_D_loss)
        
        return avg_G_loss, avg_D_loss
    
    def generate_samples(self, num_samples=64, save_path=None):
        """Generate sample images"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            
            # Convert to numpy for visualization
            fake_images = fake_images.cpu()
            
            # Create grid
            grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True, padding=2)
            
            if save_path:
                torchvision.utils.save_image(fake_images, save_path, nrow=8, normalize=True)
            
            return grid
    
    def plot_losses(self, save_path=None):
        """Plot training losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.G_losses, label='Generator Loss')
        plt.plot(self.D_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def interpolate_latent(self, steps=10, save_path=None):
        """Interpolate between two points in latent space"""
        self.generator.eval()
        
        with torch.no_grad():
            # Sample two random points
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            
            # Interpolate
            interpolated_images = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                fake_image = self.generator(z_interp)
                interpolated_images.append(fake_image)
            
            # Concatenate images
            interpolated_images = torch.cat(interpolated_images, dim=0)
            
            # Create grid
            grid = torchvision.utils.make_grid(interpolated_images, nrow=steps, normalize=True)
            
            if save_path:
                torchvision.utils.save_image(interpolated_images, save_path, nrow=steps, normalize=True)
            
            return grid

def get_dataloader(dataset_name='mnist', batch_size=64, img_size=28):
    """Get dataloader for specified dataset"""
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

def train_gan(dataset='mnist', epochs=100, batch_size=64, img_size=28, latent_dim=100, 
              model_type='basic', save_dir='results'):
    """Train GAN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataset info
    img_channels = 1 if dataset == 'mnist' else 3
    
    # Create models
    if model_type == 'basic':
        generator = Generator(latent_dim, img_channels, img_size).to(device)
        discriminator = Discriminator(img_channels, img_size).to(device)
    else:  # improved
        generator = ImprovedGenerator(latent_dim, img_channels, img_size).to(device)
        discriminator = ImprovedDiscriminator(img_channels, img_size).to(device)
    
    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Get dataloader
    dataloader = get_dataloader(dataset, batch_size, img_size)
    
    # Create trainer
    trainer = GANTrainer(generator, discriminator, device, latent_dim)
    
    print(f"Starting training on {dataset} dataset...")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    best_g_loss = float('inf')
    
    for epoch in range(epochs):
        g_loss, d_loss = trainer.train_epoch(dataloader, epoch + 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] - G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")
        
        # Save generated samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_path = os.path.join(save_dir, f'samples_epoch_{epoch+1}.png')
            trainer.generate_samples(save_path=sample_path)
        
        # Save best model
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_loss': g_loss,
                'd_loss': d_loss,
                'epoch': epoch + 1
            }, os.path.join(save_dir, 'best_gan.pt'))
    
    # Plot training curves
    trainer.plot_losses(os.path.join(save_dir, 'training_losses.png'))
    
    # Generate final samples
    trainer.generate_samples(save_path=os.path.join(save_dir, 'final_samples.png'))
    
    # Create interpolation
    trainer.interpolate_latent(save_path=os.path.join(save_dir, 'interpolation.png'))
    
    print("Training completed!")

def generate_samples(model_path, num_samples=100, save_dir='generated'):
    """Generate samples from trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate generator (you might need to adjust this based on your saved model)
    generator = Generator().to(device)  # Adjust parameters as needed
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc='Generating samples'):
            noise = torch.randn(1, 100, device=device)  # Adjust latent_dim as needed
            fake_image = generator(noise)
            
            # Save image
            save_path = os.path.join(save_dir, f'generated_{i:03d}.png')
            torchvision.utils.save_image(fake_image, save_path, normalize=True)
    
    print(f"Generated {num_samples} samples in {save_dir}/")

def evaluate_model(model_path, dataset='mnist'):
    """Evaluate GAN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate generator
    img_channels = 1 if dataset == 'mnist' else 3
    generator = Generator(img_channels=img_channels).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Generate samples for evaluation
    num_samples = 1000
    generated_images = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples // 64), desc='Generating for evaluation'):
            noise = torch.randn(64, 100, device=device)
            fake_images = generator(noise)
            generated_images.append(fake_images.cpu())
    
    generated_images = torch.cat(generated_images, dim=0)
    
    print(f"Generated {len(generated_images)} images for evaluation")
    print(f"Image shape: {generated_images.shape}")
    print(f"Value range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    
    # You can add more sophisticated evaluation metrics here (FID, IS, etc.)

def visualize_training(log_dir='logs'):
    """Visualize training progress"""
    # This would typically read from tensorboard logs or saved training data
    # For now, we'll create a simple visualization
    
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist")
        return
    
    # Look for saved loss data or images
    sample_files = [f for f in os.listdir(log_dir) if f.startswith('samples_epoch_')]
    
    if sample_files:
        print(f"Found {len(sample_files)} sample files")
        # You could create a grid showing training progression
    else:
        print("No sample files found for visualization")

def main():
    parser = argparse.ArgumentParser(description='Basic GAN Implementation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'generate', 'evaluate', 'visualize'],
                       help='Mode to run')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=28,
                       help='Image size')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension')
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'improved'],
                       help='Model architecture type')
    parser.add_argument('--model_path', type=str, default='results/best_gan.pt',
                       help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_gan(args.dataset, args.epochs, args.batch_size, args.img_size,
                 args.latent_dim, args.model_type, args.save_dir)
        
    elif args.mode == 'generate':
        generate_samples(args.model_path, args.num_samples, args.save_dir)
        
    elif args.mode == 'evaluate':
        evaluate_model(args.model_path, args.dataset)
        
    elif args.mode == 'visualize':
        visualize_training(args.log_dir)

if __name__ == "__main__":
    main()
