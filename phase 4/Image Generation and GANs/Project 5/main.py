"""
Variational Autoencoders (VAE)
=============================

This project implements Variational Autoencoders for learning latent representations and
generating new images. The implementation includes standard VAE, β-VAE for disentanglement,
and conditional VAE variants with comprehensive evaluation tools.

Key Features:
- Standard VAE with proper reparameterization trick
- β-VAE for disentangled representation learning
- Conditional VAE for class-conditional generation
- Latent space interpolation and visualization
- Comprehensive evaluation metrics

Author: Computer Vision Foundations Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VAE(nn.Module):
    """Standard Variational Autoencoder"""
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through VAE"""
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        
        # Encode
        mu, logvar = self.encode(x_flat)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Reshape back to original dimensions
        x_recon = x_recon.view_as(x)
        
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder"""
    def __init__(self, input_channels=3, latent_dim=128, img_size=32):
        super(ConvVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4x4 -> 2x2
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Calculate flattened size
        self.encoder_output_size = 256 * 2 * 2
        
        # Latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """Decode latent representation"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        """Generate samples from prior"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples


class BetaVAE(ConvVAE):
    """β-VAE for disentangled representation learning"""
    def __init__(self, input_channels=3, latent_dim=128, img_size=32, beta=4.0):
        super(BetaVAE, self).__init__(input_channels, latent_dim, img_size)
        self.beta = beta

    def loss_function(self, x_recon, x, mu, logvar):
        """β-VAE loss with weighted KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder"""
    def __init__(self, input_channels=3, latent_dim=128, num_classes=10, img_size=32):
        super(ConditionalVAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Encoder (similar to ConvVAE but with label conditioning)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        encoder_output_size = 256 * 2 * 2
        
        # Latent space with label conditioning
        self.fc_mu = nn.Linear(encoder_output_size + 50, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size + 50, latent_dim)
        
        # Decoder with label conditioning
        self.fc_decode = nn.Linear(latent_dim + 50, encoder_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        """Encode with label conditioning"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Add label information
        label_embed = self.label_embedding(labels)
        h = torch.cat([h, label_embed], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, labels):
        """Decode with label conditioning"""
        label_embed = self.label_embedding(labels)
        z = torch.cat([z, label_embed], dim=1)
        
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.decoder(h)

    def forward(self, x, labels):
        """Forward pass with labels"""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        return x_recon, mu, logvar

    def sample(self, num_samples, labels, device):
        """Generate conditional samples"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, labels)
        return samples


class VAETrainer:
    """Trainer for VAE models"""
    def __init__(self, model, device, learning_rate=1e-3, beta=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        
        # Training history
        self.history = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': []
        }

    def vae_loss(self, x_recon, x, mu, logvar):
        """Standard VAE loss function"""
        # Reconstruction loss (binary cross-entropy or MSE)
        if torch.max(x) <= 1.0:  # Normalized images
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        else:
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def train_step(self, data, labels=None):
        """Single training step"""
        data = data.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if isinstance(self.model, ConditionalVAE):
            x_recon, mu, logvar = self.model(data, labels)
        else:
            x_recon, mu, logvar = self.model(data)
        
        # Compute loss
        if hasattr(self.model, 'loss_function'):
            # Use model's custom loss function (e.g., β-VAE)
            total_loss, recon_loss, kl_loss = self.model.loss_function(x_recon, data, mu, logvar)
        else:
            total_loss, recon_loss, kl_loss = self.vae_loss(x_recon, data, mu, logvar)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item() / data.size(0),
            'recon_loss': recon_loss.item() / data.size(0),
            'kl_loss': kl_loss.item() / data.size(0)
        }

    def train(self, dataloader, epochs, save_interval=10):
        """Train the VAE"""
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('samples', exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, batch in enumerate(pbar):
                if len(batch) == 2:  # (data, labels)
                    data, labels = batch
                else:  # just data
                    data, labels = batch, None
                
                # Training step
                metrics = self.train_step(data, labels)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                
                # Update progress bar
                pbar.set_postfix(
                    Total=f"{metrics['total_loss']:.4f}",
                    Recon=f"{metrics['recon_loss']:.4f}",
                    KL=f"{metrics['kl_loss']:.4f}"
                )
            
            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)
                self.history[key].append(epoch_metrics[key])
            
            # Save samples and checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch + 1)
                self.save_checkpoint(epoch + 1)
        
        # Save final model
        self.save_checkpoint('final')

    def save_samples(self, epoch, num_samples=64):
        """Save generated samples"""
        self.model.eval()
        with torch.no_grad():
            # Generate samples from prior
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples
            if len(samples.shape) == 4:  # Image data
                grid = make_grid(samples, nrow=8, normalize=True)
                save_image(grid, f'samples/epoch_{epoch}_samples.png')
            else:  # Flattened data (e.g., MNIST)
                samples = samples.view(num_samples, 1, 28, 28)  # Assume MNIST
                grid = make_grid(samples, nrow=8, normalize=True)
                save_image(grid, f'samples/epoch_{epoch}_samples.png')
        
        self.model.train()

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, f'checkpoints/vae_epoch_{epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


def get_dataset(dataset_name, batch_size=64, img_size=32):
    """Get dataset and dataloader"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        input_channels = 1
        num_classes = 10
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        input_channels = 3
        num_classes = 10
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader, input_channels, num_classes


def interpolate_latent_space(model, img1_path, img2_path, device, steps=10):
    """Interpolate between two images in latent space"""
    model.eval()
    
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    img1 = transform(Image.open(img1_path)).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode images to latent space
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        
        # Interpolate in latent space
        interpolated_images = []
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode interpolated latent
            img_interp = model.decode(z_interp)
            interpolated_images.append(img_interp)
        
        # Concatenate all interpolated images
        interpolation = torch.cat(interpolated_images, dim=0)
        
        # Save interpolation
        grid = make_grid(interpolation, nrow=steps, normalize=True)
        save_image(grid, 'latent_interpolation.png')
    
    model.train()
    return interpolation


def visualize_latent_space(model, dataloader, device, method='tsne', num_samples=1000):
    """Visualize latent space using t-SNE or PCA"""
    model.eval()
    
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            if len(batch) == 2:
                data, batch_labels = batch
            else:
                data, batch_labels = batch, None
            
            data = data.to(device)
            
            # Encode to latent space
            if isinstance(model, ConditionalVAE):
                mu, _ = model.encode(data, batch_labels.to(device))
            else:
                mu, _ = model.encode(data)
            
            latent_codes.append(mu.cpu().numpy())
            if batch_labels is not None:
                labels.append(batch_labels.numpy())
            
            sample_count += data.size(0)
    
    # Concatenate all latent codes
    latent_codes = np.concatenate(latent_codes, axis=0)[:num_samples]
    if labels:
        labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Dimensionality reduction
    if method == 'tsne':
        embeddings = TSNE(n_components=2, random_state=42).fit_transform(latent_codes)
    elif method == 'pca':
        embeddings = PCA(n_components=2, random_state=42).fit_transform(latent_codes)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)
    
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f'latent_space_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    model.train()


def generate_conditional_samples(model, device, num_classes=10, samples_per_class=10):
    """Generate samples for each class (Conditional VAE)"""
    if not isinstance(model, ConditionalVAE):
        print("Model must be ConditionalVAE for conditional generation")
        return
    
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            labels = torch.full((samples_per_class,), class_idx, device=device)
            samples = model.sample(samples_per_class, labels, device)
            all_samples.append(samples)
    
    # Concatenate and save
    all_samples = torch.cat(all_samples, dim=0)
    grid = make_grid(all_samples, nrow=samples_per_class, normalize=True)
    save_image(grid, 'conditional_samples.png')
    
    model.train()
    return all_samples


def plot_training_history(history, save_path='vae_training_history.png'):
    """Plot VAE training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Reconstruction loss
    axes[1].plot(history['recon_loss'])
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # KL divergence
    axes[2].plot(history['kl_loss'])
    axes[2].set_title('KL Divergence')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Variational Autoencoders')
    parser.add_argument('--model', type=str, default='vae', 
                       choices=['vae', 'conv_vae', 'beta_vae', 'cvae'], 
                       help='Model type')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'], help='Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=4.0, help='Beta value for β-VAE')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'generate', 'interpolate', 'evaluate'], 
                       help='Mode')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--image1', type=str, help='First image for interpolation')
    parser.add_argument('--image2', type=str, help='Second image for interpolation')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(42)
    
    if args.mode == 'train':
        # Get dataset
        dataloader, input_channels, num_classes = get_dataset(
            args.dataset, args.batch_size
        )
        
        # Create model
        if args.model == 'vae':
            if args.dataset == 'mnist':
                model = VAE(input_dim=784, latent_dim=args.latent_dim)
            else:
                model = ConvVAE(input_channels=input_channels, latent_dim=args.latent_dim)
        elif args.model == 'conv_vae':
            model = ConvVAE(input_channels=input_channels, latent_dim=args.latent_dim)
        elif args.model == 'beta_vae':
            model = BetaVAE(input_channels=input_channels, latent_dim=args.latent_dim, beta=args.beta)
        elif args.model == 'cvae':
            model = ConditionalVAE(input_channels=input_channels, latent_dim=args.latent_dim, num_classes=num_classes)
        
        # Create trainer
        trainer = VAETrainer(model, device, learning_rate=args.lr, beta=args.beta)
        
        # Train model
        print(f"Training {args.model} on {args.dataset}...")
        trainer.train(dataloader, args.epochs, args.save_interval)
        
        # Plot training history
        plot_training_history(trainer.history)
        
        # Visualize latent space
        visualize_latent_space(model, dataloader, device)
        
        print("Training completed!")
    
    elif args.mode == 'generate':
        if not args.model_path:
            print("Please provide --model_path for generation")
            return
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Create model (you might need to adjust based on saved model type)
        if args.model == 'cvae':
            model = ConditionalVAE(latent_dim=args.latent_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            generate_conditional_samples(model, device)
        else:
            if args.dataset == 'mnist':
                model = VAE(latent_dim=args.latent_dim)
            else:
                model = ConvVAE(latent_dim=args.latent_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Generate samples
            samples = model.sample(args.num_samples, device)
            if len(samples.shape) == 2:  # Flattened
                samples = samples.view(args.num_samples, 1, 28, 28)
            
            grid = make_grid(samples, nrow=8, normalize=True)
            save_image(grid, 'generated_samples.png')
        
        print(f"Generated {args.num_samples} samples saved to 'generated_samples.png'")
    
    elif args.mode == 'interpolate':
        if not args.model_path or not args.image1 or not args.image2:
            print("Please provide --model_path, --image1, and --image2 for interpolation")
            return
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        model = ConvVAE(latent_dim=args.latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Perform interpolation
        interpolation = interpolate_latent_space(model, args.image1, args.image2, device)
        print("Latent space interpolation saved to 'latent_interpolation.png'")
    
    elif args.mode == 'evaluate':
        print("Evaluation mode - implement specific evaluation metrics here")
        # Implement reconstruction quality, latent space analysis, etc.


if __name__ == '__main__':
    main()
