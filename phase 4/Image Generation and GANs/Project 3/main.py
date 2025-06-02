"""
Conditional GAN for Controlled Image Generation
==============================================

This project implements a Conditional Generative Adversarial Network (cGAN) that can generate
images based on specific class labels or conditions. The implementation includes multiple
conditioning strategies and comprehensive evaluation metrics.

Key Features:
- Multiple conditioning strategies (embedding, projection, auxiliary classifier)
- Support for MNIST, CIFAR-10, and custom datasets
- Spectral normalization for training stability
- Self-attention mechanisms
- Comprehensive evaluation metrics

 Project
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
import argparse
import os
from tqdm import tqdm
import random
from sklearn.metrics import classification_report
import seaborn as sns
from PIL import Image


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


class SpectralNorm(nn.Module):
    """Spectral Normalization for improved training stability"""
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def spectral_norm(module, name='weight', power_iterations=1):
    """Apply spectral normalization to a module"""
    SpectralNorm(module, name, power_iterations)
    return module


class SelfAttention(nn.Module):
    """Self-Attention mechanism for GANs"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Calculate attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class ConditionalGenerator(nn.Module):
    """Conditional Generator with multiple conditioning strategies"""
    def __init__(self, latent_dim=100, num_classes=10, img_channels=3, img_size=32, 
                 conditioning='embedding', embed_dim=100):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.conditioning = conditioning
        self.embed_dim = embed_dim
        
        # Label embedding
        if conditioning in ['embedding', 'projection']:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Calculate input dimension based on conditioning strategy
        if conditioning == 'embedding':
            input_dim = latent_dim + embed_dim
        elif conditioning == 'projection':
            input_dim = latent_dim
        else:  # concatenation
            input_dim = latent_dim + num_classes
        
        # Initial projection
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 4 * 4 * 512)),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU(True)
        )
        
        # Convolutional layers
        self.conv_blocks = nn.ModuleList([
            # 4x4 -> 8x8
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            ),
            # 8x8 -> 16x16
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
        ])
        
        # Self-attention at 16x16 resolution
        self.attention = SelfAttention(128)
        
        # Final output layer
        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, img_channels, 3, 1, 1)),
            nn.Tanh()
        )
        
        # Projection layer for projection-based conditioning
        if conditioning == 'projection':
            self.projection = nn.Linear(embed_dim, 512)

    def forward(self, noise, labels):
        batch_size = noise.size(0)
        
        if self.conditioning == 'embedding':
            # Embed labels and concatenate with noise
            label_embed = self.label_embedding(labels)
            input_tensor = torch.cat([noise, label_embed], dim=1)
        elif self.conditioning == 'projection':
            # Use projection-based conditioning
            label_embed = self.label_embedding(labels)
            input_tensor = noise
        else:  # one-hot concatenation
            label_onehot = F.one_hot(labels, self.num_classes).float()
            input_tensor = torch.cat([noise, label_onehot], dim=1)
        
        # Initial projection
        x = self.fc(input_tensor)
        x = x.view(batch_size, 512, 4, 4)
        
        # Apply projection conditioning if needed
        if self.conditioning == 'projection':
            proj = self.projection(label_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + proj
        
        # Convolutional blocks
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            # Apply self-attention at 16x16 resolution
            if i == 1:  # After second conv block (16x16)
                x = self.attention(x)
        
        # Final output
        x = self.final_conv(x)
        return x


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator with auxiliary classifier option"""
    def __init__(self, num_classes=10, img_channels=3, img_size=32, 
                 conditioning='embedding', embed_dim=100, auxiliary_classifier=False):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.conditioning = conditioning
        self.auxiliary_classifier = auxiliary_classifier
        
        # Label embedding
        if conditioning in ['embedding', 'projection']:
            self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Convolutional layers
        self.conv_blocks = nn.ModuleList([
            # 32x32 -> 16x16
            nn.Sequential(
                spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 8x8 -> 4x4
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # 4x4 -> 2x2
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        
        # Self-attention at 8x8 resolution
        self.attention = SelfAttention(256)
        
        # Calculate final feature size
        final_size = 2 * 2 * 512
        
        # Add label embedding size for embedding conditioning
        if conditioning == 'embedding':
            final_size += embed_dim
        
        # Discriminator head
        self.discriminator_head = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(final_size, 1))
        )
        
        # Auxiliary classifier head
        if auxiliary_classifier:
            self.classifier_head = nn.Sequential(
                nn.Flatten(),
                spectral_norm(nn.Linear(2 * 2 * 512, num_classes))
            )
        
        # Projection layer for projection-based conditioning
        if conditioning == 'projection':
            self.projection = nn.Linear(embed_dim, 512)

    def forward(self, images, labels):
        batch_size = images.size(0)
        
        # Process images through conv blocks
        x = images
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            # Apply self-attention at 8x8 resolution
            if i == 2:  # After third conv block (8x8)
                x = self.attention(x)
        
        # Store features for auxiliary classifier
        features = x
        
        # Apply conditioning
        if self.conditioning == 'embedding':
            # Embed labels and concatenate
            label_embed = self.label_embedding(labels)
            x_flat = x.view(batch_size, -1)
            x = torch.cat([x_flat, label_embed], dim=1)
            
            # Discriminator output
            disc_output = self.discriminator_head(x)
        elif self.conditioning == 'projection':
            # Use projection-based conditioning
            label_embed = self.label_embedding(labels)
            proj = self.projection(label_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + proj
            
            # Discriminator output
            disc_output = self.discriminator_head(x)
        else:  # No conditioning in discriminator
            disc_output = self.discriminator_head(x)
        
        # Auxiliary classifier output
        if self.auxiliary_classifier:
            class_output = self.classifier_head(features)
            return disc_output, class_output
        
        return disc_output


class ConditionalGANTrainer:
    """Trainer for Conditional GAN"""
    def __init__(self, generator, discriminator, device, lr_g=0.0002, lr_d=0.0002, 
                 beta1=0.5, beta2=0.999, auxiliary_classifier=False):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.auxiliary_classifier = auxiliary_classifier
        
        # Optimizers
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        if auxiliary_classifier:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'g_loss': [], 'd_loss': [], 'real_acc': [], 'fake_acc': []
        }
        if auxiliary_classifier:
            self.history['class_acc'] = []

    def train_step(self, real_images, real_labels, latent_dim):
        batch_size = real_images.size(0)
        
        # Create labels for real and fake samples
        real_validity = torch.ones(batch_size, 1, device=self.device)
        fake_validity = torch.zeros(batch_size, 1, device=self.device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()
        
        # Real images
        if self.auxiliary_classifier:
            real_pred, real_class_pred = self.discriminator(real_images, real_labels)
            d_real_loss = self.adversarial_loss(real_pred, real_validity)
            d_real_class_loss = self.classification_loss(real_class_pred, real_labels)
            d_real_total = d_real_loss + d_real_class_loss
        else:
            real_pred = self.discriminator(real_images, real_labels)
            d_real_total = self.adversarial_loss(real_pred, real_validity)
        
        # Generate fake images
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        fake_labels = torch.randint(0, self.generator.num_classes, (batch_size,), device=self.device)
        fake_images = self.generator(noise, fake_labels)
        
        # Fake images
        if self.auxiliary_classifier:
            fake_pred, fake_class_pred = self.discriminator(fake_images.detach(), fake_labels)
            d_fake_loss = self.adversarial_loss(fake_pred, fake_validity)
            d_fake_class_loss = self.classification_loss(fake_class_pred, fake_labels)
            d_fake_total = d_fake_loss + d_fake_class_loss
        else:
            fake_pred = self.discriminator(fake_images.detach(), fake_labels)
            d_fake_total = self.adversarial_loss(fake_pred, fake_validity)
        
        # Total discriminator loss
        d_loss = (d_real_total + d_fake_total) / 2
        d_loss.backward()
        self.optimizer_d.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_g.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        fake_labels = torch.randint(0, self.generator.num_classes, (batch_size,), device=self.device)
        fake_images = self.generator(noise, fake_labels)
        
        # Generator loss
        if self.auxiliary_classifier:
            fake_pred, fake_class_pred = self.discriminator(fake_images, fake_labels)
            g_adv_loss = self.adversarial_loss(fake_pred, real_validity)
            g_class_loss = self.classification_loss(fake_class_pred, fake_labels)
            g_loss = g_adv_loss + g_class_loss
        else:
            fake_pred = self.discriminator(fake_images, fake_labels)
            g_loss = self.adversarial_loss(fake_pred, real_validity)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        # Calculate accuracies
        real_acc = (torch.sigmoid(real_pred) > 0.5).float().mean().item()
        fake_acc = (torch.sigmoid(fake_pred) < 0.5).float().mean().item()
        
        # Store metrics
        metrics = {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'real_acc': real_acc,
            'fake_acc': fake_acc
        }
        
        if self.auxiliary_classifier:
            class_acc = (fake_class_pred.argmax(1) == fake_labels).float().mean().item()
            metrics['class_acc'] = class_acc
        
        return metrics

    def train(self, dataloader, epochs, latent_dim, save_interval=10, checkpoint_dir='checkpoints'):
        """Train the conditional GAN"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs('samples', exist_ok=True)
        
        for epoch in range(epochs):
            epoch_metrics = {'g_loss': 0, 'd_loss': 0, 'real_acc': 0, 'fake_acc': 0}
            if self.auxiliary_classifier:
                epoch_metrics['class_acc'] = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (real_images, real_labels) in enumerate(pbar):
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)
                
                # Train step
                metrics = self.train_step(real_images, real_labels, latent_dim)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                
                # Update progress bar
                pbar.set_postfix(
                    G_loss=f"{metrics['g_loss']:.4f}",
                    D_loss=f"{metrics['d_loss']:.4f}",
                    Real_acc=f"{metrics['real_acc']:.2f}",
                    Fake_acc=f"{metrics['fake_acc']:.2f}"
                )
            
            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)
                self.history[key].append(epoch_metrics[key])
            
            # Save sample images
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch + 1, latent_dim)
                self.save_checkpoint(epoch + 1, checkpoint_dir)
        
        # Save final model
        self.save_checkpoint('final', checkpoint_dir)

    def save_samples(self, epoch, latent_dim, num_samples=100):
        """Save sample images for each class"""
        self.generator.eval()
        with torch.no_grad():
            # Generate samples for each class
            samples_per_class = num_samples // self.generator.num_classes
            all_samples = []
            
            for class_idx in range(self.generator.num_classes):
                noise = torch.randn(samples_per_class, latent_dim, device=self.device)
                labels = torch.full((samples_per_class,), class_idx, device=self.device)
                samples = self.generator(noise, labels)
                all_samples.append(samples)
            
            # Concatenate all samples
            all_samples = torch.cat(all_samples, dim=0)
            
            # Save grid
            grid = make_grid(all_samples, nrow=samples_per_class, normalize=True, value_range=(-1, 1))
            save_image(grid, f'samples/epoch_{epoch}.png')
        
        self.generator.train()

    def save_checkpoint(self, epoch, checkpoint_dir):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


def get_dataset(dataset_name, img_size=32, batch_size=64):
    """Get dataset and dataloader"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        num_classes = 10
        img_channels = 1
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        num_classes = 10
        img_channels = 3
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader, num_classes, img_channels


def generate_samples(generator, num_classes, latent_dim, device, num_samples_per_class=10):
    """Generate samples for each class"""
    generator.eval()
    all_samples = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            noise = torch.randn(num_samples_per_class, latent_dim, device=device)
            labels = torch.full((num_samples_per_class,), class_idx, device=device)
            samples = generator(noise, labels)
            all_samples.append(samples)
    
    return torch.cat(all_samples, dim=0)


def interpolate_between_classes(generator, class1, class2, latent_dim, device, steps=10):
    """Generate interpolation between two classes"""
    generator.eval()
    
    with torch.no_grad():
        # Fix noise
        noise = torch.randn(1, latent_dim, device=device)
        
        # Get embeddings for both classes
        labels1 = torch.tensor([class1], device=device)
        labels2 = torch.tensor([class2], device=device)
        
        if hasattr(generator, 'label_embedding'):
            embed1 = generator.label_embedding(labels1)
            embed2 = generator.label_embedding(labels2)
            
            interpolated_samples = []
            for i in range(steps):
                alpha = i / (steps - 1)
                interpolated_embed = (1 - alpha) * embed1 + alpha * embed2
                
                # Manually create input
                if generator.conditioning == 'embedding':
                    input_tensor = torch.cat([noise, interpolated_embed], dim=1)
                    x = generator.fc(input_tensor)
                    x = x.view(1, 512, 4, 4)
                    
                    for j, conv_block in enumerate(generator.conv_blocks):
                        x = conv_block(x)
                        if j == 1:
                            x = generator.attention(x)
                    
                    x = generator.final_conv(x)
                    interpolated_samples.append(x)
            
            return torch.cat(interpolated_samples, dim=0)
    
    return None


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Generator and Discriminator loss
    axes[0, 0].plot(history['g_loss'], label='Generator Loss')
    axes[0, 0].plot(history['d_loss'], label='Discriminator Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracies
    axes[0, 1].plot(history['real_acc'], label='Real Accuracy')
    axes[0, 1].plot(history['fake_acc'], label='Fake Accuracy')
    axes[0, 1].set_title('Discriminator Accuracies')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Classification accuracy (if available)
    if 'class_acc' in history:
        axes[1, 0].plot(history['class_acc'], label='Classification Accuracy')
        axes[1, 0].set_title('Auxiliary Classifier Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Loss ratio
    g_loss = np.array(history['g_loss'])
    d_loss = np.array(history['d_loss'])
    loss_ratio = g_loss / (d_loss + 1e-8)
    axes[1, 1].plot(loss_ratio, label='G_loss / D_loss')
    axes[1, 1].set_title('Loss Ratio')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Conditional GAN for Controlled Image Generation')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    parser.add_argument('--conditioning', type=str, default='embedding', 
                       choices=['embedding', 'projection', 'concatenation'],
                       help='Conditioning strategy')
    parser.add_argument('--auxiliary_classifier', action='store_true',
                       help='Use auxiliary classifier GAN')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'evaluate'],
                       help='Mode: train, generate, or evaluate')
    parser.add_argument('--model_path', type=str, help='Path to saved model for generation/evaluation')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--classes', type=str, help='Comma-separated list of classes to generate')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval for samples')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(42)
    
    if args.mode == 'train':
        # Get dataset
        dataloader, num_classes, img_channels = get_dataset(
            args.dataset, args.img_size, args.batch_size
        )
        
        # Create models
        generator = ConditionalGenerator(
            latent_dim=args.latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=args.img_size,
            conditioning=args.conditioning
        )
        
        discriminator = ConditionalDiscriminator(
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=args.img_size,
            conditioning=args.conditioning,
            auxiliary_classifier=args.auxiliary_classifier
        )
        
        # Create trainer
        trainer = ConditionalGANTrainer(
            generator, discriminator, device,
            lr_g=args.lr_g, lr_d=args.lr_d,
            auxiliary_classifier=args.auxiliary_classifier
        )
        
        # Train model
        print("Starting training...")
        trainer.train(dataloader, args.epochs, args.latent_dim, args.save_interval)
        
        # Plot training history
        plot_training_history(trainer.history)
        
        print("Training completed!")
    
    elif args.mode == 'generate':
        if not args.model_path:
            print("Please provide --model_path for generation")
            return
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Get dataset info
        _, num_classes, img_channels = get_dataset(args.dataset, args.img_size, 1)
        
        # Create generator
        generator = ConditionalGenerator(
            latent_dim=args.latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=args.img_size,
            conditioning=args.conditioning
        ).to(device)
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Generate samples
        if args.classes:
            class_list = [int(c) for c in args.classes.split(',')]
            samples_per_class = args.num_samples // len(class_list)
            
            all_samples = []
            for class_idx in class_list:
                noise = torch.randn(samples_per_class, args.latent_dim, device=device)
                labels = torch.full((samples_per_class,), class_idx, device=device)
                samples = generator(noise, labels)
                all_samples.append(samples)
            
            all_samples = torch.cat(all_samples, dim=0)
        else:
            all_samples = generate_samples(generator, num_classes, args.latent_dim, device)
        
        # Save generated images
        grid = make_grid(all_samples, nrow=10, normalize=True, value_range=(-1, 1))
        save_image(grid, 'generated_samples.png')
        print(f"Generated {len(all_samples)} samples saved to 'generated_samples.png'")
        
        # Generate interpolation between classes if applicable
        if num_classes >= 2:
            interpolation = interpolate_between_classes(generator, 0, 1, args.latent_dim, device)
            if interpolation is not None:
                grid = make_grid(interpolation, nrow=10, normalize=True, value_range=(-1, 1))
                save_image(grid, 'class_interpolation.png')
                print("Class interpolation saved to 'class_interpolation.png'")
    
    elif args.mode == 'evaluate':
        print("Evaluation mode - implement your specific evaluation metrics here")
        # Implement evaluation metrics like Inception Score, FID, etc.


if __name__ == '__main__':
    main()
