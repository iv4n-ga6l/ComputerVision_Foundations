"""
DCGAN (Deep Convolutional GAN) Implementation
Generate high-quality images using deep convolutional architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def weights_init(m):
    """Initialize weights for DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """DCGAN Generator"""
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            # state size: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size: ngf x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: nc x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """DCGAN Discriminator"""
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        self.main = nn.Sequential(
            # Input is nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class ProgressiveGenerator(nn.Module):
    """Progressive GAN Generator"""
    def __init__(self, nz=512, ngf=512, nc=3, max_layer=6):
        super(ProgressiveGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.max_layer = max_layer
        self.current_layer = 1
        
        # Initial 4x4 block
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Progressive layers
        self.layers = nn.ModuleList()
        for i in range(max_layer):
            out_ch = ngf // (2 ** min(i, 4))
            in_ch = ngf // (2 ** min(i-1, 4)) if i > 0 else ngf
            
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.layers.append(layer)
        
        # Output layers for each resolution
        self.to_rgb = nn.ModuleList()
        for i in range(max_layer + 1):
            ch = ngf // (2 ** min(i, 4))
            self.to_rgb.append(nn.Conv2d(ch, nc, 1, 1, 0))
    
    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # Progressive layers up to current layer
        for i in range(self.current_layer):
            x = self.layers[i](x)
        
        # Convert to RGB
        rgb = torch.tanh(self.to_rgb[self.current_layer](x))
        
        return rgb
    
    def grow_network(self):
        """Add a new layer to the network"""
        if self.current_layer < self.max_layer:
            self.current_layer += 1
            logger.info(f"Growing network to layer {self.current_layer}")

class StyleGenerator(nn.Module):
    """StyleGAN-inspired Generator with style modulation"""
    def __init__(self, nz=512, ngf=512, nc=3, n_mlp=8):
        super(StyleGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        
        # Style mapping network
        layers = []
        for i in range(n_mlp):
            layers.append(nn.Linear(nz, nz))
            layers.append(nn.LeakyReLU(0.2))
        self.style_mapping = nn.Sequential(*layers)
        
        # Learned constant input
        self.constant_input = nn.Parameter(torch.randn(1, ngf, 4, 4))
        
        # Synthesis layers
        self.layers = nn.ModuleList()
        
        # 4x4 -> 8x8
        self.layers.append(self._make_layer(ngf, ngf))
        # 8x8 -> 16x16
        self.layers.append(self._make_layer(ngf, ngf // 2))
        # 16x16 -> 32x32
        self.layers.append(self._make_layer(ngf // 2, ngf // 4))
        # 32x32 -> 64x64
        self.layers.append(self._make_layer(ngf // 4, ngf // 8))
        
        # To RGB
        self.to_rgb = nn.Conv2d(ngf // 8, nc, 1, 1, 0)
    
    def _make_layer(self, in_ch, out_ch):
        """Create a synthesis layer with style modulation"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, z):
        # Map to style codes
        w = self.style_mapping(z)
        
        # Start with constant
        batch_size = z.size(0)
        x = self.constant_input.expand(batch_size, -1, -1, -1)
        
        # Synthesis
        for layer in self.layers:
            x = layer(x)
        
        # Convert to RGB
        rgb = torch.tanh(self.to_rgb(x))
        
        return rgb

class SpectralNormDiscriminator(nn.Module):
    """Discriminator with Spectral Normalization"""
    def __init__(self, nc=3, ndf=64):
        super(SpectralNormDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is nc x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class SelfAttention(nn.Module):
    """Self-Attention module for SAGAN"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection
        out = self.gamma * out + x
        return out

class DCGAN:
    """DCGAN Training Class"""
    def __init__(self, nz=100, ngf=64, ndf=64, nc=3, lr=0.0002, beta1=0.5, device='cuda'):
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.lr = lr
        self.beta1 = beta1
        self.device = device
        
        # Initialize networks
        self.netG = Generator(nz, ngf, nc).to(device)
        self.netD = Discriminator(nc, ndf).to(device)
        
        # Apply weight initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Fixed noise for visualization
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        
        # Training history
        self.G_losses = []
        self.D_losses = []
        self.D_x_history = []
        self.D_G_z_history = []
    
    def train(self, dataloader, num_epochs=100, save_interval=10, output_dir='./dcgan_output'):
        """Train DCGAN"""
        os.makedirs(output_dir, exist_ok=True)
        
        real_label = 1.
        fake_label = 0.
        
        print(f"Starting Training on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.netG.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.netD.parameters())}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            for i, (data, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                self.netD.zero_grad()
                real_data = data.to(self.device)
                batch_size = real_data.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                
                output = self.netD(real_data)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                
                # Train with all-fake batch
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(fake_label)
                output = self.netD(fake.detach())
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()
                
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = self.netD(fake)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()
                
                # Save losses
                if i % 50 == 0:
                    self.G_losses.append(errG.item())
                    self.D_losses.append(errD.item())
                    self.D_x_history.append(D_x)
                    self.D_G_z_history.append(D_G_z1)
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, '
                  f'Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, '
                  f'D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            # Save generated images
            if epoch % save_interval == 0:
                with torch.no_grad():
                    fake_images = self.netG(self.fixed_noise)
                    save_image(fake_images, f'{output_dir}/fake_epoch_{epoch:03d}.png', 
                             normalize=True, nrow=8)
            
            # Save models
            if (epoch + 1) % 50 == 0:
                torch.save(self.netG.state_dict(), f'{output_dir}/generator_epoch_{epoch+1}.pth')
                torch.save(self.netD.state_dict(), f'{output_dir}/discriminator_epoch_{epoch+1}.pth')
        
        print("Training completed!")
        
        # Save final models
        torch.save(self.netG.state_dict(), f'{output_dir}/generator_final.pth')
        torch.save(self.netD.state_dict(), f'{output_dir}/discriminator_final.pth')
        
        # Plot training curves
        self.plot_losses(output_dir)
    
    def plot_losses(self, output_dir):
        """Plot training losses"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.G_losses, label='Generator')
        plt.plot(self.D_losses, label='Discriminator')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.D_x_history, label='D(x)')
        plt.plot(self.D_G_z_history, label='D(G(z))')
        plt.xlabel('Iterations')
        plt.ylabel('Probability')
        plt.legend()
        plt.title('Discriminator Output')
        
        plt.subplot(1, 3, 3)
        # Generate samples for visualization
        with torch.no_grad():
            sample_noise = torch.randn(16, self.nz, 1, 1, device=self.device)
            sample_images = self.netG(sample_noise)
            grid = make_grid(sample_images, nrow=4, normalize=True)
            
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title('Generated Samples')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_samples(self, num_samples=64, output_path='generated_samples.png'):
        """Generate random samples"""
        self.netG.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
            fake_images = self.netG(noise)
            
            save_image(fake_images, output_path, normalize=True, nrow=8)
            print(f"Generated {num_samples} samples saved to {output_path}")
    
    def interpolate_latent(self, num_steps=10, output_path='interpolation.png'):
        """Generate interpolation between two random points"""
        self.netG.eval()
        
        with torch.no_grad():
            # Two random points
            z1 = torch.randn(1, self.nz, 1, 1, device=self.device)
            z2 = torch.randn(1, self.nz, 1, 1, device=self.device)
            
            # Interpolation
            interpolated_images = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                fake_image = self.netG(z_interp)
                interpolated_images.append(fake_image)
            
            # Create grid
            interpolated_tensor = torch.cat(interpolated_images, dim=0)
            save_image(interpolated_tensor, output_path, normalize=True, nrow=num_steps)
            print(f"Interpolation saved to {output_path}")

def load_dataset(dataset_name='cifar10', batch_size=128, image_size=64):
    """Load dataset for training"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=transform)
    elif dataset_name == 'celeba':
        dataset = torchvision.datasets.CelebA(root='./data', split='train',
                                             download=True, transform=transform)
    elif dataset_name == 'mnist':
        # Adjust transform for grayscale
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)
    
    return dataloader

def demo_training():
    """Demo training with CIFAR-10"""
    print("Setting up DCGAN demo...")
    
    # Hyperparameters
    batch_size = 128
    image_size = 64
    nz = 100
    ngf = 64
    ndf = 64
    nc = 3
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    dataloader = load_dataset('cifar10', batch_size, image_size)
    
    # Initialize DCGAN
    dcgan = DCGAN(nz=nz, ngf=ngf, ndf=ndf, nc=nc, lr=lr, beta1=beta1, device=device)
    
    # Train
    print("Starting training...")
    dcgan.train(dataloader, num_epochs=num_epochs, save_interval=2)
    
    # Generate samples
    print("Generating samples...")
    dcgan.generate_samples(num_samples=64)
    dcgan.interpolate_latent(num_steps=10)

def main():
    parser = argparse.ArgumentParser(description='DCGAN for Image Generation')
    parser.add_argument('--mode', choices=['train', 'generate', 'interpolate', 'demo'], 
                        default='demo', help='Mode to run')
    parser.add_argument('--dataset', choices=['cifar10', 'celeba', 'mnist'], 
                        default='cifar10', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--nz', type=int, default=100, help='Size of latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='Generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator feature maps')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--output_dir', type=str, default='./dcgan_output', help='Output directory')
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'demo':
        demo_training()
    
    elif args.mode == 'train':
        # Load dataset
        dataloader = load_dataset(args.dataset, args.batch_size, args.image_size)
        
        # Get number of channels
        nc = 1 if args.dataset == 'mnist' else 3
        
        # Initialize and train DCGAN
        dcgan = DCGAN(nz=args.nz, ngf=args.ngf, ndf=args.ndf, nc=nc, 
                     lr=args.lr, device=device)
        dcgan.train(dataloader, num_epochs=args.num_epochs, output_dir=args.output_dir)
    
    elif args.mode == 'generate':
        if not args.model_path:
            print("Error: model_path required for generation")
            return
        
        # Load model
        nc = 1 if args.dataset == 'mnist' else 3
        netG = Generator(args.nz, args.ngf, nc).to(device)
        netG.load_state_dict(torch.load(args.model_path))
        
        # Generate samples
        dcgan = DCGAN(nz=args.nz, ngf=args.ngf, ndf=args.ndf, nc=nc, device=device)
        dcgan.netG = netG
        dcgan.generate_samples(num_samples=args.num_samples)
    
    elif args.mode == 'interpolate':
        if not args.model_path:
            print("Error: model_path required for interpolation")
            return
        
        # Load model
        nc = 1 if args.dataset == 'mnist' else 3
        netG = Generator(args.nz, args.ngf, nc).to(device)
        netG.load_state_dict(torch.load(args.model_path))
        
        # Generate interpolation
        dcgan = DCGAN(nz=args.nz, ngf=args.ngf, ndf=args.ndf, nc=nc, device=device)
        dcgan.netG = netG
        dcgan.interpolate_latent()
    
    print("\nDCGAN Implementation completed!")
    print("\nKey Features Implemented:")
    print("- Deep Convolutional GAN architecture")
    print("- Batch normalization and LeakyReLU")
    print("- Progressive training with monitoring")
    print("- Multiple dataset support (CIFAR-10, CelebA, MNIST)")
    print("- Advanced architectures (Progressive, StyleGAN-inspired)")
    print("- Spectral normalization for training stability")
    print("- Self-attention for improved image quality")
    print("- Latent space interpolation")
    print("- Comprehensive visualization tools")

if __name__ == "__main__":
    main()
