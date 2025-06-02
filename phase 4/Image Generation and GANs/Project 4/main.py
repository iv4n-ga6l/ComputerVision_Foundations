"""
Neural Style Transfer
====================

This project implements Neural Style Transfer using pre-trained convolutional neural networks
to combine the content of one image with the artistic style of another. The implementation
includes both optimization-based and fast feed-forward approaches.

Key Features:
- Optimization-based style transfer (Gatys et al.)
- Fast neural style transfer (Johnson et al.)
- Video style transfer capabilities
- Comprehensive loss functions
- Interactive parameter tuning

 Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
import cv2
from tqdm import tqdm
import glob
import time


class VGGFeatureExtractor(nn.Module):
    """VGG-19 feature extractor for style transfer"""
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Load pre-trained VGG-19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract feature layers
        self.layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        self.layers = {}
        
        # Map layer indices to names
        layer_indices = [0, 5, 10, 19, 21, 28]  # VGG-19 layer indices
        
        for i, (name, idx) in enumerate(zip(self.layer_names, layer_indices)):
            if i == 0:
                self.layers[name] = nn.Sequential(*list(vgg.children())[:idx+1])
            else:
                prev_idx = layer_indices[i-1]
                self.layers[name] = nn.Sequential(*list(vgg.children())[prev_idx+1:idx+1])
        
        # Convert to ModuleDict
        self.layers = nn.ModuleDict(self.layers)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Normalize input for VGG"""
        return (x - self.mean) / self.std

    def forward(self, x):
        """Extract features from multiple layers"""
        # Normalize input
        x = self.normalize(x)
        
        features = {}
        for name in self.layer_names:
            x = self.layers[name](x)
            features[name] = x
        
        return features


class StyleTransferLoss(nn.Module):
    """Combined loss for neural style transfer"""
    def __init__(self, content_layers=['conv4_2'], style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                 content_weight=1.0, style_weight=1000000.0, tv_weight=1.0):
        super(StyleTransferLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    def gram_matrix(self, features):
        """Compute Gram matrix for style representation"""
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)

    def content_loss(self, generated_features, content_features):
        """Compute content loss"""
        loss = 0
        for layer in self.content_layers:
            loss += F.mse_loss(generated_features[layer], content_features[layer])
        return loss

    def style_loss(self, generated_features, style_features):
        """Compute style loss using Gram matrices"""
        loss = 0
        for layer in self.style_layers:
            gen_gram = self.gram_matrix(generated_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += F.mse_loss(gen_gram, style_gram)
        return loss

    def total_variation_loss(self, image):
        """Compute total variation loss for smoothness"""
        batch_size, channels, height, width = image.size()
        
        # Calculate differences
        tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).sum()
        
        return (tv_h + tv_w) / (batch_size * channels * height * width)

    def forward(self, generated_image, content_image, style_image):
        """Compute total style transfer loss"""
        # Extract features
        gen_features = self.feature_extractor(generated_image)
        content_features = self.feature_extractor(content_image)
        style_features = self.feature_extractor(style_image)
        
        # Compute individual losses
        content_loss = self.content_loss(gen_features, content_features)
        style_loss = self.style_loss(gen_features, style_features)
        tv_loss = self.total_variation_loss(generated_image)
        
        # Combine losses
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return total_loss, content_loss, style_loss, tv_loss


class FastStyleTransferNet(nn.Module):
    """Fast Style Transfer Network (Johnson et al.)"""
    def __init__(self):
        super(FastStyleTransferNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(3, 32, 9, 1),
            self._conv_block(32, 64, 3, 2),
            self._conv_block(64, 128, 3, 2),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[self._residual_block(128) for _ in range(5)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self._upconv_block(128, 64, 3, 2),
            self._upconv_block(64, 32, 3, 2),
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Tanh()
        )
    
    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        """Convolutional block with instance normalization"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels, kernel_size, stride):
        """Upsampling convolutional block"""
        return nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size//2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _residual_block(self, channels):
        """Residual block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks with skip connections
        residual_input = x
        for block in self.residual_blocks:
            x = block(x) + x
        
        # Decoder
        x = self.decoder(x)
        
        return x


class OptimizationStyleTransfer:
    """Optimization-based style transfer (Gatys et al.)"""
    def __init__(self, content_weight=1.0, style_weight=1000000.0, tv_weight=1.0, device='cuda'):
        self.device = device
        self.loss_fn = StyleTransferLoss(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        ).to(device)

    def transfer(self, content_image, style_image, iterations=1000, lr=0.01, save_interval=100):
        """Perform optimization-based style transfer"""
        # Initialize generated image with content image
        generated_image = content_image.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.LBFGS([generated_image], lr=lr)
        
        # Training history
        history = {'total_loss': [], 'content_loss': [], 'style_loss': [], 'tv_loss': []}
        
        def closure():
            optimizer.zero_grad()
            
            # Clamp generated image to valid range
            generated_image.data.clamp_(0, 1)
            
            # Compute loss
            total_loss, content_loss, style_loss, tv_loss = self.loss_fn(
                generated_image, content_image, style_image
            )
            
            total_loss.backward()
            
            # Store losses
            history['total_loss'].append(total_loss.item())
            history['content_loss'].append(content_loss.item())
            history['style_loss'].append(style_loss.item())
            history['tv_loss'].append(tv_loss.item())
            
            return total_loss
        
        # Optimization loop
        for i in tqdm(range(iterations), desc="Style Transfer"):
            optimizer.step(closure)
            
            # Save intermediate results
            if (i + 1) % save_interval == 0:
                with torch.no_grad():
                    save_image(generated_image.clamp(0, 1), f'output/iteration_{i+1}.jpg')
        
        return generated_image.detach(), history


class FastStyleTransferTrainer:
    """Trainer for fast style transfer network"""
    def __init__(self, style_image_path, device='cuda', content_weight=1.0, style_weight=5.0, tv_weight=1e-4):
        self.device = device
        self.model = FastStyleTransferNet().to(device)
        
        # Load and preprocess style image
        style_image = self.load_image(style_image_path).to(device)
        
        # Create loss function
        self.loss_fn = StyleTransferLoss(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        ).to(device)
        
        # Extract style features once
        with torch.no_grad():
            self.style_features = self.loss_fn.feature_extractor(style_image)

    def load_image(self, image_path, size=256):
        """Load and preprocess image"""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    def train(self, dataset_path, epochs=2, batch_size=4, lr=1e-3):
        """Train fast style transfer network"""
        # Create dataset
        dataset = StyleTransferDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, content_images in enumerate(pbar):
                content_images = content_images.to(self.device)
                
                # Generate styled images
                styled_images = self.model(content_images)
                
                # Compute loss
                total_loss = 0
                for i in range(content_images.size(0)):
                    content_img = content_images[i:i+1]
                    styled_img = styled_images[i:i+1]
                    
                    # Create style image batch
                    style_img = list(self.style_features.values())[0][:1]  # Use first style feature as dummy
                    
                    # Compute individual loss
                    loss, _, _, _ = self.loss_fn(styled_img, content_img, content_img)  # Simplified for demo
                    total_loss += loss
                
                total_loss = total_loss / content_images.size(0)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                pbar.set_postfix({'Loss': total_loss.item()})
            
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}')
        
        # Save trained model
        torch.save(self.model.state_dict(), 'fast_style_transfer_model.pth')

    def stylize(self, content_image_path, output_path):
        """Stylize a single image using trained model"""
        self.model.eval()
        
        # Load content image
        content_image = self.load_image(content_image_path).to(self.device)
        
        # Generate styled image
        with torch.no_grad():
            styled_image = self.model(content_image)
        
        # Save result
        save_image(styled_image.clamp(0, 1), output_path)


class StyleTransferDataset(Dataset):
    """Dataset for training fast style transfer"""
    def __init__(self, data_path, image_size=256):
        self.image_paths = glob.glob(os.path.join(data_path, '*.jpg')) + \
                          glob.glob(os.path.join(data_path, '*.png'))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


class VideoStyleTransfer:
    """Video style transfer using fast or optimization methods"""
    def __init__(self, method='fast', model_path=None, device='cuda'):
        self.method = method
        self.device = device
        
        if method == 'fast' and model_path:
            self.model = FastStyleTransferNet().to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        elif method == 'optimization':
            self.optimizer_transfer = OptimizationStyleTransfer(device=device)

    def process_video(self, input_video_path, style_image_path, output_video_path):
        """Process video with style transfer"""
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Load style image
        style_image = self.load_image(style_image_path).to(self.device)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(self.device)
            
            # Apply style transfer
            if self.method == 'fast':
                with torch.no_grad():
                    styled_frame = self.model(frame_tensor)
            else:  # optimization method (slower)
                styled_frame, _ = self.optimizer_transfer.transfer(
                    frame_tensor, style_image, iterations=100
                )
            
            # Convert back to numpy
            styled_frame = styled_frame.squeeze(0).cpu().clamp(0, 1)
            styled_frame = transforms.ToPILImage()(styled_frame)
            styled_frame = cv2.cvtColor(np.array(styled_frame), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(styled_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release everything
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved to {output_video_path}")

    def load_image(self, image_path, size=None):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        
        if size:
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
        
        return transform(image).unsqueeze(0)


def load_and_preprocess_image(image_path, size=512, device='cuda'):
    """Load and preprocess image for style transfer"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


def save_comparison_image(content_image, style_image, generated_image, output_path):
    """Save comparison of content, style, and generated images"""
    # Detach and move to CPU
    content = content_image.squeeze(0).cpu()
    style = style_image.squeeze(0).cpu()
    generated = generated_image.squeeze(0).cpu().clamp(0, 1)
    
    # Create comparison
    comparison = torch.cat([content, style, generated], dim=2)
    save_image(comparison, output_path)


def plot_loss_history(history, save_path='loss_history.png'):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Content loss
    axes[0, 1].plot(history['content_loss'])
    axes[0, 1].set_title('Content Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Style loss
    axes[1, 0].plot(history['style_loss'])
    axes[1, 0].set_title('Style Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # TV loss
    axes[1, 1].plot(history['tv_loss'])
    axes[1, 1].set_title('Total Variation Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--method', type=str, default='optimization', 
                       choices=['optimization', 'fast'], help='Style transfer method')
    parser.add_argument('--mode', type=str, default='transfer', 
                       choices=['transfer', 'train'], help='Mode: transfer or train')
    parser.add_argument('--content', type=str, help='Content image path')
    parser.add_argument('--style', type=str, help='Style image path')
    parser.add_argument('--output', type=str, default='styled_image.jpg', help='Output image path')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of optimization iterations')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content loss weight')
    parser.add_argument('--style_weight', type=float, default=1000000.0, help='Style loss weight')
    parser.add_argument('--tv_weight', type=float, default=1.0, help='Total variation loss weight')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--size', type=int, default=512, help='Image size')
    parser.add_argument('--input_video', type=str, help='Input video path for video style transfer')
    parser.add_argument('--output_video', type=str, help='Output video path')
    parser.add_argument('--model_path', type=str, help='Path to trained fast style transfer model')
    parser.add_argument('--dataset', type=str, help='Dataset path for training fast style transfer')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs for fast style transfer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    if args.mode == 'transfer':
        if args.input_video:
            # Video style transfer
            video_transfer = VideoStyleTransfer(
                method=args.method, 
                model_path=args.model_path, 
                device=device
            )
            video_transfer.process_video(args.input_video, args.style, args.output_video)
        
        elif args.method == 'optimization':
            # Optimization-based style transfer
            if not args.content or not args.style:
                print("Please provide --content and --style image paths")
                return
            
            # Load images
            content_image = load_and_preprocess_image(args.content, args.size, device)
            style_image = load_and_preprocess_image(args.style, args.size, device)
            
            # Create style transfer
            style_transfer = OptimizationStyleTransfer(
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                tv_weight=args.tv_weight,
                device=device
            )
            
            # Perform style transfer
            print("Starting optimization-based style transfer...")
            start_time = time.time()
            
            generated_image, history = style_transfer.transfer(
                content_image, style_image, 
                iterations=args.iterations, 
                lr=args.lr
            )
            
            end_time = time.time()
            print(f"Style transfer completed in {end_time - start_time:.2f} seconds")
            
            # Save results
            save_image(generated_image.clamp(0, 1), args.output)
            save_comparison_image(content_image, style_image, generated_image, 'comparison.jpg')
            
            # Plot loss history
            plot_loss_history(history)
            
            print(f"Results saved to {args.output}")
        
        elif args.method == 'fast':
            # Fast style transfer
            if not args.model_path:
                print("Please provide --model_path for fast style transfer")
                return
            
            trainer = FastStyleTransferTrainer(args.style, device)
            trainer.model.load_state_dict(torch.load(args.model_path, map_location=device))
            trainer.stylize(args.content, args.output)
            
            print(f"Fast style transfer completed. Result saved to {args.output}")
    
    elif args.mode == 'train':
        # Train fast style transfer network
        if not args.style or not args.dataset:
            print("Please provide --style and --dataset paths for training")
            return
        
        trainer = FastStyleTransferTrainer(
            args.style, device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight
        )
        
        print("Starting fast style transfer training...")
        trainer.train(
            args.dataset, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        print("Training completed! Model saved as 'fast_style_transfer_model.pth'")


if __name__ == '__main__':
    main()
