"""
U-Net Semantic Segmentation Implementation
========================================

A comprehensive implementation of U-Net for semantic segmentation with support
for multiple datasets, advanced loss functions, and evaluation metrics.

Features:
- Complete U-Net architecture with skip connections
- Multi-dataset support (Cityscapes, Pascal VOC, custom)
- Advanced loss functions (Dice, Focal, Combo)
- Comprehensive evaluation metrics
- Data augmentation for segmentation
- Model visualization and interpretation

Author: Computer Vision Foundations Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    
    def __init__(self, images_dir: str, masks_dir: str, img_size: int = 512, 
                 augment: bool = True, num_classes: int = 21):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Define augmentations
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.RandomRotate90(p=0.3),
                A.ElasticTransform(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_file = self.image_files[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        mask = mask.long()
        
        return image, mask

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    
    def __init__(self, n_channels=3, n_classes=21, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

class ResNetBackbone(nn.Module):
    """ResNet backbone for U-Net"""
    
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        if backbone == 'resnet34':
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.channels = [64, 256, 512, 1024, 2048]
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
    
    def forward(self, x):
        features = []
        x = self.backbone[0](x)  # Conv1
        x = self.backbone[1](x)  # BN1
        x = self.backbone[2](x)  # ReLU
        features.append(x)
        
        x = self.backbone[3](x)  # MaxPool
        x = self.backbone[4](x)  # Layer1
        features.append(x)
        
        x = self.backbone[5](x)  # Layer2
        features.append(x)
        
        x = self.backbone[6](x)  # Layer3
        features.append(x)
        
        x = self.backbone[7](x)  # Layer4
        features.append(x)
        
        return features

class UNetResNet(nn.Module):
    """U-Net with ResNet backbone"""
    
    def __init__(self, n_classes=21, backbone='resnet34', pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        
        self.backbone = ResNetBackbone(backbone, pretrained)
        channels = self.backbone.channels
        
        # Decoder
        self.up1 = Up(channels[4] + channels[3], 256)
        self.up2 = Up(256 + channels[2], 128)
        self.up3 = Up(128 + channels[1], 64)
        self.up4 = Up(64 + channels[0], 32)
        
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Decoder with skip connections
        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        
        # Upsample to original size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        return self.final(x)

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ComboLoss(nn.Module):
    """Combination of CrossEntropy and Dice loss"""
    
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.alpha * ce + self.beta * dice

class SegmentationMetrics:
    """Metrics for segmentation evaluation"""
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """Update confusion matrix"""
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Get predictions
        predictions = np.argmax(predictions, axis=1)
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignore index
        mask = targets != self.ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for t, p in zip(targets, predictions):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_iou(self):
        """Compute IoU for each class"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - intersection)
        
        # Avoid division by zero
        union = np.maximum(union, 1e-10)
        iou = intersection / union
        
        return iou
    
    def compute_dice(self):
        """Compute Dice coefficient for each class"""
        intersection = np.diag(self.confusion_matrix)
        dice_denominator = (self.confusion_matrix.sum(axis=1) + 
                           self.confusion_matrix.sum(axis=0))
        
        # Avoid division by zero
        dice_denominator = np.maximum(dice_denominator, 1e-10)
        dice = (2.0 * intersection) / dice_denominator
        
        return dice
    
    def compute_pixel_accuracy(self):
        """Compute pixel accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / max(total, 1e-10)
    
    def compute_mean_iou(self):
        """Compute mean IoU"""
        iou = self.compute_iou()
        return np.nanmean(iou)
    
    def compute_mean_dice(self):
        """Compute mean Dice"""
        dice = self.compute_dice()
        return np.nanmean(dice)

class SegmentationTrainer:
    """Training pipeline for segmentation"""
    
    def __init__(self, model, device, num_classes=21):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = ComboLoss()
        self.metrics = SegmentationMetrics(num_classes)
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics
            self.metrics.update(outputs.detach(), masks.detach())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'mIoU': f'{self.metrics.compute_mean_iou():.4f}'
            })
        
        if scheduler:
            scheduler.step()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Update metrics
                self.metrics.update(outputs, masks)
        
        avg_loss = total_loss / len(dataloader)
        miou = self.metrics.compute_mean_iou()
        dice = self.metrics.compute_mean_dice()
        pixel_acc = self.metrics.compute_pixel_accuracy()
        
        return avg_loss, miou, dice, pixel_acc

class SegmentationInference:
    """Inference pipeline for segmentation"""
    
    def __init__(self, model_path, device, num_classes=21):
        self.device = device
        self.num_classes = num_classes
        
        # Load model
        self.model = UNet(n_classes=num_classes)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Define transform
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Color palette for visualization
        self.colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0]  # Background is black
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor = transformed['image'].unsqueeze(0)
        
        return tensor, original_size
    
    def predict(self, image_path):
        """Predict segmentation mask"""
        with torch.no_grad():
            tensor, original_size = self.preprocess_image(image_path)
            tensor = tensor.to(self.device)
            
            # Inference
            outputs = self.model(tensor)
            predictions = F.softmax(outputs, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            
            # Convert to numpy and resize to original size
            mask = predictions[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
            
            return mask
    
    def visualize_prediction(self, image_path, mask, save_path=None, alpha=0.6):
        """Visualize segmentation result"""
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(self.num_classes):
            colored_mask[mask == class_id] = self.colors[class_id]
        
        # Blend image and mask
        result = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        if save_path:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, result_bgr)
        
        return result

def create_demo_dataset(data_dir='demo_data', num_samples=20):
    """Create demo dataset for testing"""
    os.makedirs(f'{data_dir}/images', exist_ok=True)
    os.makedirs(f'{data_dir}/masks', exist_ok=True)
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(f'{data_dir}/images/image_{i:03d}.jpg', image)
        
        # Create random mask
        mask = np.random.randint(0, 21, (512, 512), dtype=np.uint8)
        cv2.imwrite(f'{data_dir}/masks/image_{i:03d}.png', mask)
    
    print(f"Demo dataset created in {data_dir}/")

def train_model(data_dir, epochs=50, batch_size=4, learning_rate=0.001, model_type='unet'):
    """Train segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SegmentationDataset(
        images_dir=f'{data_dir}/images',
        masks_dir=f'{data_dir}/masks',
        augment=True
    )
    
    val_dataset = SegmentationDataset(
        images_dir=f'{data_dir}/images',
        masks_dir=f'{data_dir}/masks',
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    if model_type == 'unet':
        model = UNet(n_classes=21)
    elif model_type == 'unet_resnet':
        model = UNetResNet(n_classes=21, backbone='resnet34')
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Trainer
    trainer = SegmentationTrainer(model, device)
    
    print("Starting training...")
    best_miou = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler)
        
        # Validate
        val_loss, miou, dice, pixel_acc = trainer.validate(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, mIoU: {miou:.4f}, Dice: {dice:.4f}, Pixel Acc: {pixel_acc:.4f}")
        
        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_segmentation.pt')
            print(f"Saved best model with mIoU: {best_miou:.4f}")
    
    print("Training completed!")

def segment_images(input_path, output_dir='results', model_path='best_segmentation.pt'):
    """Segment images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inference pipeline
    inference = SegmentationInference(model_path, device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        image_files = [input_path]
    else:
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        # Predict mask
        mask = inference.predict(image_path)
        
        # Visualize and save
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f'segmented_{filename}')
        inference.visualize_prediction(image_path, mask, output_path)
        
        # Save mask
        mask_path = os.path.join(output_dir, f'mask_{filename}')
        cv2.imwrite(mask_path, mask)

def evaluate_model(data_dir, model_path='best_segmentation.pt'):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = SegmentationDataset(
        images_dir=f'{data_dir}/images',
        masks_dir=f'{data_dir}/masks',
        augment=False
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Load model
    model = UNet(n_classes=21).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate
    trainer = SegmentationTrainer(model, device)
    val_loss, miou, dice, pixel_acc = trainer.validate(dataloader)
    
    print(f"\nEvaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Mean Dice: {dice:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    
    # Per-class IoU
    iou_per_class = trainer.metrics.compute_iou()
    print(f"\nPer-class IoU:")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i}: {iou:.4f}")

def main():
    parser = argparse.ArgumentParser(description='U-Net Semantic Segmentation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['create_demo', 'train', 'segment', 'evaluate'],
                       help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='demo_data',
                       help='Data directory')
    parser.add_argument('--input', type=str, default='demo_data/images',
                       help='Input images directory or single image')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='best_segmentation.pt',
                       help='Model weights path')
    parser.add_argument('--model_type', type=str, default='unet',
                       choices=['unet', 'unet_resnet'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'create_demo':
        create_demo_dataset(args.data_dir)
        
    elif args.mode == 'train':
        train_model(args.data_dir, args.epochs, args.batch_size, 
                   args.learning_rate, args.model_type)
        
    elif args.mode == 'segment':
        segment_images(args.input, args.output, args.model)
        
    elif args.mode == 'evaluate':
        evaluate_model(args.data_dir, args.model)

if __name__ == "__main__":
    main()
