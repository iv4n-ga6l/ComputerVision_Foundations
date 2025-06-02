"""
Custom Dataset Classification
Complete pipeline for training CNN models on custom image datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import os
import json
from PIL import Image, ImageEnhance, ImageFilter
import random
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Custom Dataset Class
# ============================================================================

class CustomImageDataset(Dataset):
    """Custom dataset for image classification"""
    def __init__(self, data_dir, transform=None, class_to_idx=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan directory structure
        class_idx = 0
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            # Get all images in class directory
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.extend([f for f in os.listdir(class_path) 
                                  if f.lower().endswith(ext)])
            
            # Limit samples per class if specified
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                image_files = random.sample(image_files, max_samples_per_class)
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                self.samples.append((img_path, class_idx))
            
            class_idx += 1
        
        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get class distribution statistics"""
        class_counts = Counter([label for _, label in self.samples])
        return {self.idx_to_class[idx]: count for idx, count in class_counts.items()}

# ============================================================================
# Advanced Data Augmentation
# ============================================================================

class AdvancedAugmentation:
    """Advanced data augmentation techniques"""
    
    @staticmethod
    def cutout(img, n_holes=1, length=16):
        """Apply cutout augmentation"""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img
    
    @staticmethod
    def mixup_data(x, y, alpha=1.0):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Mixup loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_transforms(input_size=224, augmentation_level='basic'):
    """Get data transforms based on augmentation level"""
    
    if augmentation_level == 'basic':
        train_transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_level == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augmentation_level == 'advanced':
        train_transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.2), int(input_size * 1.2))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ============================================================================
# Model Architecture Factory
# ============================================================================

def create_model(model_name, num_classes, pretrained=True):
    """Create model by name"""
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

# ============================================================================
# Training Framework
# ============================================================================

class CustomDatasetTrainer:
    def __init__(self, model, device, num_classes, class_names):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        self.history = defaultdict(list)
        self.best_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader, criterion, optimizer, use_mixup=False):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup if specified
            if use_mixup and random.random() < 0.5:
                data, target_a, target_b, lam = AdvancedAugmentation.mixup_data(data, target)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = AdvancedAugmentation.mixup_criterion(criterion, output, target_a, target_b, lam)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += (lam * predicted.eq(target_a).sum().float() + 
                          (1 - lam) * predicted.eq(target_b).sum().float()).item()
            else:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader, criterion, return_predictions=False):
        """Evaluate model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        if return_predictions:
            return val_loss, val_acc, all_predictions, all_targets
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs, learning_rate=0.001, 
              weight_decay=1e-4, use_mixup=False, scheduler_type='step'):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.1, patience=10)
        
        print(f"Training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using mixup: {use_mixup}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, use_mixup)
            
            # Validation
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Learning rate scheduling
            if scheduler_type == 'reduce':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Best Val Acc: {self.best_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}, Time: {epoch_time:.2f}s')
            print('-' * 70)
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history

# ============================================================================
# Dataset Analysis and Visualization
# ============================================================================

def analyze_dataset(dataset, save_dir='./analysis'):
    """Analyze dataset characteristics"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Class distribution
    class_dist = dataset.get_class_distribution()
    print(f"Number of classes: {len(class_dist)}")
    print(f"Total samples: {len(dataset)}")
    print("\nClass distribution:")
    for class_name, count in class_dist.items():
        print(f"  {class_name}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    classes, counts = zip(*class_dist.items())
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=classes, autopct='%1.1f%%')
    plt.title('Class Distribution (Pie Chart)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sample images from each class
    fig, axes = plt.subplots(2, min(5, len(classes)), figsize=(15, 6))
    if len(classes) == 1:
        axes = [axes]
    elif len(classes) <= 5:
        axes = axes.reshape(-1)
    
    for idx, class_name in enumerate(classes[:min(5, len(classes))]):
        class_idx = dataset.class_to_idx[class_name]
        
        # Find samples from this class
        class_samples = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        
        for row in range(2):
            if idx < min(5, len(classes)):
                sample_idx = class_samples[row] if row < len(class_samples) else class_samples[0]
                img, _ = dataset[sample_idx]
                
                if isinstance(img, torch.Tensor):
                    # Denormalize if normalized
                    img = img.permute(1, 2, 0)
                    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                    img = torch.clamp(img, 0, 1)
                    img = img.numpy()
                
                axes[row * min(5, len(classes)) + idx].imshow(img)
                axes[row * min(5, len(classes)) + idx].set_title(f'{class_name}')
                axes[row * min(5, len(classes)) + idx].axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_results(history, save_path='training_results.png'):
    """Plot comprehensive training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Validation accuracy smoothed
    window_size = max(1, len(history['val_acc']) // 10)
    smoothed_val_acc = []
    for i in range(len(history['val_acc'])):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(history['val_acc']), i + window_size // 2 + 1)
        smoothed_val_acc.append(np.mean(history['val_acc'][start_idx:end_idx]))
    
    ax4.plot(epochs, history['val_acc'], 'lightcoral', alpha=0.6, label='Validation Accuracy')
    ax4.plot(epochs, smoothed_val_acc, 'r-', linewidth=2, label='Smoothed Validation Accuracy')
    ax4.set_title('Validation Accuracy (Smoothed)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Model Export and Deployment
# ============================================================================

def export_model(model, save_path, input_size=(1, 3, 224, 224), export_format='torch'):
    """Export model for deployment"""
    model.eval()
    
    if export_format == 'torch':
        # Save PyTorch model
        torch.save(model.state_dict(), f"{save_path}.pth")
        print(f"Model saved as {save_path}.pth")
    
    elif export_format == 'torchscript':
        # Export as TorchScript
        dummy_input = torch.randn(input_size)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(f"{save_path}.pt")
        print(f"Model exported as TorchScript: {save_path}.pt")
    
    elif export_format == 'onnx':
        # Export as ONNX
        dummy_input = torch.randn(input_size)
        torch.onnx.export(model, dummy_input, f"{save_path}.onnx",
                         export_params=True, opset_version=11,
                         do_constant_folding=True,
                         input_names=['input'], output_names=['output'])
        print(f"Model exported as ONNX: {save_path}.onnx")

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Custom Dataset Classification')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'resnet101', 'densenet121', 
                               'efficientnet', 'vgg16', 'mobilenet'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--augmentation', type=str, default='medium',
                       choices=['basic', 'medium', 'advanced'],
                       help='Data augmentation level')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--use-mixup', action='store_true',
                       help='Use mixup augmentation')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'reduce'],
                       help='Learning rate scheduler')
    parser.add_argument('--export-model', action='store_true',
                       help='Export trained model')
    parser.add_argument('--export-format', type=str, default='torch',
                       choices=['torch', 'torchscript', 'onnx'],
                       help='Export format')
    parser.add_argument('--model-path', type=str, default='best_model',
                       help='Path to save/load model')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset without training')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        print("Please organize your dataset as follows:")
        print("data_dir/")
        print("  ├── class1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── class2/")
        print("      ├── image1.jpg")
        print("      └── image2.jpg")
        return
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    train_transform, val_transform = get_transforms(224, args.augmentation)
    
    # Create full dataset first to get class information
    full_dataset = CustomImageDataset(args.data_dir, transform=None)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Analyze dataset
    analyze_dataset(full_dataset)
    
    if args.analyze_only:
        return
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(args.val_split * total_size)
    train_size = total_size - val_size
    
    # Create train and validation datasets with transforms
    train_dataset = CustomImageDataset(args.data_dir, transform=train_transform)
    val_dataset = CustomImageDataset(args.data_dir, transform=val_transform)
    
    # Split indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # Create model
    model = create_model(args.model, num_classes, pretrained=True)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    trainer = CustomDatasetTrainer(model, device, num_classes, class_names)
    history = trainer.train(train_loader, val_loader, args.epochs, 
                           args.lr, args.weight_decay, args.use_mixup, args.scheduler)
    
    # Final evaluation
    val_loss, val_acc, predictions, targets = trainer.evaluate(val_loader, 
                                                               nn.CrossEntropyLoss(),
                                                               return_predictions=True)
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {trainer.best_acc:.2f}%")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    
    # Visualizations
    plot_training_results(history, 'training_results.png')
    plot_confusion_matrix(targets, predictions, class_names, 'confusion_matrix.png')
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Export model
    if args.export_model:
        export_model(model, args.model_path, export_format=args.export_format)
        
        # Save class mapping
        class_mapping = {
            'class_to_idx': full_dataset.class_to_idx,
            'idx_to_class': full_dataset.idx_to_class,
            'classes': class_names
        }
        with open(f'{args.model_path}_classes.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"Class mapping saved to {args.model_path}_classes.json")
    
    print("\n🎉 Custom dataset classification completed successfully!")

if __name__ == "__main__":
    main()
