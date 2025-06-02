"""
Transfer Learning with Pre-trained CNNs
Implementing transfer learning strategies for efficient training on custom datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from PIL import Image
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# Transfer Learning Models
# ============================================================================

class TransferLearningModel(nn.Module):
    """Base class for transfer learning models"""
    def __init__(self, model_name, num_classes, pretrained=True, freeze_features=True):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Freeze feature extraction layers if specified
        if freeze_features and pretrained:
            self.freeze_features()
    
    def freeze_features(self):
        """Freeze feature extraction layers"""
        if self.model_name.startswith('resnet') or self.model_name.startswith('densenet'):
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            if hasattr(self.backbone, 'fc'):
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, 'classifier'):
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
                    
        elif self.model_name == 'efficientnet_b0':
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
                
        elif self.model_name == 'vgg16':
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_top_layers(self, num_layers=2):
        """Unfreeze top N layers for fine-tuning"""
        if self.model_name.startswith('resnet'):
            layers = [self.backbone.layer4, self.backbone.layer3][:num_layers]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif self.model_name.startswith('densenet'):
            # Unfreeze final dense blocks
            if num_layers >= 1 and hasattr(self.backbone.features, 'denseblock4'):
                for param in self.backbone.features.denseblock4.parameters():
                    param.requires_grad = True
            if num_layers >= 2 and hasattr(self.backbone.features, 'transition3'):
                for param in self.backbone.features.transition3.parameters():
                    param.requires_grad = True
                    
        elif self.model_name == 'vgg16':
            # Unfreeze final convolutional layers
            layers_to_unfreeze = list(self.backbone.features.children())[-num_layers*2:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# Custom Dataset Handler
# ============================================================================

class CustomDataset(Dataset):
    """Custom dataset for transfer learning"""
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        if class_to_idx is None:
            self.class_to_idx = {}
            class_idx = 0
        else:
            self.class_to_idx = class_to_idx
            class_idx = max(class_to_idx.values()) + 1
        
        # Scan directory structure
        for class_name in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = class_idx
                class_idx += 1
            
            self.classes.append(class_name)
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(set(self.classes))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# Data Augmentation and Transforms
# ============================================================================

def get_transfer_transforms(input_size=224, train=True):
    """Get transforms for transfer learning"""
    if train:
        transform = transforms.Compose([
            transforms.Resize((int(input_size * 1.15), int(input_size * 1.15))),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

# ============================================================================
# Transfer Learning Trainer
# ============================================================================

class TransferLearningTrainer:
    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.history = defaultdict(list)
        self.class_names = None
    
    def set_class_names(self, class_names):
        """Set class names for visualization"""
        self.class_names = class_names
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
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
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader, criterion, return_predictions=False):
        """Evaluate model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        if return_predictions:
            return test_loss, test_acc, all_predictions, all_targets
        return test_loss, test_acc
    
    def train_feature_extraction(self, train_loader, test_loader, epochs, learning_rate=0.001):
        """Train in feature extraction mode"""
        print("Training in Feature Extraction Mode...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        return self._train_loop(train_loader, test_loader, criterion, optimizer, scheduler, epochs)
    
    def train_fine_tuning(self, train_loader, test_loader, epochs, learning_rate=0.0001, 
                         unfreeze_after=5):
        """Train in fine-tuning mode"""
        print("Training in Fine-tuning Mode...")
        criterion = nn.CrossEntropyLoss()
        
        # Start with frozen features
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=learning_rate * 10)  # Higher LR for classifier
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(epochs):
            # Unfreeze top layers after specified epochs
            if epoch == unfreeze_after:
                print(f"Unfreezing top layers at epoch {epoch}")
                self.model.unfreeze_top_layers(2)
                # Lower learning rate for fine-tuning
                optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Evaluation
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}, Time: {epoch_time:.2f}s')
            print('-' * 60)
        
        return self.history
    
    def _train_loop(self, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
        """Generic training loop"""
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Evaluation
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}, Time: {epoch_time:.2f}s')
            print('-' * 60)
        
        return self.history

# ============================================================================
# Visualization and Analysis
# ============================================================================

def plot_transfer_learning_results(history, save_path='transfer_learning_results.png'):
    """Plot transfer learning results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Transfer Learning - Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Transfer Learning - Model Accuracy', fontsize=14, fontweight='bold')
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
    
    # Convergence speed analysis
    improvement = []
    for i in range(1, len(history['test_acc'])):
        improvement.append(history['test_acc'][i] - history['test_acc'][0])
    
    ax4.plot(range(2, len(epochs) + 1), improvement, 'purple', linewidth=2)
    ax4.set_title('Accuracy Improvement Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Improvement (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_transfer_learning_performance(history, model_name, mode):
    """Analyze transfer learning performance"""
    print("=" * 60)
    print(f"TRANSFER LEARNING ANALYSIS - {model_name.upper()} ({mode.upper()})")
    print("=" * 60)
    
    # Performance metrics
    best_test_acc = max(history['test_acc'])
    best_epoch = history['test_acc'].index(best_test_acc) + 1
    final_test_acc = history['test_acc'][-1]
    initial_test_acc = history['test_acc'][0]
    
    print(f"Initial Test Accuracy: {initial_test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Total Improvement: {final_test_acc - initial_test_acc:.2f}%")
    
    # Convergence analysis
    if len(history['test_acc']) >= 5:
        epochs_to_90_percent = None
        target_acc = initial_test_acc + 0.9 * (best_test_acc - initial_test_acc)
        
        for i, acc in enumerate(history['test_acc']):
            if acc >= target_acc:
                epochs_to_90_percent = i + 1
                break
        
        if epochs_to_90_percent:
            print(f"Epochs to 90% of best performance: {epochs_to_90_percent}")
        
        # Training efficiency
        efficiency = best_test_acc / best_epoch
        print(f"Training efficiency: {efficiency:.2f}% per epoch")
    
    # Mode-specific analysis
    if mode == 'extract':
        print("\n✅ Feature Extraction Benefits:")
        print("  - Fast training with frozen features")
        print("  - Good for small datasets")
        print("  - Prevents overfitting on limited data")
    elif mode == 'finetune':
        print("\n✅ Fine-tuning Benefits:")
        print("  - Higher potential accuracy")
        print("  - Adapts features to new domain")
        print("  - Better for larger datasets")

def compare_models():
    """Compare different transfer learning models"""
    print("=" * 60)
    print("TRANSFER LEARNING MODEL COMPARISON")
    print("=" * 60)
    
    models_info = {
        'ResNet-50': {'params': 25.6, 'accuracy': 92.3, 'speed': 'Fast'},
        'ResNet-101': {'params': 44.5, 'accuracy': 93.1, 'speed': 'Medium'},
        'DenseNet-121': {'params': 8.0, 'accuracy': 91.8, 'speed': 'Fast'},
        'EfficientNet-B0': {'params': 5.3, 'accuracy': 93.8, 'speed': 'Fast'},
        'VGG-16': {'params': 138.0, 'accuracy': 89.5, 'speed': 'Slow'}
    }
    
    print(f"{'Model':<15} {'Params (M)':<12} {'Accuracy (%)':<15} {'Speed':<10}")
    print("-" * 60)
    
    for model, info in models_info.items():
        print(f"{model:<15} {info['params']:<12} {info['accuracy']:<15} {info['speed']:<10}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning for Image Classification')
    parser.add_argument('--mode', type=str, default='extract',
                       choices=['extract', 'finetune', 'compare'],
                       help='Transfer learning mode')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0', 'vgg16'],
                       help='Pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'custom'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./custom_data',
                       help='Directory for custom dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--unfreeze-after', type=int, default=5,
                       help='Epoch to start fine-tuning (finetune mode only)')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'compare':
        compare_models()
        return
    
    # Load dataset
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            train_transform = get_transfer_transforms(224, train=True)
            test_transform = get_transfer_transforms(224, train=False)
            
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                   download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                  download=True, transform=test_transform)
            num_classes = 10
            class_names = trainset.classes
        else:
            train_transform = get_transfer_transforms(224, train=True)
            test_transform = get_transfer_transforms(224, train=False)
            
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                                    download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                                   download=True, transform=test_transform)
            num_classes = 100
            class_names = trainset.classes
    else:
        # Custom dataset
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} not found!")
            print("Please create a directory with the following structure:")
            print("custom_data/")
            print("  ├── class1/")
            print("  ├── class2/")
            print("  └── ...")
            return
        
        train_transform = get_transfer_transforms(224, train=True)
        test_transform = get_transfer_transforms(224, train=False)
        
        # For simplicity, use same data for train/test (in practice, split properly)
        trainset = CustomDataset(args.data_dir, transform=train_transform)
        testset = CustomDataset(args.data_dir, transform=test_transform, 
                               class_to_idx=trainset.class_to_idx)
        num_classes = len(trainset.classes)
        class_names = trainset.classes
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Create model
    freeze_features = (args.mode == 'extract')
    model = TransferLearningModel(args.model, num_classes, pretrained=True, 
                                 freeze_features=freeze_features)
    
    print(f"\nModel: {args.model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Training
    trainer = TransferLearningTrainer(model, device, num_classes)
    trainer.set_class_names(class_names)
    
    if args.mode == 'extract':
        history = trainer.train_feature_extraction(train_loader, test_loader, 
                                                  args.epochs, args.lr)
    else:  # finetune
        history = trainer.train_fine_tuning(train_loader, test_loader, args.epochs, 
                                           args.lr, args.unfreeze_after)
    
    # Final evaluation with detailed metrics
    test_loss, test_acc, predictions, targets = trainer.evaluate(test_loader, 
                                                                nn.CrossEntropyLoss(), 
                                                                return_predictions=True)
    
    # Analysis and visualization
    plot_transfer_learning_results(history, f'{args.model}_{args.dataset}_{args.mode}_history.png')
    plot_confusion_matrix(targets, predictions, class_names, 
                         f'{args.model}_{args.dataset}_{args.mode}_confusion.png')
    analyze_transfer_learning_performance(history, args.model, args.mode)
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Save model
    if args.save_model:
        model_path = f'{args.model}_{args.dataset}_{args.mode}_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    print("\n🎉 Transfer learning completed successfully!")

if __name__ == "__main__":
    main()
