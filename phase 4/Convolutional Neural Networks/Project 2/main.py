"""
Image Classification with CNNs
Implementing modern CNN architectures for large-scale image classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from collections import defaultdict

# ============================================================================
# CNN Architectures
# ============================================================================

class LeNet(nn.Module):
    """Classic LeNet-5 architecture"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    """Simplified AlexNet for CIFAR datasets"""
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    """VGG-like architecture for CIFAR"""
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet architecture"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# ============================================================================
# Data Loading and Augmentation
# ============================================================================

def get_transforms(dataset_name, train=True):
    """Get appropriate transforms for different datasets"""
    if dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar100':
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:  # ImageNet-like transforms
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    return transform

def get_dataset(dataset_name, root='./data'):
    """Load dataset"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        train_transform = get_transforms('cifar10', train=True)
        test_transform = get_transforms('cifar10', train=False)
        
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        num_classes = 10
        
    elif dataset_name == 'cifar100':
        train_transform = get_transforms('cifar100', train=True)
        test_transform = get_transforms('cifar100', train=False)
        
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
        num_classes = 100
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return trainset, testset, num_classes

# ============================================================================
# Training and Evaluation
# ============================================================================

class CNNTrainer:
    def __init__(self, model, device, num_classes):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.history = defaultdict(list)
    
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
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, test_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc
    
    def train(self, train_loader, test_loader, epochs, learning_rate=0.001, weight_decay=5e-4):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        print(f"Training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Evaluation
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            # Learning rate scheduling
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

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['test_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    # Best accuracy progress
    best_acc = []
    current_best = 0
    for acc in history['test_acc']:
        current_best = max(current_best, acc)
        best_acc.append(current_best)
    
    ax4.plot(epochs, best_acc, 'purple', label='Best Validation Accuracy')
    ax4.set_title('Best Accuracy Progress')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_model_performance(history):
    """Analyze model performance"""
    print("=" * 60)
    print("TRAINING ANALYSIS")
    print("=" * 60)
    
    # Best performance
    best_test_acc = max(history['test_acc'])
    best_epoch = history['test_acc'].index(best_test_acc) + 1
    final_test_acc = history['test_acc'][-1]
    
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Training stability
    final_10_epochs = history['test_acc'][-10:]
    std_final_10 = np.std(final_10_epochs)
    print(f"Standard deviation (last 10 epochs): {std_final_10:.2f}%")
    
    # Overfitting analysis
    train_test_gap = history['train_acc'][-1] - history['test_acc'][-1]
    print(f"Train-Test accuracy gap: {train_test_gap:.2f}%")
    
    if train_test_gap > 10:
        print("⚠️  High overfitting detected")
    elif train_test_gap > 5:
        print("⚠️  Moderate overfitting detected")
    else:
        print("✅ Good generalization")
    
    # Convergence analysis
    if len(history['test_acc']) >= 10:
        recent_improvement = history['test_acc'][-1] - history['test_acc'][-10]
        if abs(recent_improvement) < 0.5:
            print("✅ Training converged")
        else:
            print("⚠️  Training may need more epochs")

def compare_architectures():
    """Compare different CNN architectures"""
    print("=" * 60)
    print("CNN ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    architectures = {
        'LeNet': LeNet(),
        'AlexNet': AlexNet(),
        'VGG': VGG(),
        'ResNet18': ResNet18()
    }
    
    for name, model in architectures.items():
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name}:")
        print(f"  Total parameters: {params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {params * 4 / 1024 / 1024:.2f} MB")
        print()

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CNN Image Classification')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['lenet', 'alexnet', 'vgg', 'resnet18', 'resnet34'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    parser.add_argument('--compare', action='store_true', help='Compare architectures')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.compare:
        compare_architectures()
        return
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    trainset, testset, num_classes = get_dataset(args.dataset)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Create model
    model_name = args.model.lower()
    if model_name == 'lenet':
        model = LeNet(num_classes)
    elif model_name == 'alexnet':
        model = AlexNet(num_classes)
    elif model_name == 'vgg':
        model = VGG(num_classes)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes)
    
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    trainer = CNNTrainer(model, device, num_classes)
    history = trainer.train(train_loader, test_loader, args.epochs, args.lr, args.weight_decay)
    
    # Analysis and visualization
    plot_training_history(history, f'{args.model}_{args.dataset}_history.png')
    analyze_model_performance(history)
    
    # Save model
    if args.save_model:
        model_path = f'{args.model}_{args.dataset}_best.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
