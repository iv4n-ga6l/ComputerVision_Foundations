"""
CNN Architectures Comparison
Comprehensive comparison of different CNN architectures for performance analysis.
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
import seaborn as sns
import argparse
import time
import psutil
import gc
from collections import defaultdict, OrderedDict
import pandas as pd
from thop import profile
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CNN Architecture Implementations
# ============================================================================

class LeNet5(nn.Module):
    """LeNet-5 Architecture"""
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    """AlexNet Architecture (adapted for CIFAR)"""
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
    """VGG Architecture"""
    def __init__(self, num_classes=10, depth=16):
        super(VGG, self).__init__()
        
        if depth == 11:
            config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif depth == 13:
            config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif depth == 16:
            config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif depth == 19:
            config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        else:
            raise ValueError(f"VGG depth {depth} not supported")
            
        self.features = self._make_layers(config)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    """ResNet Basic Block"""
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
    """ResNet Architecture"""
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

def ResNet50(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)  # Simplified for CIFAR

class DenseBlock(nn.Module):
    """DenseNet Dense Block"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def _make_layer(self, in_channels, growth_rate, bn_size, drop_rate):
        layers = []
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(bn_size * growth_rate))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class SimpleDenseNet(nn.Module):
    """Simplified DenseNet for CIFAR"""
    def __init__(self, num_classes=10, growth_rate=12, block_config=(6, 12, 24, 16)):
        super(SimpleDenseNet, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Dense blocks
        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=num_features,
                             growth_rate=growth_rate, bn_size=4, drop_rate=0.1)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                # Transition layer
                trans = nn.Sequential(
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name, num_classes=10):
    """Create model by name"""
    model_name = model_name.lower()
    
    if model_name == 'lenet':
        return LeNet5(num_classes)
    elif model_name == 'alexnet':
        return AlexNet(num_classes)
    elif model_name == 'vgg11':
        return VGG(num_classes, depth=11)
    elif model_name == 'vgg16':
        return VGG(num_classes, depth=16)
    elif model_name == 'vgg19':
        return VGG(num_classes, depth=19)
    elif model_name == 'resnet18':
        return ResNet18(num_classes)
    elif model_name == 'resnet34':
        return ResNet34(num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes)
    elif model_name == 'densenet':
        return SimpleDenseNet(num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

# ============================================================================
# Benchmarking Framework
# ============================================================================

class ModelBenchmark:
    def __init__(self, device):
        self.device = device
        self.results = defaultdict(dict)
    
    def profile_model(self, model, model_name, input_size=(1, 3, 32, 32)):
        """Profile model performance metrics"""
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        dummy_input = torch.randn(input_size).to(self.device)
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        except:
            flops = 0  # Fallback if profiling fails
        
        # Model size in MB
        model_size = total_params * 4 / 1024 / 1024  # 4 bytes per parameter
        
        # Inference speed test
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(100):
                _ = model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            inference_time = (end_time - start_time) / 100  # Average per inference
        
        self.results[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size,
            'flops': flops,
            'inference_time_ms': inference_time * 1000,
            'fps': 1.0 / inference_time if inference_time > 0 else 0
        }
        
        return self.results[model_name]
    
    def train_and_evaluate(self, model, model_name, train_loader, test_loader, epochs=10):
        """Train model and measure performance"""
        print(f"\nTraining {model_name}...")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Memory usage before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        train_times = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            start_time = time.time()
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            train_time = time.time() - start_time
            train_acc = 100. * correct / total
            
            # Evaluation
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            
            test_acc = 100. * test_correct / test_total
            
            train_times.append(train_time)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            scheduler.step()
            
            print(f'  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {train_time:.2f}s')
        
        # Memory usage after training
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
        else:
            memory_used = 0
        
        # Store training results
        self.results[model_name].update({
            'final_train_acc': train_accuracies[-1],
            'final_test_acc': test_accuracies[-1],
            'best_test_acc': max(test_accuracies),
            'avg_epoch_time': np.mean(train_times),
            'total_train_time': sum(train_times),
            'memory_usage_mb': memory_used / 1024 / 1024,
            'train_history': train_accuracies,
            'test_history': test_accuracies
        })
        
        return self.results[model_name]
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to report!")
            return
        
        print("\n" + "="*80)
        print("CNN ARCHITECTURES COMPARISON REPORT")
        print("="*80)
        
        # Create DataFrame for easy comparison
        df_data = []
        for model_name, metrics in self.results.items():
            df_data.append({
                'Model': model_name,
                'Parameters (M)': metrics.get('total_params', 0) / 1e6,
                'Model Size (MB)': metrics.get('model_size_mb', 0),
                'FLOPs (M)': metrics.get('flops', 0) / 1e6,
                'Best Test Acc (%)': metrics.get('best_test_acc', 0),
                'Final Test Acc (%)': metrics.get('final_test_acc', 0),
                'Inference Time (ms)': metrics.get('inference_time_ms', 0),
                'FPS': metrics.get('fps', 0),
                'Avg Epoch Time (s)': metrics.get('avg_epoch_time', 0),
                'Memory Usage (MB)': metrics.get('memory_usage_mb', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        print("\nPERFORMANCE SUMMARY:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS AND RECOMMENDATIONS")
        print("="*80)
        
        # Best in each category
        best_accuracy = df.loc[df['Best Test Acc (%)'].idxmax()]
        most_efficient = df.loc[df['Parameters (M)'].idxmin()]
        fastest_inference = df.loc[df['Inference Time (ms)'].idxmin()]
        
        print(f"\n🏆 BEST ACCURACY: {best_accuracy['Model']} ({best_accuracy['Best Test Acc (%)']:.2f}%)")
        print(f"🚀 MOST EFFICIENT: {most_efficient['Model']} ({most_efficient['Parameters (M)']:.2f}M params)")
        print(f"⚡ FASTEST INFERENCE: {fastest_inference['Model']} ({fastest_inference['Inference Time (ms)']:.2f}ms)")
        
        # Recommendations
        print("\n📊 RECOMMENDATIONS:")
        print("- For high accuracy: Use ResNet or DenseNet architectures")
        print("- For mobile/edge devices: Use smaller models like LeNet or VGG11")
        print("- For real-time applications: Consider inference speed vs accuracy trade-offs")
        print("- For limited memory: Choose models with fewer parameters")

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_comparison_charts(benchmark_results, save_dir='./comparison_plots'):
    """Create comprehensive comparison visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    models = list(benchmark_results.keys())
    metrics = {}
    
    for metric in ['total_params', 'model_size_mb', 'best_test_acc', 'inference_time_ms', 'memory_usage_mb']:
        metrics[metric] = [benchmark_results[model].get(metric, 0) for model in models]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CNN Architectures Comparison', fontsize=16, fontweight='bold')
    
    # 1. Parameters comparison
    axes[0, 0].bar(models, np.array(metrics['total_params']) / 1e6)
    axes[0, 0].set_title('Model Parameters (Millions)')
    axes[0, 0].set_ylabel('Parameters (M)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Model size comparison
    axes[0, 1].bar(models, metrics['model_size_mb'])
    axes[0, 1].set_title('Model Size (MB)')
    axes[0, 1].set_ylabel('Size (MB)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy comparison
    axes[0, 2].bar(models, metrics['best_test_acc'])
    axes[0, 2].set_title('Best Test Accuracy (%)')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Inference time comparison
    axes[1, 0].bar(models, metrics['inference_time_ms'])
    axes[1, 0].set_title('Inference Time (ms)')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Memory usage comparison
    axes[1, 1].bar(models, metrics['memory_usage_mb'])
    axes[1, 1].set_title('Memory Usage (MB)')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Accuracy vs Parameters scatter plot
    axes[1, 2].scatter(np.array(metrics['total_params']) / 1e6, metrics['best_test_acc'], s=100)
    for i, model in enumerate(models):
        axes[1, 2].annotate(model, (metrics['total_params'][i] / 1e6, metrics['best_test_acc'][i]))
    axes[1, 2].set_title('Accuracy vs Parameters')
    axes[1, 2].set_xlabel('Parameters (M)')
    axes[1, 2].set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training curves
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, results) in enumerate(benchmark_results.items()):
        if 'test_history' in results:
            epochs = range(1, len(results['test_history']) + 1)
            plt.subplot(2, 3, i + 1)
            plt.plot(epochs, results['train_history'], 'b-', label='Train', linewidth=2)
            plt.plot(epochs, results['test_history'], 'r-', label='Test', linewidth=2)
            plt.title(f'{model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CNN Architectures Comparison')
    parser.add_argument('--models', nargs='+', 
                       choices=['lenet', 'alexnet', 'vgg11', 'vgg16', 'vgg19', 
                               'resnet18', 'resnet34', 'resnet50', 'densenet'],
                       help='Models to compare')
    parser.add_argument('--compare-all', action='store_true', 
                       help='Compare all available models')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--profile-only', action='store_true',
                       help='Only profile models without training')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model selection
    if args.compare_all:
        models_to_compare = ['lenet', 'alexnet', 'vgg11', 'vgg16', 'resnet18', 'resnet34', 'densenet']
    elif args.models:
        models_to_compare = args.models
    else:
        models_to_compare = ['lenet', 'alexnet', 'resnet18']  # Default comparison
    
    print(f"Comparing models: {models_to_compare}")
    
    # Load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(device)
    
    # Compare models
    for model_name in models_to_compare:
        try:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING: {model_name.upper()}")
            print('='*60)
            
            # Create model
            model = create_model(model_name, num_classes)
            
            # Profile model
            profile_results = benchmark.profile_model(model, model_name)
            print(f"\nProfile Results for {model_name}:")
            print(f"  Parameters: {profile_results['total_params']:,}")
            print(f"  Model Size: {profile_results['model_size_mb']:.2f} MB")
            print(f"  FLOPs: {profile_results['flops']:,}")
            print(f"  Inference Time: {profile_results['inference_time_ms']:.2f} ms")
            print(f"  FPS: {profile_results['fps']:.1f}")
            
            # Train and evaluate (if not profile-only)
            if not args.profile_only:
                train_results = benchmark.train_and_evaluate(
                    model, model_name, train_loader, test_loader, args.epochs
                )
                print(f"\nTraining Results for {model_name}:")
                print(f"  Best Test Accuracy: {train_results['best_test_acc']:.2f}%")
                print(f"  Final Test Accuracy: {train_results['final_test_acc']:.2f}%")
                print(f"  Total Training Time: {train_results['total_train_time']:.2f}s")
                print(f"  Memory Usage: {train_results['memory_usage_mb']:.2f} MB")
            
            # Clean up memory
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            continue
    
    # Generate comprehensive report
    benchmark.generate_report()
    
    # Create visualizations
    if not args.profile_only:
        plot_comparison_charts(benchmark.results)
    
    # Save results
    if args.save_results:
        import json
        with open(f'cnn_comparison_{args.dataset}.json', 'w') as f:
            json.dump(benchmark.results, f, indent=2)
        print(f"\nResults saved to cnn_comparison_{args.dataset}.json")
    
    print("\n🎉 CNN architecture comparison completed successfully!")

if __name__ == "__main__":
    main()
