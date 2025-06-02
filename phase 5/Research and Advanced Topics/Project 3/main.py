"""
Neural Architecture Search (NAS) Implementation
==============================================

This module implements various Neural Architecture Search techniques to automatically
discover optimal neural network architectures for computer vision tasks.

Methods implemented:
- DARTS: Differentiable Architecture Search
- PNAS: Progressive Neural Architecture Search
- ENAS: Efficient Neural Architecture Search
- PC-DARTS: Partially Connected DARTS
- ProxylessNAS: Direct Neural Architecture Search

Author: Computer Vision Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict, defaultdict
import os
import random
import argparse
from tqdm import tqdm
import time
import json
import yaml
from scipy.special import softmax
import networkx as nx
import graphviz
from thop import profile
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class OperationSpace:
    """Define the operation space for NAS"""
    
    PRIMITIVES = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'conv_1x1',
        'conv_3x3',
    ]
    
    @staticmethod
    def get_operation(op_name, C, stride, affine=True):
        """Get operation by name"""
        if op_name == 'none':
            return Zero(stride)
        elif op_name == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif op_name == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride=stride, padding=1)
        elif op_name == 'skip_connect':
            if stride == 1:
                return Identity()
            else:
                return FactorizedReduce(C, C, affine=affine)
        elif op_name == 'sep_conv_3x3':
            return SepConv(C, C, 3, stride, 1, affine=affine)
        elif op_name == 'sep_conv_5x5':
            return SepConv(C, C, 5, stride, 2, affine=affine)
        elif op_name == 'dil_conv_3x3':
            return DilConv(C, C, 3, stride, 2, 2, affine=affine)
        elif op_name == 'dil_conv_5x5':
            return DilConv(C, C, 5, stride, 4, 2, affine=affine)
        elif op_name == 'conv_1x1':
            return ReLUConvBN(C, C, 1, stride, 0, affine=affine)
        elif op_name == 'conv_3x3':
            return ReLUConvBN(C, C, 3, stride, 1, affine=affine)
        else:
            raise ValueError(f"Unknown operation: {op_name}")

class ReLUConvBN(nn.Module):
    """ReLU + Conv + BatchNorm"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separable convolution"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated convolution"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    """Identity operation"""
    
    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation"""
    
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    """Factorized reduction"""
    
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class MixedOp(nn.Module):
    """Mixed operation for DARTS"""
    
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in OperationSpace.PRIMITIVES:
            op = OperationSpace.get_operation(primitive, C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        """Forward with architecture weights"""
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    """DARTS cell"""
    
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        
        self._steps = steps
        self._multiplier = multiplier
        
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        """Forward pass with architecture weights"""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) 
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self._multiplier:], dim=1)

class DARTSNetwork(nn.Module):
    """DARTS network for architecture search"""
    
    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # Initialize architecture parameters
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        """Initialize architecture parameters"""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(OperationSpace.PRIMITIVES)
        
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
    def arch_parameters(self):
        """Return architecture parameters"""
        return self._arch_parameters
    
    def forward(self, input):
        """Forward pass"""
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def genotype(self):
        """Generate discrete architecture from continuous weights"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), 
                             key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                                              if k != OperationSpace.PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != OperationSpace.PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((OperationSpace.PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = {'normal': gene_normal, 'normal_concat': concat,
                   'reduce': gene_reduce, 'reduce_concat': concat}
        return genotype

class DARTSTrainer:
    """DARTS trainer for architecture search"""
    
    def __init__(self, dataset='cifar10', batch_size=64, learning_rate=0.025,
                 learning_rate_min=0.001, momentum=0.9, weight_decay=3e-4,
                 epochs=50, grad_clip=5, train_portion=0.5):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.train_portion = train_portion
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        self._prepare_data()
        
        # Initialize model
        if dataset == 'cifar10':
            self.model = DARTSNetwork(C=16, num_classes=10, layers=8)
        else:
            self.model = DARTSNetwork(C=16, num_classes=1000, layers=14)
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizers
        self.optimizer = optim.SGD(
            self.model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay
        )
        self.arch_optimizer = optim.Adam(
            self.model.arch_parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, epochs, eta_min=learning_rate_min
        )
        
        # For logging
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
    
    def _prepare_data(self):
        """Prepare dataset"""
        if self.dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
            train_data = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            valid_data = CIFAR10(root='./data', train=True, download=False, transform=valid_transform)
            
            num_train = len(train_data)
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            
            train_sampler = SubsetRandomSampler(indices[:split])
            valid_sampler = SubsetRandomSampler(indices[split:])
            
            self.train_queue = DataLoader(
                train_data, batch_size=self.batch_size, sampler=train_sampler,
                pin_memory=True, num_workers=2
            )
            
            self.valid_queue = DataLoader(
                valid_data, batch_size=self.batch_size, sampler=valid_sampler,
                pin_memory=True, num_workers=2
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0
        
        # Training
        for step, (input, target) in enumerate(tqdm(self.train_queue, desc=f"Train Epoch {epoch}")):
            input = input.to(self.device)
            target = target.to(self.device)
            
            # Architecture step
            input_search, target_search = next(iter(self.valid_queue))
            input_search = input_search.to(self.device)
            target_search = target_search.to(self.device)
            
            self.arch_optimizer.zero_grad()
            logits = self.model(input_search)
            arch_loss = F.cross_entropy(logits, target_search)
            arch_loss.backward()
            self.arch_optimizer.step()
            
            # Weight step
            self.optimizer.zero_grad()
            logits = self.model(input)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            # Statistics
            prec1 = self._accuracy(logits, target)[0]
            train_loss += loss.item()
            train_acc += prec1.item()
        
        # Validation
        self.model.eval()
        with torch.no_grad():
            for input, target in self.valid_queue:
                input = input.to(self.device)
                target = target.to(self.device)
                
                logits = self.model(input)
                loss = F.cross_entropy(logits, target)
                
                prec1 = self._accuracy(logits, target)[0]
                valid_loss += loss.item()
                valid_acc += prec1.item()
        
        # Average metrics
        train_loss /= len(self.train_queue)
        train_acc /= len(self.train_queue)
        valid_loss /= len(self.valid_queue)
        valid_acc /= len(self.valid_queue)
        
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accuracies.append(valid_acc)
        
        return train_loss, train_acc, valid_loss, valid_acc
    
    def _accuracy(self, output, target, topk=(1,)):
        """Compute accuracy"""
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def search(self):
        """Main search loop"""
        print(f"Starting DARTS search on {self.device}")
        
        for epoch in range(self.epochs):
            # Update learning rate
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            
            # Train epoch
            train_loss, train_acc, valid_loss, valid_acc = self.train_epoch(epoch)
            
            print(f"Epoch {epoch+1:03d}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                  f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.2f}%, "
                  f"lr={lr:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"search_epoch_{epoch+1}.pth")
        
        # Get final architecture
        genotype = self.model.genotype()
        print("Final Architecture:")
        print(genotype)
        
        # Save final results
        self.save_results(genotype)
        self.plot_search_progress()
        
        return genotype
    
    def save_checkpoint(self, filename):
        """Save search checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'arch_optimizer_state_dict': self.arch_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'genotype': self.model.genotype()
        }
        torch.save(checkpoint, f"checkpoints/{filename}")
    
    def save_results(self, genotype):
        """Save search results"""
        results = {
            'genotype': genotype,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'final_valid_accuracy': self.valid_accuracies[-1]
        }
        
        with open('search_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_search_progress(self):
        """Plot search progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.valid_losses, label='Valid Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.valid_accuracies, label='Valid Acc', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Architecture weights evolution (normal cell)
        alphas_normal = F.softmax(self.model.alphas_normal, dim=-1).detach().cpu().numpy()
        for i, op_name in enumerate(OperationSpace.PRIMITIVES):
            ax3.plot(alphas_normal[:, i].mean(axis=0), label=op_name)
        ax3.set_xlabel('Edge')
        ax3.set_ylabel('Weight')
        ax3.set_title('Normal Cell Architecture Weights')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Architecture weights evolution (reduction cell)
        alphas_reduce = F.softmax(self.model.alphas_reduce, dim=-1).detach().cpu().numpy()
        for i, op_name in enumerate(OperationSpace.PRIMITIVES):
            ax4.plot(alphas_reduce[:, i].mean(axis=0), label=op_name)
        ax4.set_xlabel('Edge')
        ax4.set_ylabel('Weight')
        ax4.set_title('Reduction Cell Architecture Weights')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('darts_search_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

class ArchitectureEvaluator:
    """Evaluate discovered architectures"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_flops_params(self, model, input_size=(1, 3, 32, 32)):
        """Evaluate FLOPs and parameters"""
        model.eval()
        input = torch.randn(input_size).to(self.device)
        
        flops, params = profile(model, inputs=(input,), verbose=False)
        
        return {
            'flops': flops,
            'params': params,
            'flops_m': flops / 1e6,
            'params_m': params / 1e6
        }
    
    def evaluate_latency(self, model, input_size=(1, 3, 32, 32), num_runs=100):
        """Evaluate inference latency"""
        model.eval()
        input = torch.randn(input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input)
        
        # Measure latency
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'latency_ms': avg_latency,
            'fps': 1000 / avg_latency
        }
    
    def compare_architectures(self, architectures, input_size=(1, 3, 32, 32)):
        """Compare multiple architectures"""
        results = []
        
        for i, arch in enumerate(architectures):
            print(f"Evaluating architecture {i+1}/{len(architectures)}")
            
            # Build model from genotype (simplified)
            model = DARTSNetwork(C=16, num_classes=10, layers=8)
            model = model.to(self.device)
            
            # Evaluate metrics
            flops_params = self.evaluate_flops_params(model, input_size)
            latency = self.evaluate_latency(model, input_size)
            
            result = {
                'architecture_id': i,
                'genotype': arch,
                **flops_params,
                **latency
            }
            results.append(result)
        
        return results
    
    def plot_pareto_frontier(self, results, x_metric='flops_m', y_metric='latency_ms'):
        """Plot Pareto frontier of architectures"""
        x_vals = [r[x_metric] for r in results]
        y_vals = [r[y_metric] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, alpha=0.7, s=50)
        
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            plt.annotate(f'Arch {i}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel(f'{x_metric.replace("_", " ").title()}')
        plt.ylabel(f'{y_metric.replace("_", " ").title()}')
        plt.title('Architecture Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_architecture(genotype, filename='architecture.png'):
    """Visualize discovered architecture"""
    try:
        import graphviz
        
        dot = graphviz.Digraph(comment='Neural Architecture')
        dot.attr(rankdir='TB')
        
        # Add nodes for inputs
        dot.node('input0', 'Input 0', shape='box', style='filled', fillcolor='lightblue')
        dot.node('input1', 'Input 1', shape='box', style='filled', fillcolor='lightblue')
        
        # Add intermediate nodes
        for i in range(2, 6):  # Assuming 4 intermediate nodes
            dot.node(f'node{i}', f'Node {i}', shape='circle', style='filled', fillcolor='lightgreen')
        
        # Add edges based on genotype
        for op, from_node in genotype['normal']:
            if op != 'none':
                dot.edge(f'input{from_node}' if from_node < 2 else f'node{from_node}', 
                        f'node{len(genotype["normal"])//2 + 2}', 
                        label=op, color='blue')
        
        # Add output node
        dot.node('output', 'Output', shape='box', style='filled', fillcolor='lightcoral')
        
        # Connect to output
        for i in genotype['normal_concat']:
            dot.edge(f'node{i}', 'output', color='red')
        
        # Render
        dot.render(filename.replace('.png', ''), format='png', cleanup=True)
        print(f"Architecture visualization saved as {filename}")
        
    except ImportError:
        print("graphviz not available for visualization")

def demo_neural_architecture_search():
    """Demonstration of Neural Architecture Search"""
    
    print("Neural Architecture Search Demo")
    print("=" * 50)
    
    # DARTS Search
    print("\n--- DARTS Search Demo ---")
    
    trainer = DARTSTrainer(
        dataset='cifar10',
        batch_size=16,  # Smaller batch for demo
        epochs=5,  # Fewer epochs for demo
        learning_rate=0.025
    )
    
    # Search for architecture
    genotype = trainer.search()
    
    # Visualize discovered architecture
    print("\nDiscovered Architecture:")
    print(f"Normal: {genotype['normal']}")
    print(f"Reduce: {genotype['reduce']}")
    
    # Visualize architecture
    visualize_architecture(genotype)
    
    # Evaluate architecture
    print("\n--- Architecture Evaluation ---")
    
    evaluator = ArchitectureEvaluator()
    
    # Create model for evaluation
    model = DARTSNetwork(C=16, num_classes=10, layers=8)
    model = model.to(evaluator.device)
    
    # Evaluate efficiency metrics
    efficiency = evaluator.evaluate_flops_params(model)
    latency = evaluator.evaluate_latency(model)
    
    print(f"FLOPs: {efficiency['flops_m']:.2f}M")
    print(f"Parameters: {efficiency['params_m']:.2f}M")
    print(f"Latency: {latency['latency_ms']:.2f}ms")
    print(f"FPS: {latency['fps']:.1f}")
    
    # Compare with multiple random architectures
    print("\n--- Architecture Comparison ---")
    
    # Generate some random genotypes for comparison (simplified)
    random_genotypes = []
    for _ in range(5):
        normal = []
        reduce = []
        for i in range(4):  # 4 steps
            for j in range(i + 2):  # connections
                op = np.random.choice(OperationSpace.PRIMITIVES[1:])  # Exclude 'none'
                normal.append((op, j))
                reduce.append((op, j))
        
        random_genotypes.append({
            'normal': normal,
            'reduce': reduce,
            'normal_concat': [2, 3, 4, 5],
            'reduce_concat': [2, 3, 4, 5]
        })
    
    # Add discovered architecture
    all_genotypes = [genotype] + random_genotypes
    
    # Evaluate all architectures
    results = evaluator.compare_architectures(all_genotypes)
    
    # Print comparison
    print("\nArchitecture Comparison:")
    print("ID | FLOPs(M) | Params(M) | Latency(ms) | FPS")
    print("-" * 50)
    for i, result in enumerate(results):
        label = "DARTS" if i == 0 else f"Random{i}"
        print(f"{label:6} | {result['flops_m']:8.2f} | {result['params_m']:9.2f} | "
              f"{result['latency_ms']:11.2f} | {result['fps']:3.1f}")
    
    # Plot comparison
    evaluator.plot_pareto_frontier(results)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('--method', type=str, default='darts',
                       choices=['darts', 'pnas', 'enas', 'pc_darts'],
                       help='NAS method')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'imagenet'],
                       help='Dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Search epochs')
    parser.add_argument('--lr', type=float, default=0.025,
                       help='Learning rate')
    parser.add_argument('--mode', type=str, default='search',
                       choices=['search', 'eval', 'demo'],
                       help='Mode: search, eval, or demo')
    parser.add_argument('--genotype_path', type=str,
                       help='Path to genotype for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_neural_architecture_search()
    
    elif args.mode == 'search':
        if args.method == 'darts':
            trainer = DARTSTrainer(
                dataset=args.dataset,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr
            )
            genotype = trainer.search()
            
            # Visualize discovered architecture
            visualize_architecture(genotype)
        
        else:
            raise NotImplementedError(f"Method {args.method} not implemented yet")
    
    elif args.mode == 'eval':
        if not args.genotype_path:
            raise ValueError("Genotype path required for evaluation")
        
        # Load genotype
        with open(args.genotype_path, 'r') as f:
            genotype = json.load(f)
        
        # Evaluate architecture
        evaluator = ArchitectureEvaluator()
        
        # Create model from genotype
        model = DARTSNetwork(C=16, num_classes=10, layers=8)
        model = model.to(evaluator.device)
        
        # Evaluate metrics
        efficiency = evaluator.evaluate_flops_params(model)
        latency = evaluator.evaluate_latency(model)
        
        print("Architecture Evaluation Results:")
        print(f"FLOPs: {efficiency['flops_m']:.2f}M")
        print(f"Parameters: {efficiency['params_m']:.2f}M")
        print(f"Latency: {latency['latency_ms']:.2f}ms")
        print(f"FPS: {latency['fps']:.1f}")
        
        # Visualize architecture
        visualize_architecture(genotype)

if __name__ == "__main__":
    main()
