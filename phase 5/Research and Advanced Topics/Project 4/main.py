"""
Few-Shot Learning Implementation
===============================

This module implements state-of-the-art few-shot learning methods for computer vision.
Few-shot learning enables models to learn new concepts from only a few examples.

Methods implemented:
- Prototypical Networks: Distance-based classification with prototype learning
- MAML: Model-Agnostic Meta-Learning for gradient-based adaptation
- Relation Networks: Learnable similarity metrics
- Matching Networks: Attention-based matching for one-shot learning
- Memory-Augmented Networks: External memory for rapid binding

Author: Computer Vision Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot, ImageFolder
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
from PIL import Image
import copy
from scipy.stats import sem, t
try:
    import higher
except ImportError:
    print("Warning: 'higher' library not found. MAML implementation will be limited.")
    higher = None

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Conv4Backbone(nn.Module):
    """4-layer CNN backbone for few-shot learning"""
    
    def __init__(self, hidden_size=64, output_size=64):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Output projection
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EpisodeSampler(Sampler):
    """Sampler for episodic training"""
    
    def __init__(self, dataset, n_way, k_shot, query_shots, episodes_per_epoch=1000):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots
        self.episodes_per_epoch = episodes_per_epoch
        
        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            # Sample N classes
            episode_classes = random.sample(self.classes, self.n_way)
            
            support_indices = []
            query_indices = []
            
            for class_idx in episode_classes:
                class_indices = self.class_to_indices[class_idx]
                
                # Sample K+query_shots examples from this class
                sampled_indices = random.sample(class_indices, self.k_shot + self.query_shots)
                
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])
            
            # Yield episode as (support_indices, query_indices, episode_classes)
            yield support_indices + query_indices, episode_classes
    
    def __len__(self):
        return self.episodes_per_epoch

class FewShotDataset(Dataset):
    """Dataset wrapper for few-shot learning"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_episode_batch(batch_data, episode_classes, n_way, k_shot, query_shots):
    """Create episodic batch from sampled data"""
    images, labels = zip(*batch_data)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # Create mapping from original labels to episode labels (0 to n_way-1)
    label_map = {class_label: i for i, class_label in enumerate(episode_classes)}
    episode_labels = torch.tensor([label_map[label.item()] for label in labels])
    
    # Split into support and query
    support_size = n_way * k_shot
    
    support_images = images[:support_size]
    support_labels = episode_labels[:support_size]
    query_images = images[support_size:]
    query_labels = episode_labels[support_size:]
    
    return support_images, support_labels, query_images, query_labels

class PrototypicalNetworks(nn.Module):
    """Prototypical Networks for Few-Shot Learning"""
    
    def __init__(self, backbone, embedding_dim=64):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
    
    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """Forward pass for prototypical networks"""
        
        # Get embeddings
        support_embeddings = self.backbone(support_images)
        query_embeddings = self.backbone(query_images)
        
        # Compute prototypes (class centroids)
        prototypes = []
        for class_idx in range(n_way):
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances from queries to prototypes
        distances = self.euclidean_distance(query_embeddings, prototypes)
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        return logits
    
    def euclidean_distance(self, x, y):
        """Compute euclidean distance between x and y"""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        return torch.pow(x - y, 2).sum(2)

class RelationNetwork(nn.Module):
    """Relation Networks for Few-Shot Learning"""
    
    def __init__(self, backbone, relation_dim=8):
        super().__init__()
        self.backbone = backbone
        self.relation_dim = relation_dim
        
        # Relation module
        self.relation_module = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(64, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """Forward pass for relation networks"""
        
        # Get feature maps (not just embeddings)
        support_features = self.get_feature_maps(support_images)
        query_features = self.get_feature_maps(query_images)
        
        # Compute class representatives (prototypes)
        support_features_reshaped = support_features.view(n_way, k_shot, *support_features.shape[1:])
        class_representatives = support_features_reshaped.mean(dim=1)  # Average over k_shot
        
        # Compute relations
        query_size = query_features.size(0)
        relations = []
        
        for i in range(query_size):
            query_feature = query_features[i:i+1]  # Keep batch dimension
            
            # Concatenate query with each class representative
            query_extended = query_feature.repeat(n_way, 1, 1, 1)
            concatenated = torch.cat([class_representatives, query_extended], dim=1)
            
            # Compute relation scores
            relation_scores = self.relation_module(concatenated)
            relations.append(relation_scores.squeeze())
        
        relations = torch.stack(relations)
        return relations
    
    def get_feature_maps(self, x):
        """Get feature maps from backbone (modify based on backbone architecture)"""
        # This is a simplified version - adapt based on your backbone
        x = self.backbone.features(x)  # Assuming backbone has 'features' attribute
        return x

class MatchingNetworks(nn.Module):
    """Matching Networks for One-Shot Learning"""
    
    def __init__(self, backbone, fce=True, lstm_layers=1, lstm_input_size=64):
        super().__init__()
        self.backbone = backbone
        self.fce = fce
        
        if fce:
            # Full Context Embeddings
            self.lstm = nn.LSTM(
                lstm_input_size, lstm_input_size, 
                lstm_layers, batch_first=True, bidirectional=True
            )
            self.attention_lstm = nn.LSTMCell(lstm_input_size, lstm_input_size)
    
    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """Forward pass for matching networks"""
        
        # Get embeddings
        support_embeddings = self.backbone(support_images)
        query_embeddings = self.backbone(query_images)
        
        if self.fce:
            # Apply Full Context Embeddings
            support_embeddings = self.apply_fce(support_embeddings)
        
        # Compute attention weights
        logits = []
        for query_emb in query_embeddings:
            # Compute cosine similarity with support embeddings
            query_emb = query_emb.unsqueeze(0)
            cosine_sim = F.cosine_similarity(query_emb, support_embeddings, dim=1)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(cosine_sim, dim=0)
            
            # Compute weighted sum of support labels
            logit_per_class = torch.zeros(n_way).to(query_emb.device)
            for i, label in enumerate(support_labels):
                logit_per_class[label] += attention_weights[i]
            
            logits.append(logit_per_class)
        
        return torch.stack(logits)
    
    def apply_fce(self, embeddings):
        """Apply Full Context Embeddings"""
        # Bidirectional LSTM
        output, _ = self.lstm(embeddings.unsqueeze(0))
        
        # Take the mean of forward and backward directions
        forward = output[:, :, :embeddings.size(1)]
        backward = output[:, :, embeddings.size(1):]
        fce_embeddings = (forward + backward) / 2
        
        return fce_embeddings.squeeze(0)

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning"""
    
    def __init__(self, backbone, n_way, inner_lr=0.01, inner_steps=5, first_order=False):
        super().__init__()
        self.backbone = backbone
        self.n_way = n_way
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Classification head
        self.classifier = nn.Linear(backbone.output_size, n_way)
    
    def forward(self, support_images, support_labels, query_images, query_labels=None):
        """Forward pass for MAML"""
        
        if higher is None:
            # Fallback implementation without higher library
            return self.forward_simple(support_images, support_labels, query_images)
        
        # Use higher library for efficient gradient computation
        with higher.innerloop_ctx(self, self.inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
            
            # Inner loop adaptation
            for step in range(self.inner_steps):
                # Forward pass on support set
                support_features = fnet.backbone(support_images)
                support_logits = fnet.classifier(support_features)
                
                # Compute loss
                inner_loss = F.cross_entropy(support_logits, support_labels)
                
                # Update parameters
                diffopt.step(inner_loss)
            
            # Forward pass on query set with adapted parameters
            query_features = fnet.backbone(query_images)
            query_logits = fnet.classifier(query_features)
            
            return query_logits
    
    def forward_simple(self, support_images, support_labels, query_images):
        """Simple MAML implementation without higher library"""
        
        # Clone parameters for inner loop
        fast_weights = OrderedDict(self.named_parameters())
        
        # Inner loop adaptation
        for step in range(self.inner_steps):
            # Forward pass on support set
            support_features = self.backbone(support_images)
            support_logits = F.linear(support_features, fast_weights['classifier.weight'], 
                                    fast_weights['classifier.bias'])
            
            # Compute loss
            inner_loss = F.cross_entropy(support_logits, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(inner_loss, fast_weights.values(), 
                                      create_graph=not self.first_order)
            
            # Update fast weights
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
        
        # Forward pass on query set with adapted parameters
        query_features = self.backbone(query_images)
        query_logits = F.linear(query_features, fast_weights['classifier.weight'], 
                              fast_weights['classifier.bias'])
        
        return query_logits
    
    def create_inner_optimizer(self):
        """Create inner loop optimizer"""
        if higher is not None:
            self.inner_optimizer = optim.SGD(self.parameters(), lr=self.inner_lr)

class FewShotTrainer:
    """Base trainer for few-shot learning methods"""
    
    def __init__(self, model, n_way=5, k_shot=1, query_shots=15, 
                 meta_lr=0.001, device='cuda'):
        
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots
        self.meta_lr = meta_lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = self.model.to(self.device)
        
        # Meta optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # For MAML
        if isinstance(self.model, MAML):
            self.model.create_inner_optimizer()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_episode(self, episode_data):
        """Train on a single episode"""
        
        support_images, support_labels, query_images, query_labels, episode_classes = episode_data
        
        # Move to device
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward pass
        if isinstance(self.model, MAML):
            logits = self.model(support_images, support_labels, query_images, query_labels)
        else:
            logits = self.model(support_images, support_labels, query_images, 
                              self.n_way, self.k_shot)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_labels).float().mean()
        
        return loss, accuracy
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_accuracy = 0
        num_episodes = 0
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (indices, episode_classes) in enumerate(pbar):
            
            # Get episode data
            batch_data = [dataloader.dataset[i] for i in indices]
            episode_data = create_episode_batch(
                batch_data, episode_classes, self.n_way, self.k_shot, self.query_shots
            )
            episode_data = episode_data + (episode_classes,)
            
            # Train on episode
            self.meta_optimizer.zero_grad()
            loss, accuracy = self.train_episode(episode_data)
            loss.backward()
            self.meta_optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_episodes += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.3f}'
            })
        
        avg_loss = total_loss / num_episodes
        avg_accuracy = total_accuracy / num_episodes
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, dataloader, num_episodes=1000):
        """Evaluate model on test episodes"""
        self.model.eval()
        
        total_accuracy = 0
        accuracies = []
        
        with torch.no_grad():
            episode_count = 0
            
            for batch_idx, (indices, episode_classes) in enumerate(dataloader):
                if episode_count >= num_episodes:
                    break
                
                # Get episode data
                batch_data = [dataloader.dataset[i] for i in indices]
                episode_data = create_episode_batch(
                    batch_data, episode_classes, self.n_way, self.k_shot, self.query_shots
                )
                episode_data = episode_data + (episode_classes,)
                
                # Evaluate episode
                _, accuracy = self.train_episode(episode_data)
                
                total_accuracy += accuracy.item()
                accuracies.append(accuracy.item())
                episode_count += 1
        
        # Compute confidence interval
        accuracies = np.array(accuracies)
        mean_accuracy = accuracies.mean()
        confidence_interval = 1.96 * sem(accuracies)  # 95% confidence interval
        
        return mean_accuracy, confidence_interval
    
    def train(self, train_dataset, val_dataset=None, epochs=100, episodes_per_epoch=1000,
              val_episodes=600):
        """Full training loop"""
        
        print(f"Starting few-shot training on {self.device}")
        print(f"Method: {self.model.__class__.__name__}")
        print(f"Setting: {self.n_way}-way {self.k_shot}-shot")
        
        # Create data loaders
        train_sampler = EpisodeSampler(
            train_dataset, self.n_way, self.k_shot, self.query_shots, episodes_per_epoch
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        
        if val_dataset is not None:
            val_sampler = EpisodeSampler(
                val_dataset, self.n_way, self.k_shot, self.query_shots, val_episodes
            )
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
        
        # Training loop
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_dataset is not None and (epoch + 1) % 10 == 0:
                val_acc, val_ci = self.evaluate(val_loader, val_episodes)
                self.val_accuracies.append(val_acc)
                
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                      f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}±{val_ci:.3f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(f"best_{self.model.__class__.__name__.lower()}.pth")
            else:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"{self.model.__class__.__name__.lower()}_epoch_{epoch+1}.pth")
        
        # Plot training progress
        self.plot_training_progress()
        
        return best_val_acc
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'query_shots': self.query_shots
        }
        torch.save(checkpoint, f"checkpoints/{filename}")
        print(f"Checkpoint saved: checkpoints/{filename}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        if self.val_accuracies:
            val_epochs = [i * 10 for i in range(len(self.val_accuracies))]
            ax2.plot(val_epochs, self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.model.__class__.__name__.lower()}_training_progress.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

def create_omniglot_dataset(root='./data', download=True):
    """Create Omniglot dataset for few-shot learning"""
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Background set (training)
    background_set = Omniglot(root, background=True, download=download, transform=transform)
    
    # Evaluation set (testing)
    evaluation_set = Omniglot(root, background=False, download=download, transform=transform)
    
    # Convert to RGB (Omniglot is grayscale)
    class OmniglotRGB(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            # Convert grayscale to RGB
            image = image.repeat(3, 1, 1)
            return image, label
    
    return OmniglotRGB(background_set), OmniglotRGB(evaluation_set)

def create_mini_imagenet_dataset(root='./data'):
    """Create miniImageNet dataset (simplified version using regular ImageNet structure)"""
    
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # For demonstration, we'll create a synthetic dataset
    # In practice, you would download the actual miniImageNet dataset
    print("Note: Using synthetic miniImageNet data for demonstration")
    print("Please download the actual miniImageNet dataset for real experiments")
    
    class SyntheticMiniImageNet(Dataset):
        def __init__(self, num_classes=64, samples_per_class=600, image_size=(3, 84, 84)):
            self.num_classes = num_classes
            self.samples_per_class = samples_per_class
            self.image_size = image_size
            self.transform = transform
        
        def __len__(self):
            return self.num_classes * self.samples_per_class
        
        def __getitem__(self, idx):
            class_id = idx // self.samples_per_class
            # Generate synthetic image
            image = torch.randn(self.image_size)
            
            if self.transform:
                # Convert to PIL for transforms
                image_pil = transforms.ToPILImage()(image)
                image = self.transform(image_pil)
            
            return image, class_id
    
    train_dataset = SyntheticMiniImageNet(num_classes=64)
    val_dataset = SyntheticMiniImageNet(num_classes=16) 
    test_dataset = SyntheticMiniImageNet(num_classes=20)
    
    return train_dataset, val_dataset, test_dataset

def compare_few_shot_methods():
    """Compare different few-shot learning methods"""
    
    print("Few-Shot Learning Methods Comparison")
    print("=" * 50)
    
    # Create datasets
    train_dataset, test_dataset = create_omniglot_dataset()
    
    # Settings
    n_way = 5
    k_shot = 1
    query_shots = 15
    
    methods = {
        'Prototypical': PrototypicalNetworks(Conv4Backbone(), 64),
        'Relation': RelationNetwork(Conv4Backbone()),
        'Matching': MatchingNetworks(Conv4Backbone()),
        'MAML': MAML(Conv4Backbone(), n_way)
    }
    
    results = {}
    
    for method_name, model in methods.items():
        print(f"\n--- Training {method_name} Networks ---")
        
        trainer = FewShotTrainer(
            model=model,
            n_way=n_way,
            k_shot=k_shot,
            query_shots=query_shots,
            meta_lr=0.001
        )
        
        # Train for fewer epochs in demo
        best_acc = trainer.train(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            epochs=20,  # Reduced for demo
            episodes_per_epoch=100,  # Reduced for demo
            val_episodes=100  # Reduced for demo
        )
        
        results[method_name] = best_acc
        print(f"{method_name} Best Validation Accuracy: {best_acc:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(methods, accuracies, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title(f'Few-Shot Learning Comparison ({n_way}-way {k_shot}-shot)')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('few_shot_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def demo_few_shot_learning():
    """Demonstration of few-shot learning"""
    
    print("Few-Shot Learning Demo")
    print("=" * 50)
    
    # Create synthetic dataset for quick demo
    print("Creating demo dataset...")
    
    class DemoDataset(Dataset):
        def __init__(self, num_classes=20, samples_per_class=20):
            self.num_classes = num_classes
            self.samples_per_class = samples_per_class
            
        def __len__(self):
            return self.num_classes * self.samples_per_class
        
        def __getitem__(self, idx):
            class_id = idx // self.samples_per_class
            # Generate synthetic image with class-specific pattern
            image = torch.randn(3, 28, 28) + class_id * 0.1
            return image, class_id
    
    train_dataset = DemoDataset(num_classes=15, samples_per_class=20)
    test_dataset = DemoDataset(num_classes=5, samples_per_class=20)
    
    print(f"Train dataset: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
    print(f"Test dataset: {len(test_dataset)} samples, {test_dataset.num_classes} classes")
    
    # Demo Prototypical Networks
    print("\n--- Prototypical Networks Demo ---")
    
    model = PrototypicalNetworks(Conv4Backbone(hidden_size=32, output_size=32), 32)
    trainer = FewShotTrainer(
        model=model,
        n_way=5,
        k_shot=1,
        query_shots=5,
        meta_lr=0.001
    )
    
    # Quick training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=10,
        episodes_per_epoch=50,
        val_episodes=50
    )
    
    # Evaluate
    val_sampler = EpisodeSampler(test_dataset, 5, 1, 5, 50)
    val_loader = DataLoader(test_dataset, batch_sampler=val_sampler)
    
    final_acc, ci = trainer.evaluate(val_loader, 50)
    print(f"\nFinal Test Accuracy: {final_acc:.3f} ± {ci:.3f}")
    
    # Demo episode visualization
    print("\n--- Episode Structure Demo ---")
    
    # Sample an episode
    sampler = EpisodeSampler(test_dataset, n_way=3, k_shot=2, query_shots=3, episodes_per_epoch=1)
    sample_loader = DataLoader(test_dataset, batch_sampler=sampler)
    
    for indices, episode_classes in sample_loader:
        batch_data = [test_dataset[i] for i in indices]
        support_images, support_labels, query_images, query_labels = create_episode_batch(
            batch_data, episode_classes, 3, 2, 3
        )
        
        print(f"Episode classes: {episode_classes}")
        print(f"Support set: {support_images.shape}, labels: {support_labels}")
        print(f"Query set: {query_images.shape}, labels: {query_labels}")
        
        # Visualize episode
        fig, axes = plt.subplots(2, 6, figsize=(15, 6))
        
        # Support set
        for i in range(6):
            if i < len(support_images):
                img = support_images[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Support Class {support_labels[i].item()}')
                axes[0, i].axis('off')
            else:
                axes[0, i].axis('off')
        
        # Query set
        for i in range(6):
            if i < len(query_images):
                img = query_images[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
                axes[1, i].imshow(img)
                axes[1, i].set_title(f'Query Class {query_labels[i].item()}')
                axes[1, i].axis('off')
            else:
                axes[1, i].axis('off')
        
        plt.suptitle('Few-Shot Episode Structure (3-way 2-shot)')
        plt.tight_layout()
        plt.savefig('episode_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
        break

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Few-Shot Learning')
    parser.add_argument('--method', type=str, default='prototypical',
                       choices=['prototypical', 'maml', 'relation', 'matching'],
                       help='Few-shot learning method')
    parser.add_argument('--dataset', type=str, default='omniglot',
                       choices=['omniglot', 'miniimagenet'],
                       help='Dataset')
    parser.add_argument('--n_way', type=int, default=5,
                       help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='Number of examples per class')
    parser.add_argument('--query_shots', type=int, default=15,
                       help='Number of query examples per class')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of meta-training epochs')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                       help='Meta learning rate')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'demo', 'compare'],
                       help='Mode: train, eval, demo, or compare')
    parser.add_argument('--model_path', type=str,
                       help='Path to pre-trained model for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_few_shot_learning()
    
    elif args.mode == 'compare':
        compare_few_shot_methods()
    
    elif args.mode == 'train':
        # Create datasets
        if args.dataset == 'omniglot':
            train_dataset, test_dataset = create_omniglot_dataset()
        elif args.dataset == 'miniimagenet':
            train_dataset, val_dataset, test_dataset = create_mini_imagenet_dataset()
        
        # Create model
        backbone = Conv4Backbone()
        
        if args.method == 'prototypical':
            model = PrototypicalNetworks(backbone)
        elif args.method == 'maml':
            model = MAML(backbone, args.n_way)
        elif args.method == 'relation':
            model = RelationNetwork(backbone)
        elif args.method == 'matching':
            model = MatchingNetworks(backbone)
        
        # Train
        trainer = FewShotTrainer(
            model=model,
            n_way=args.n_way,
            k_shot=args.k_shot,
            query_shots=args.query_shots,
            meta_lr=args.meta_lr
        )
        
        val_data = test_dataset if args.dataset == 'omniglot' else val_dataset
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_data,
            epochs=args.epochs
        )
    
    elif args.mode == 'eval':
        if not args.model_path:
            raise ValueError("Model path required for evaluation")
        
        # Load model and evaluate
        # Implementation depends on saved checkpoint format
        print("Evaluation mode - implement based on your checkpoint format")

if __name__ == "__main__":
    main()
