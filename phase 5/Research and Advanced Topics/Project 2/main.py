"""
Self-Supervised Learning Implementation
=====================================

This module implements state-of-the-art self-supervised learning methods
for visual representation learning without human annotations.

Methods implemented:
- SimCLR: Simple Contrastive Learning of Visual Representations
- MoCo: Momentum Contrast
- BYOL: Bootstrap Your Own Latent
- SwAV: Swapping Assignments between Views
- MAE: Masked Autoencoder

Author: Computer Vision Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import cv2
import os
import random
import argparse
from tqdm import tqdm
import wandb
import yaml
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimCLRAugmentation:
    """Data augmentation pipeline for SimCLR"""
    
    def __init__(self, image_size=224, strength=1.0):
        self.image_size = image_size
        self.strength = strength
        
        # Color jittering parameters
        s = self.strength
        self.color_jitter = transforms.ColorJitter(
            brightness=0.8*s, contrast=0.8*s, 
            saturation=0.8*s, hue=0.2*s
        )
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.2, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * image_size), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class ContrastiveDataset(Dataset):
    """Dataset wrapper for contrastive learning"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            # Return two augmented views
            view1, view2 = self.transform(image)
            return view1, view2, label
        else:
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image)
            return image, label

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class SimCLRModel(nn.Module):
    """SimCLR: Simple Contrastive Learning of Visual Representations"""
    
    def __init__(self, backbone='resnet50', projection_dim=128):
        super().__init__()
        
        # Backbone encoder
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        encoder_dim = self.encoder.num_features
        
        # Projection head
        self.projection_head = ProjectionHead(
            encoder_dim, encoder_dim, projection_dim
        )
        
        self.temperature = 0.1
    
    def forward(self, x):
        # Extract features
        features = self.encoder(x)
        # Project to contrastive space
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
    
    def contrastive_loss(self, z1, z2):
        """NT-Xent loss (Normalized Temperature-scaled Cross Entropy)"""
        batch_size = z1.shape[0]
        
        # Concatenate features
        z = torch.cat([z1, z2], dim=0)  # 2N x D
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class MomentumEncoder(nn.Module):
    """Momentum encoder for MoCo"""
    
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Initialize momentum encoder
        self.momentum_encoder = self._copy_encoder(encoder)
        
        # Freeze momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    def _copy_encoder(self, encoder):
        """Create a copy of the encoder"""
        momentum_encoder = type(encoder)(
            backbone=encoder.encoder.__class__.__name__.lower(),
            projection_dim=encoder.projection_head.projection[-1].out_features
        )
        momentum_encoder.load_state_dict(encoder.state_dict())
        return momentum_encoder
    
    @torch.no_grad()
    def update_momentum_encoder(self, encoder):
        """Update momentum encoder with exponential moving average"""
        for param_q, param_k in zip(encoder.parameters(), 
                                  self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1. - self.momentum)
    
    def forward(self, x):
        return self.momentum_encoder(x)

class MoCoModel(nn.Module):
    """MoCo: Momentum Contrast"""
    
    def __init__(self, backbone='resnet50', projection_dim=128, 
                 queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        
        # Query encoder
        self.encoder_q = SimCLRModel(backbone, projection_dim)
        
        # Key encoder (momentum)
        self.encoder_k = MomentumEncoder(self.encoder_q, momentum)
        
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """Forward pass for training"""
        
        # Compute query features
        q = self.encoder_q(im_q)
        
        # Compute key features
        with torch.no_grad():
            # Update momentum encoder
            self.encoder_k.update_momentum_encoder(self.encoder_q)
            
            k = self.encoder_k(im_k)
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels

class PredictorHead(nn.Module):
    """Predictor head for BYOL"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class BYOLModel(nn.Module):
    """BYOL: Bootstrap Your Own Latent"""
    
    def __init__(self, backbone='resnet50', projection_dim=256, 
                 hidden_dim=4096, momentum=0.99):
        super().__init__()
        
        # Online network
        self.online_encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        encoder_dim = self.online_encoder.num_features
        
        self.online_projector = ProjectionHead(encoder_dim, hidden_dim, projection_dim)
        self.predictor = PredictorHead(projection_dim, hidden_dim, projection_dim)
        
        # Target network
        self.target_encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.target_projector = ProjectionHead(encoder_dim, hidden_dim, projection_dim)
        
        # Initialize target network
        self._copy_params()
        
        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
    
    def _copy_params(self):
        """Copy parameters from online to target network"""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data.copy_(online_param.data)
    
    @torch.no_grad()
    def _update_target_network(self):
        """Update target network with exponential moving average"""
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = target_param.data * self.momentum + \
                              online_param.data * (1. - self.momentum)
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data = target_param.data * self.momentum + \
                              online_param.data * (1. - self.momentum)
    
    def forward(self, x1, x2):
        """Forward pass for training"""
        
        # Online network forward pass
        online_feat_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_feat_1)
        online_pred_1 = self.predictor(online_proj_1)
        
        online_feat_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_feat_2)
        online_pred_2 = self.predictor(online_proj_2)
        
        # Target network forward pass
        with torch.no_grad():
            target_feat_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_feat_1)
            
            target_feat_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_feat_2)
        
        # Update target network
        self._update_target_network()
        
        return online_pred_1, online_pred_2, target_proj_1, target_proj_2
    
    def loss_fn(self, p1, p2, z1, z2):
        """BYOL loss function"""
        def regression_loss(p, z):
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return 2 - 2 * (p * z).sum(dim=1).mean()
        
        loss1 = regression_loss(p1, z2)
        loss2 = regression_loss(p2, z1)
        return (loss1 + loss2) / 2

class MaskedAutoencoder(nn.Module):
    """MAE: Masked Autoencoder for vision"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=encoder_num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True
            ) for _ in range(encoder_depth)
        ])
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim, nhead=decoder_num_heads,
                dim_feedforward=int(decoder_embed_dim * mlp_ratio),
                batch_first=True
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        # Initialize positional embeddings
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], 3, h * p, h * p)
        return x
    
    def random_masking(self, x, mask_ratio):
        """Random masking"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder"""
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder"""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Decoder
        for layer in self.decoder:
            x = layer(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss"""
        target = self.patchify(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        
        loss = (loss * mask).sum() / mask.sum()  # Mean on removed patches
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        """Forward pass"""
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class SelfSupervisedTrainer:
    """Trainer for self-supervised learning methods"""
    
    def __init__(self, method='simclr', backbone='resnet50', 
                 batch_size=256, lr=0.03, epochs=100, device='cuda'):
        self.method = method
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        # For logging
        self.train_losses = []
    
    def _create_model(self):
        """Create model based on method"""
        if self.method == 'simclr':
            return SimCLRModel(self.backbone)
        elif self.method == 'moco':
            return MoCoModel(self.backbone)
        elif self.method == 'byol':
            return BYOLModel(self.backbone)
        elif self.method == 'mae':
            return MaskedAutoencoder()
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Training {self.method.upper()}")
        
        for batch_idx, batch in enumerate(pbar):
            
            if self.method in ['simclr', 'byol']:
                view1, view2, _ = batch
                view1, view2 = view1.to(self.device), view2.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.method == 'simclr':
                    z1 = self.model(view1)
                    z2 = self.model(view2)
                    loss = self.model.contrastive_loss(z1, z2)
                
                elif self.method == 'byol':
                    p1, p2, z1, z2 = self.model(view1, view2)
                    loss = self.model.loss_fn(p1, p2, z1, z2)
            
            elif self.method == 'moco':
                view1, view2, _ = batch
                view1, view2 = view1.to(self.device), view2.to(self.device)
                
                self.optimizer.zero_grad()
                logits, labels = self.model(view1, view2)
                loss = F.cross_entropy(logits, labels)
            
            elif self.method == 'mae':
                images, _ = batch
                images = images.to(self.device)
                
                self.optimizer.zero_grad()
                loss, _, _ = self.model(images)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, dataset_path, val_dataset_path=None):
        """Full training loop"""
        print(f"Training {self.method.upper()} on {self.device}")
        
        # Prepare data
        if self.method == 'mae':
            # MAE uses single view
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            base_dataset = ImageFolder(dataset_path)
            dataset = ContrastiveDataset(base_dataset)
        else:
            # Contrastive methods use augmented pairs
            augmentation = SimCLRAugmentation()
            base_dataset = ImageFolder(dataset_path)
            dataset = ContrastiveDataset(base_dataset, augmentation)
        
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(dataloader)
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{self.method}_epoch_{epoch+1}.pth")
        
        # Final save
        self.save_checkpoint(f"{self.method}_final.pth")
        
        # Plot training curve
        self.plot_training_curve()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'method': self.method,
            'backbone': self.backbone
        }
        torch.save(checkpoint, f"checkpoints/{filename}")
        print(f"Checkpoint saved: checkpoints/{filename}")
    
    def plot_training_curve(self):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label=f'{self.method.upper()} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.method.upper()} Training Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.method}_training_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

class LinearEvaluator:
    """Linear evaluation of learned representations"""
    
    def __init__(self, model_path, method='simclr', backbone='resnet50'):
        self.method = method
        self.backbone = backbone
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pre-trained model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Extract encoder
        if method == 'simclr':
            self.encoder = self.model.encoder
        elif method == 'moco':
            self.encoder = self.model.encoder_q.encoder
        elif method == 'byol':
            self.encoder = self.model.online_encoder
        elif method == 'mae':
            # For MAE, use the encoder part
            self.encoder = lambda x: self.model.forward_encoder(x, mask_ratio=0)[0][:, 0]  # CLS token
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _load_model(self, model_path):
        """Load pre-trained model"""
        if self.method == 'simclr':
            model = SimCLRModel(self.backbone)
        elif self.method == 'moco':
            model = MoCoModel(self.backbone)
        elif self.method == 'byol':
            model = BYOLModel(self.backbone)
        elif self.method == 'mae':
            model = MaskedAutoencoder()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device)
    
    def extract_features(self, dataloader):
        """Extract features from dataset"""
        features = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if len(batch) == 3:  # Contrastive dataset
                    images, _, batch_labels = batch
                else:  # Regular dataset
                    images, batch_labels = batch
                
                images = images.to(self.device)
                
                # Extract features
                batch_features = self.encoder(images)
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.mean(dim=[2, 3])  # Global average pooling
                
                features.append(batch_features.cpu())
                labels.append(batch_labels)
        
        features = torch.cat(features, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        
        return features, labels
    
    def linear_evaluation(self, train_dataset, test_dataset):
        """Perform linear evaluation"""
        print("Performing linear evaluation...")
        
        # Prepare data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        train_dataset = ImageFolder(train_dataset, transform=transform)
        test_dataset = ImageFolder(test_dataset, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
        
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)
        
        # Train linear classifier
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        train_pred = classifier.predict(train_features)
        test_pred = classifier.predict(test_features)
        
        train_acc = accuracy_score(train_labels, train_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        
        print(f"Linear Evaluation Results:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return test_acc
    
    def visualize_representations(self, dataset_path, num_samples=1000):
        """Visualize learned representations using t-SNE"""
        print("Visualizing representations...")
        
        # Prepare data
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        dataset = ImageFolder(dataset_path, transform=transform)
        
        # Sample subset for visualization
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=4)
        
        # Extract features
        features, labels = self.extract_features(loader)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'{self.method.upper()} Learned Representations (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(f"{self.method}_tsne.png", dpi=300, bbox_inches='tight')
        plt.show()

def demo_self_supervised_learning():
    """Demonstration of self-supervised learning methods"""
    
    print("Self-Supervised Learning Demo")
    print("=" * 50)
    
    # Create synthetic dataset for demo
    print("Creating synthetic dataset...")
    os.makedirs('demo_data/train', exist_ok=True)
    os.makedirs('demo_data/test', exist_ok=True)
    
    # Create CIFAR-10 for demo
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Download CIFAR-10
    train_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Save some samples for demo
    class_names = train_dataset.classes
    for i, class_name in enumerate(class_names[:3]):  # Only first 3 classes for demo
        os.makedirs(f'demo_data/train/{class_name}', exist_ok=True)
        os.makedirs(f'demo_data/test/{class_name}', exist_ok=True)
        
        # Save training samples
        class_indices = [j for j, (_, label) in enumerate(train_dataset) if label == i]
        for k, idx in enumerate(class_indices[:50]):  # 50 samples per class
            image, _ = train_dataset[idx]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(f'demo_data/train/{class_name}/image_{k}.png')
        
        # Save test samples
        class_indices = [j for j, (_, label) in enumerate(test_dataset) if label == i]
        for k, idx in enumerate(class_indices[:20]):  # 20 samples per class
            image, _ = test_dataset[idx]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(f'demo_data/test/{class_name}/image_{k}.png')
    
    # Demo different methods
    methods = ['simclr', 'moco', 'byol']
    
    for method in methods:
        print(f"\n--- {method.upper()} Demo ---")
        
        # Train model
        trainer = SelfSupervisedTrainer(
            method=method, 
            backbone='resnet18',  # Smaller model for demo
            batch_size=32,  # Smaller batch for demo
            epochs=5,  # Fewer epochs for demo
            lr=0.03
        )
        
        trainer.train('demo_data/train')
        
        # Evaluate model
        evaluator = LinearEvaluator(
            model_path=f'checkpoints/{method}_final.pth',
            method=method,
            backbone='resnet18'
        )
        
        accuracy = evaluator.linear_evaluation(
            'demo_data/train', 'demo_data/test'
        )
        
        # Visualize representations
        evaluator.visualize_representations('demo_data/test')
        
        print(f"{method.upper()} Test Accuracy: {accuracy:.4f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Self-Supervised Learning')
    parser.add_argument('--method', type=str, default='simclr',
                       choices=['simclr', 'moco', 'byol', 'mae'],
                       help='Self-supervised learning method')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.03,
                       help='Learning rate')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to training dataset')
    parser.add_argument('--eval_dataset', type=str,
                       help='Path to evaluation dataset')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'demo'],
                       help='Mode: train, eval, or demo')
    parser.add_argument('--model_path', type=str,
                       help='Path to pre-trained model for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_self_supervised_learning()
    
    elif args.mode == 'train':
        trainer = SelfSupervisedTrainer(
            method=args.method,
            backbone=args.backbone,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        trainer.train(args.dataset)
    
    elif args.mode == 'eval':
        if not args.model_path:
            raise ValueError("Model path required for evaluation")
        
        evaluator = LinearEvaluator(
            model_path=args.model_path,
            method=args.method,
            backbone=args.backbone
        )
        
        if args.eval_dataset:
            accuracy = evaluator.linear_evaluation(args.dataset, args.eval_dataset)
            print(f"Linear Evaluation Accuracy: {accuracy:.4f}")
        
        evaluator.visualize_representations(args.dataset)

if __name__ == "__main__":
    main()
