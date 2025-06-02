"""
Vision Transformers (ViTs) Implementation

A comprehensive implementation of Vision Transformers including various architectures,
training strategies, and applications for computer vision tasks.

This module provides:
- Original ViT implementation
- Advanced variants (DeiT, Swin, PVT, CaiT, CrossViT)
- Training and evaluation utilities
- Attention visualization tools
- Transfer learning capabilities

Author: Computer Vision Researcher
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import math
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import argparse
import yaml
import os
import time
from pathlib import Path
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from transformers import ViTModel, ViTConfig as HFViTConfig
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""
    # Model architecture
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    
    # Training parameters
    dropout: float = 0.1
    emb_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Position encoding
    use_pos_encoding: bool = True
    pos_encoding_type: str = "learnable"  # learnable, sinusoidal, relative
    
    # Advanced features
    use_cls_token: bool = True
    use_distillation: bool = False
    pool: str = "cls"  # cls, mean
    
    # Regularization
    drop_path_rate: float = 0.1
    layer_scale: bool = False
    layer_scale_init_value: float = 1e-4

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Patch embeddings of shape [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.image_size}x{self.image_size})"
        
        # Convert to patches: [B, embed_dim, H//P, W//P]
        x = self.projection(x)
        
        # Flatten patches: [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # Transpose: [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x

class PositionalEncoding(nn.Module):
    """Various types of positional encoding."""
    
    def __init__(self, num_patches: int, embed_dim: int, encoding_type: str = "learnable"):
        super().__init__()
        self.encoding_type = encoding_type
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        if encoding_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        elif encoding_type == "sinusoidal":
            self.register_buffer("pos_embedding", self._get_sinusoidal_encoding())
        elif encoding_type == "relative":
            # Relative position encoding (simplified version)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * int(math.sqrt(num_patches)) - 1) ** 2, embed_dim)
            )
        
    def _get_sinusoidal_encoding(self) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(self.num_patches + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * 
                           -(math.log(10000.0) / self.embed_dim))
        
        pos_encoding = torch.zeros(1, self.num_patches + 1, self.embed_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        if self.encoding_type in ["learnable", "sinusoidal"]:
            return x + self.pos_embedding
        elif self.encoding_type == "relative":
            # Simplified relative position encoding
            return x
        else:
            return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [B, N, D]
            return_attention: Whether to return attention weights
        Returns:
            Output tensor and optionally attention weights
        """
        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if return_attention:
            return out, attn
        return out

class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0., 
                 drop_path: float = 0., layer_scale: bool = False, 
                 layer_scale_init_value: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Layer Scale (from CaiT)
        self.use_layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through transformer block."""
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
        else:
            attn_out = self.attn(self.norm1(x))
        
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attn_weights
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer implementation."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            config.image_size, config.patch_size, 3, config.dim
        )
        
        num_patches = self.patch_embedding.num_patches
        
        # Class token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches
        
        # Positional encoding
        if config.use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                num_patches, config.dim, config.pos_encoding_type
            )
        
        self.dropout = nn.Dropout(config.emb_dropout)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                drop_path=dpr[i],
                layer_scale=config.layer_scale,
                layer_scale_init_value=config.layer_scale_init_value
            )
            for i in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.dim)
        
        # Classification head
        self.head = nn.Linear(config.dim, config.num_classes)
        
        # Distillation head (for DeiT)
        if config.use_distillation:
            self.dist_token = nn.Parameter(torch.randn(1, 1, config.dim))
            self.head_dist = nn.Linear(config.dim, config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
            return_attention: Whether to return attention weights
        Returns:
            Class logits and optionally attention weights
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, num_patches, dim]
        
        # Add class token
        if self.config.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add distillation token (for DeiT)
        if self.config.use_distillation:
            dist_tokens = repeat(self.dist_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([x[:, :1], dist_tokens, x[:, 1:]], dim=1)
        
        # Add positional encoding
        if self.config.use_pos_encoding:
            x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        # Transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x)
        
        x = self.norm(x)
        
        # Classification
        if self.config.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, 0]  # Use CLS token
        
        logits = self.head(x)
        
        # Distillation output
        if self.config.use_distillation:
            logits_dist = self.head_dist(x[:, 1])
            if return_attention:
                return (logits, logits_dist), attention_weights
            return logits, logits_dist
        
        if return_attention:
            return logits, attention_weights
        return logits

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with Window-based and Shifted Window attention."""
    
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.,
                 dropout: float = 0., drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, dropout
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Create attention mask for shifted window
        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._create_attention_mask())
        else:
            self.attn_mask = None
    
    def _create_attention_mask(self) -> torch.Tensor:
        """Create attention mask for shifted window attention."""
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = self._window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition tensor into windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows
    
    def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin Transformer block."""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self._window_partition(shifted_x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window-based attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention for Swin Transformer."""
    
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through window attention."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionVisualizer:
    """Utility class for visualizing attention patterns."""
    
    def __init__(self, model: VisionTransformer):
        self.model = model
        self.model.eval()
    
    def generate_attention_maps(self, image: torch.Tensor, layers: List[int] = None,
                              heads: Union[str, List[int]] = "mean") -> Dict[str, torch.Tensor]:
        """
        Generate attention maps for specified layers and heads.
        
        Args:
            image: Input image tensor [B, C, H, W]
            layers: List of layer indices to visualize (default: all)
            heads: Which attention heads to visualize ("mean", "all", or list of indices)
        
        Returns:
            Dictionary mapping layer names to attention tensors
        """
        if layers is None:
            layers = list(range(len(self.model.transformer_blocks)))
        
        with torch.no_grad():
            _, attention_weights = self.model(image, return_attention=True)
        
        attention_maps = {}
        
        for layer_idx in layers:
            if layer_idx >= len(attention_weights):
                continue
                
            attn = attention_weights[layer_idx]  # [B, H, N, N]
            
            # Remove CLS token attention
            if self.model.config.use_cls_token:
                attn = attn[:, :, 0, 1:]  # [B, H, N-1] - attention from CLS to patches
            
            if heads == "mean":
                attn = attn.mean(dim=1)  # Average across heads
            elif heads == "all":
                pass  # Keep all heads
            elif isinstance(heads, list):
                attn = attn[:, heads]  # Select specific heads
            
            attention_maps[f"layer_{layer_idx}"] = attn
        
        return attention_maps
    
    def plot_attention_maps(self, attention_maps: Dict[str, torch.Tensor], 
                          original_image: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None):
        """Plot attention maps in a grid."""
        num_layers = len(attention_maps)
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, (layer_name, attn) in enumerate(attention_maps.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            # Convert attention to 2D map
            patch_size = self.model.config.patch_size
            image_size = self.model.config.image_size
            grid_size = image_size // patch_size
            
            attn_2d = attn[0].view(grid_size, grid_size).cpu().numpy()
            
            # Plot
            im = ax.imshow(attn_2d, cmap='viridis', interpolation='bilinear')
            ax.set_title(f'{layer_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for idx in range(num_layers, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row][col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class ViTTrainer:
    """Training utilities for Vision Transformers."""
    
    def __init__(self, config: ViTConfig, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.config = config
        self.device = device
        self.model = VisionTransformer(config).to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 0.05):
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=(0.9, 0.999))
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if self.config.use_distillation:
                logits, logits_dist = output
                loss = self.criterion(logits, target) + self.criterion(logits_dist, target)
                pred = logits.argmax(dim=1)
            else:
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if self.config.use_distillation:
                    logits, logits_dist = output
                    loss = self.criterion(logits, target) + self.criterion(logits_dist, target)
                    pred = logits.argmax(dim=1)
                else:
                    loss = self.criterion(output, target)
                    pred = output.argmax(dim=1)
                
                total_loss += loss.item()
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, learning_rate: float = 1e-3, save_path: str = "vit_model.pth"):
        """Complete training loop."""
        self.setup_optimizer(learning_rate)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class ViTTransferLearning:
    """Transfer learning utilities for Vision Transformers."""
    
    def __init__(self, pretrained_model: str, num_classes: int, freeze_backbone: bool = False):
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained model
        if pretrained_model.startswith('vit'):
            self.model = timm.create_model(pretrained_model, pretrained=True, num_classes=num_classes)
        else:
            # Load custom pre-trained model
            checkpoint = torch.load(pretrained_model, map_location='cpu')
            config = checkpoint['config']
            config.num_classes = num_classes
            
            self.model = VisionTransformer(config)
            
            # Load pre-trained weights (except classification head)
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = self.model.state_dict()
            
            # Filter out classification head weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and not k.startswith('head')}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for name, param in self.model.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
    
    def fine_tune(self, dataset: Dataset, epochs: int = 20, 
                  learning_rate: float = 1e-4, batch_size: int = 32):
        """Fine-tune the model on a custom dataset."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer (only for unfrozen parameters)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            avg_loss = total_loss / len(data_loader)
            accuracy = 100. * correct / total
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

def create_sample_config() -> ViTConfig:
    """Create a sample ViT configuration."""
    return ViTConfig(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )

def load_model_from_checkpoint(checkpoint_path: str) -> VisionTransformer:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = VisionTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def demo_attention_visualization():
    """Demonstrate attention visualization."""
    # Create a small ViT model for demo
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536
    )
    
    model = VisionTransformer(config)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Visualize attention
    visualizer = AttentionVisualizer(model)
    attention_maps = visualizer.generate_attention_maps(
        dummy_input, 
        layers=[2, 4, 5],
        heads="mean"
    )
    
    print("Attention maps generated for layers:", list(attention_maps.keys()))
    print("Attention map shapes:", {k: v.shape for k, v in attention_maps.items()})

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Vision Transformers Implementation')
    parser.add_argument('--mode', choices=['train', 'demo', 'visualize'], default='demo',
                       help='Mode of operation')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_path', type=str, help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Vision Transformer demo...")
        demo_attention_visualization()
    
    elif args.mode == 'train':
        print("Training Vision Transformer...")
        # Implementation would depend on dataset format
        print("Training mode requires dataset setup - see README for details")
    
    elif args.mode == 'visualize':
        if not args.model_path:
            print("Model path required for visualization")
            return
        
        print("Visualizing attention patterns...")
        # Load model and run visualization
        print("Visualization mode requires trained model - see README for details")

if __name__ == "__main__":
    main()
