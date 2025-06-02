# Project 1: Vision Transformers (ViTs) Implementation

## Overview
This project provides a comprehensive implementation of Vision Transformers (ViTs), including various architectures, training strategies, and applications. It covers the complete spectrum from basic ViT implementation to advanced variants and practical deployment scenarios.

## Features

### Core ViT Architectures
- **Original ViT**: "An Image is Worth 16x16 Words" implementation
- **DeiT**: Data-efficient Image Transformers with distillation
- **Swin Transformer**: Hierarchical vision transformer with shifted windows
- **PVT**: Pyramid Vision Transformer for dense prediction tasks
- **CaiT**: Class-Attention in Image Transformers
- **CrossViT**: Cross-attention multi-scale vision transformer

### Advanced Features
- **Patch Embedding**: Multiple patch embedding strategies (linear, convolutional, overlapping)
- **Position Encoding**: Learnable, sinusoidal, and relative position encodings
- **Attention Mechanisms**: Multi-head self-attention, cross-attention, local attention
- **Training Strategies**: Progressive resizing, mixup, cutmix, label smoothing
- **Optimization**: AdamW, warmup scheduling, gradient clipping
- **Regularization**: Dropout, DropPath, LayerScale

### Applications & Tasks
- **Image Classification**: Fine-tuning on custom datasets
- **Object Detection**: ViT as backbone for detection frameworks
- **Semantic Segmentation**: Dense prediction with transformer models
- **Transfer Learning**: Pre-trained model adaptation and fine-tuning
- **Multi-modal Learning**: Vision-language models with CLIP-style training
- **Self-supervised Learning**: MAE (Masked Autoencoder) implementation

### Visualization & Analysis
- **Attention Visualization**: Attention map generation and analysis
- **Feature Visualization**: Intermediate layer feature visualization
- **Model Interpretation**: Class activation maps for ViTs
- **Patch Importance**: Analysis of patch-level contributions
- **Attention Rollout**: Recursive attention pattern analysis

## Architecture Details

### Vision Transformer (ViT) Components
```
Input Image (224x224x3)
    ↓
Patch Embedding (16x16 patches → 768D vectors)
    ↓
Position Embedding Addition
    ↓
[CLS] Token Concatenation
    ↓
Transformer Encoder Blocks (12 layers)
├── Multi-Head Self-Attention
├── Layer Normalization
├── MLP (Feed-Forward Network)
└── Residual Connections
    ↓
[CLS] Token → Classification Head
    ↓
Class Probabilities
```

### Swin Transformer Architecture
```
Input Image
    ↓
Patch Partition (4x4 patches)
    ↓
Stage 1: Swin Transformer Block
├── Window-based Multi-Head Self-Attention
├── Shifted Window Multi-Head Self-Attention
    ↓
Stage 2: Patch Merging + Swin Blocks
    ↓
Stage 3: Patch Merging + Swin Blocks
    ↓
Stage 4: Patch Merging + Swin Blocks
    ↓
Global Average Pooling → Classification
```

## Requirements
- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+
- timm 0.6+ (for pre-trained models)
- transformers 4.20+ (for HuggingFace models)
- einops 0.4+ (for tensor operations)
- matplotlib, seaborn (for visualization)

## Usage

### Basic ViT Training
```python
from main import ViTTrainer, ViTConfig

# Configure model
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072
)

# Initialize trainer
trainer = ViTTrainer(config)

# Train model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-3
)
```

### Transfer Learning Example
```python
from main import ViTTransferLearning

# Load pre-trained ViT
transfer = ViTTransferLearning(
    pretrained_model='vit_base_patch16_224',
    num_classes=10,  # Custom dataset classes
    freeze_backbone=True
)

# Fine-tune on custom dataset
transfer.fine_tune(
    dataset=custom_dataset,
    epochs=20,
    learning_rate=1e-4
)
```

### Attention Visualization
```python
from main import AttentionVisualizer

# Initialize visualizer
visualizer = AttentionVisualizer(model)

# Generate attention maps
attention_maps = visualizer.generate_attention_maps(
    image=input_image,
    layers=[6, 9, 12],  # Which layers to visualize
    heads='mean'  # Average across attention heads
)

# Plot results
visualizer.plot_attention_maps(attention_maps)
```

## Project Structure
```
Project 1/
├── main.py                    # Main implementation and training
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── vit.py                # Original Vision Transformer
│   ├── deit.py               # Data-efficient Image Transformers
│   ├── swin.py               # Swin Transformer
│   ├── pvt.py                # Pyramid Vision Transformer
│   ├── cait.py               # Class-Attention in Image Transformers
│   └── crossvit.py           # Cross-attention ViT
├── components/                # Model components
│   ├── __init__.py
│   ├── attention.py          # Attention mechanisms
│   ├── embeddings.py         # Patch and position embeddings
│   ├── layers.py             # Transformer layers
│   └── utils.py              # Utility functions
├── training/                  # Training utilities
│   ├── __init__.py
│   ├── trainer.py            # Training loop implementation
│   ├── optimizers.py         # Custom optimizers and schedulers
│   ├── losses.py             # Loss functions and regularization
│   └── augmentation.py       # Data augmentation strategies
├── evaluation/                # Evaluation and analysis
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Attention and feature visualization
│   ├── interpretation.py     # Model interpretation tools
│   └── benchmarks.py         # Performance benchmarking
├── datasets/                  # Dataset utilities
│   ├── __init__.py
│   ├── imagenet.py           # ImageNet data loading
│   ├── cifar.py              # CIFAR dataset utilities
│   └── custom.py             # Custom dataset handling
├── configs/                   # Configuration files
│   ├── vit_base.yaml         # Base ViT configuration
│   ├── vit_large.yaml        # Large ViT configuration
│   ├── swin_tiny.yaml        # Swin Transformer configurations
│   └── training.yaml         # Training hyperparameters
├── pretrained/                # Pre-trained model utilities
│   ├── __init__.py
│   ├── download.py           # Model download utilities
│   └── convert.py            # Model format conversion
└── examples/                  # Example usage scripts
    ├── classification.py      # Image classification example
    ├── detection.py           # Object detection with ViT backbone
    ├── segmentation.py        # Semantic segmentation example
    ├── attention_analysis.py  # Attention pattern analysis
    └── transfer_learning.py   # Transfer learning example
```

## Key Components

### 1. Vision Transformer (ViT)
- **Patch Embedding**: Converts image patches to token embeddings
- **Transformer Encoder**: Standard transformer architecture
- **Classification Head**: MLP for final predictions
- **Position Encoding**: Learnable positional embeddings

### 2. Data-efficient Image Transformers (DeiT)
- **Knowledge Distillation**: Teacher-student training framework
- **Distillation Token**: Additional token for distillation loss
- **Hard/Soft Distillation**: Different distillation strategies
- **Efficient Training**: Reduced training time and data requirements

### 3. Swin Transformer
- **Hierarchical Architecture**: Multi-scale feature representation
- **Shifted Window Attention**: Efficient attention computation
- **Patch Merging**: Downsampling between stages
- **Relative Position Bias**: Improved position encoding

### 4. Attention Mechanisms
- **Multi-Head Self-Attention**: Core attention computation
- **Window-based Attention**: Local attention for efficiency
- **Cross-Attention**: Attention between different modalities
- **Sparse Attention**: Attention pattern sparsification

### 5. Training Strategies
- **Progressive Resizing**: Gradual image size increase during training
- **Mixup/CutMix**: Data augmentation techniques
- **Label Smoothing**: Regularization technique
- **Warmup Scheduling**: Learning rate warmup
- **Gradient Clipping**: Gradient norm clipping

## Performance Benchmarks

### ImageNet-1K Classification
| Model | Parameters | Top-1 Acc | FLOPs |
|-------|------------|-----------|--------|
| ViT-B/16 | 86M | 81.8% | 17.6G |
| ViT-L/16 | 307M | 82.6% | 61.6G |
| Swin-T | 29M | 81.3% | 4.5G |
| Swin-B | 88M | 83.5% | 15.4G |
| DeiT-B | 86M | 81.8% | 17.6G |

### Training Time Comparison
- **ViT-B/16**: ~3 days on 8xV100 GPUs
- **Swin-T**: ~2 days on 8xV100 GPUs
- **DeiT-B**: ~1.5 days on 8xV100 GPUs (with distillation)

## Advanced Features

### 1. Masked Autoencoder (MAE)
- **Self-supervised Pre-training**: Learn representations without labels
- **High Masking Ratio**: 75% of patches masked during training
- **Asymmetric Encoder-Decoder**: Efficient architecture design
- **Reconstruction Loss**: Pixel-level reconstruction objective

### 2. Multi-modal Learning
- **CLIP-style Training**: Vision-language contrastive learning
- **Cross-modal Attention**: Attention between vision and text
- **Zero-shot Classification**: Classification without task-specific training
- **Image-Text Retrieval**: Bidirectional retrieval tasks

### 3. Object Detection Integration
- **DETR**: Detection Transformer with ViT backbone
- **ViT-RCNN**: ViT integration with R-CNN frameworks
- **Patch-based Detection**: Native transformer object detection
- **Multi-scale Features**: Hierarchical feature extraction

### 4. Semantic Segmentation
- **SETR**: Segmentation Transformer implementation
- **Dense Prediction**: Pixel-level classification
- **Multi-scale Segmentation**: Hierarchical segmentation
- **Attention Upsampling**: Attention-based feature upsampling

## Research Applications

### 1. Attention Pattern Analysis
- **Attention Distance**: How far attention reaches across patches
- **Head Specialization**: Different attention heads for different features
- **Layer-wise Evolution**: How attention patterns evolve through layers
- **Task-specific Attention**: Attention patterns for different tasks

### 2. Model Scaling Studies
- **Depth vs Width**: Impact of model depth and width on performance
- **Patch Size Analysis**: Effect of different patch sizes
- **Position Encoding**: Comparison of position encoding methods
- **Training Data Scaling**: Performance vs training data size

### 3. Transfer Learning Analysis
- **Feature Transferability**: Which features transfer across domains
- **Fine-tuning Strategies**: Different approaches to adaptation
- **Domain Gap Analysis**: Measuring domain differences
- **Few-shot Learning**: Performance with limited target data

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Quick Start - Image Classification
```python
from main import ViTClassifier

# Load pre-trained model
model = ViTClassifier.from_pretrained('vit_base_patch16_224')

# Classify image
predictions = model.predict('path/to/image.jpg')
print(f"Predicted class: {predictions['class']}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

### 3. Training Custom Model
```bash
python main.py train \
    --model vit_base \
    --dataset imagenet \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 1e-3
```

### 4. Attention Visualization
```bash
python examples/attention_analysis.py \
    --model_path checkpoints/vit_base.pth \
    --image_path examples/sample.jpg \
    --output_dir visualizations/
```

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Use mixed precision training (AMP)
3. **Poor Convergence**: Adjust learning rate and warmup schedule
4. **Attention Collapse**: Use proper weight initialization and layer normalization

### Performance Tips
- **Use Data Parallel Training**: Distribute across multiple GPUs
- **Enable Mixed Precision**: Use automatic mixed precision for faster training
- **Optimize Data Loading**: Use multiple workers and pin memory
- **Profile Training**: Identify and optimize bottlenecks

## Future Research Directions
- **Efficient Attention**: Linear attention mechanisms
- **Dynamic Models**: Adaptive computation based on input complexity
- **Multimodal Transformers**: Joint vision-language understanding
- **Neural Architecture Search**: Automated ViT design
- **Continual Learning**: Lifelong learning with transformers
