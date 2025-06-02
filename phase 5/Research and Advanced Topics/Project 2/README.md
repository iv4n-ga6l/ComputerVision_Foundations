# Project 2: Self-Supervised Learning

## Overview
This project implements state-of-the-art self-supervised learning methods for visual representation learning. Self-supervised learning leverages unlabeled data to learn meaningful representations without human annotations.

## Methods Implemented

### 1. SimCLR (Simple Contrastive Learning of Visual Representations)
- Contrastive learning framework with data augmentation
- Temperature-scaled cosine similarity
- Momentum-based training with projection heads

### 2. MoCo (Momentum Contrast)
- Momentum-updated encoder for consistent representations
- Dynamic dictionary with queue mechanism
- Contrastive loss with negative sampling

### 3. BYOL (Bootstrap Your Own Latent)
- Self-supervised learning without negative samples
- Target network with exponential moving average
- Predictor and projection networks

### 4. SwAV (Swapping Assignments between Views)
- Cluster-based contrastive learning
- Online clustering with Sinkhorn-Knopp algorithm
- Multi-crop training strategy

### 5. MAE (Masked Autoencoder)
- Vision Transformer-based masked modeling
- High masking ratio (75%) for efficiency
- Asymmetric encoder-decoder architecture

## Features

### Data Augmentation Pipeline
- **Color Jittering**: Brightness, contrast, saturation, hue variations
- **Geometric Transforms**: Random crops, flips, rotations
- **Advanced Augmentations**: Gaussian blur, random grayscale
- **Multi-crop Strategy**: Different scales and aspect ratios

### Training Infrastructure
- **Mixed Precision Training**: Automatic mixed precision for efficiency
- **Gradient Clipping**: Stable training with large batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Model state saving and resuming

### Evaluation Framework
- **Linear Probing**: Evaluate learned representations
- **Fine-tuning**: End-to-end task-specific training
- **Transfer Learning**: Cross-dataset evaluation
- **Representation Quality**: t-SNE visualization, nearest neighbors

### Visualization Tools
- **Loss Curves**: Training progress monitoring
- **Representation Space**: t-SNE and UMAP embeddings
- **Attention Maps**: For transformer-based methods
- **Augmentation Preview**: Data transformation visualization

## Requirements
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.5.4
- scikit-learn >= 1.0.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- tensorboard >= 2.7.0
- wandb >= 0.12.0
- opencv-python >= 4.5.0
- pillow >= 8.3.0
- numpy >= 1.21.0
- tqdm >= 4.62.0

## Usage

### Basic Training
```python
from self_supervised import SimCLRTrainer, MoCoTrainer, BYOLTrainer

# SimCLR training
trainer = SimCLRTrainer(
    backbone='resnet50',
    projection_dim=128,
    temperature=0.1,
    batch_size=256
)
trainer.train(dataset_path='path/to/unlabeled/data', epochs=100)

# MoCo training
trainer = MoCoTrainer(
    backbone='resnet50',
    momentum=0.999,
    queue_size=65536
)
trainer.train(dataset_path='path/to/unlabeled/data', epochs=200)
```

### Linear Evaluation
```python
from self_supervised import LinearEvaluator

evaluator = LinearEvaluator(
    model_path='checkpoints/simclr_epoch_100.pth',
    backbone='resnet50'
)
accuracy = evaluator.evaluate(
    train_dataset='path/to/labeled/train',
    test_dataset='path/to/labeled/test'
)
```

### Transfer Learning
```python
from self_supervised import TransferLearner

transfer = TransferLearner(
    pretrained_path='checkpoints/byol_epoch_300.pth',
    backbone='resnet50',
    num_classes=10
)
transfer.fine_tune(
    train_dataset='path/to/target/train',
    val_dataset='path/to/target/val',
    epochs=50
)
```

## Project Structure
```
Project 2/
├── main.py                 # Main training and evaluation script
├── models/
│   ├── simclr.py          # SimCLR implementation
│   ├── moco.py            # MoCo implementation
│   ├── byol.py            # BYOL implementation
│   ├── swav.py            # SwAV implementation
│   └── mae.py             # MAE implementation
├── data/
│   ├── augmentations.py   # Data augmentation pipeline
│   ├── datasets.py        # Custom dataset classes
│   └── transforms.py      # Transform utilities
├── training/
│   ├── trainers.py        # Training loops for each method
│   ├── losses.py          # Contrastive and reconstruction losses
│   └── optimizers.py      # Optimizer configurations
├── evaluation/
│   ├── linear_eval.py     # Linear probing evaluation
│   ├── transfer.py        # Transfer learning utilities
│   └── metrics.py         # Evaluation metrics
├── utils/
│   ├── visualization.py   # Plotting and visualization
│   ├── checkpoints.py     # Model saving/loading
│   └── logging.py         # Experiment tracking
└── configs/
    ├── simclr_config.yaml
    ├── moco_config.yaml
    ├── byol_config.yaml
    ├── swav_config.yaml
    └── mae_config.yaml
```

## Key Insights

### Self-Supervised Learning Principles
1. **Pretext Tasks**: Designing tasks that require understanding visual structure
2. **Data Efficiency**: Learning from unlimited unlabeled data
3. **Representation Quality**: Features that transfer well to downstream tasks
4. **Augmentation Importance**: Strong augmentations are crucial for contrastive methods

### Method Comparisons
- **SimCLR**: Simple and effective, requires large batch sizes
- **MoCo**: Memory-efficient with momentum updates
- **BYOL**: No negative samples needed, stable training
- **SwAV**: Cluster-based approach, good for large-scale data
- **MAE**: Efficient masked modeling for Vision Transformers

### Training Tips
1. **Batch Size**: Larger batches generally improve contrastive learning
2. **Temperature**: Critical hyperparameter for contrastive losses
3. **Augmentation Strength**: Balance between too weak and too strong
4. **Training Duration**: Self-supervised methods often need longer training
5. **Architecture Choice**: Backbone selection affects representation quality

## Evaluation Metrics

### Representation Quality
- **Linear Probing Accuracy**: Freeze features, train linear classifier
- **Fine-tuning Accuracy**: End-to-end training on downstream task
- **Transfer Learning**: Performance across different datasets
- **Clustering Metrics**: Silhouette score, adjusted rand index

### Training Metrics
- **Contrastive Loss**: For contrastive methods
- **Reconstruction Loss**: For generative methods
- **Alignment**: Feature alignment between augmented views
- **Uniformity**: Feature distribution uniformity

## Applications

### Computer Vision Tasks
- **Image Classification**: Pre-training for supervised learning
- **Object Detection**: Backbone initialization
- **Semantic Segmentation**: Feature extraction
- **Medical Imaging**: Domain adaptation with limited labels

### Research Directions
- **Multi-modal Learning**: Vision-language pre-training
- **Video Understanding**: Temporal self-supervision
- **3D Vision**: Self-supervised 3D representation learning
- **Robustness**: Learning robust visual representations

## References
- SimCLR: "A Simple Framework for Contrastive Learning of Visual Representations"
- MoCo: "Momentum Contrast for Unsupervised Visual Representation Learning"
- BYOL: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
- SwAV: "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"
- MAE: "Masked Autoencoders Are Scalable Vision Learners"
