# Project 4: Few-Shot Learning

## Overview
This project implements state-of-the-art few-shot learning methods for computer vision tasks. Few-shot learning enables models to learn new concepts from only a few examples, mimicking human-like learning capabilities.

## Methods Implemented

### 1. Prototypical Networks
- Distance-based classification with prototype learning
- Episodic training with support and query sets
- Euclidean distance in embedding space
- Meta-learning for rapid adaptation

### 2. Model-Agnostic Meta-Learning (MAML)
- Gradient-based meta-learning algorithm
- Fast adaptation to new tasks with few gradient steps
- Bi-level optimization for meta-parameters
- First and second-order MAML implementations

### 3. Relation Networks
- Learnable similarity metrics for few-shot classification
- Deep network for computing relations between samples
- End-to-end learning of embeddings and relations
- Flexible architecture for different modalities

### 4. Matching Networks
- Attention-based matching for one-shot learning
- Differentiable nearest neighbor classification
- Full Context Embeddings (FCE) for support sets
- Attention LSTM for adaptive matching

### 5. Meta-Learning with Memory-Augmented Networks
- External memory for storing few-shot examples
- Neural Turing Machine-inspired architectures
- Rapid binding of new information
- Episodic memory for meta-learning

## Features

### Episode Sampling
- **N-way K-shot**: Support sets with N classes, K examples each
- **Query Set**: Test examples for evaluation within episodes
- **Balanced Sampling**: Equal representation across classes
- **Meta-Batch Processing**: Multiple episodes per training batch

### Meta-Learning Framework
- **Episodic Training**: Task-level optimization
- **Gradient-Based Methods**: MAML, Reptile, first-order approximations
- **Metric Learning**: Learning distance functions for similarity
- **Memory-Augmented**: External memory for rapid adaptation

### Evaluation Protocols
- **Standard Benchmarks**: Omniglot, miniImageNet, tieredImageNet, CIFAR-FS
- **Cross-Domain**: Evaluation across different domains
- **Few-Shot Detection**: Object detection with few examples
- **Continual Learning**: Sequential task learning without forgetting

### Data Augmentation
- **Episode-Level**: Consistent augmentations within episodes
- **Support-Query**: Different augmentations for support/query
- **Meta-Augmentation**: Learning augmentation strategies
- **Domain Randomization**: Robust representation learning

## Requirements
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.5.4
- numpy >= 1.21.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- tensorboard >= 2.7.0
- wandb >= 0.12.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- Pillow >= 8.3.0
- higher >= 0.2.1
- learn2learn >= 0.1.7
- pytorch-lightning >= 1.5.0

## Usage

### Prototypical Networks Training
```python
from few_shot import PrototypicalTrainer

# Initialize trainer
trainer = PrototypicalTrainer(
    backbone='conv4',
    n_way=5,
    k_shot=1,
    query_shots=15,
    embedding_dim=64
)

# Train on meta-learning dataset
trainer.train(
    dataset='omniglot',
    meta_epochs=100,
    episodes_per_epoch=1000
)

# Evaluate on test set
accuracy = trainer.evaluate(test_episodes=600)
```

### MAML Training
```python
from few_shot import MAMLTrainer

# Initialize MAML trainer
trainer = MAMLTrainer(
    backbone='conv4',
    n_way=5,
    k_shot=5,
    inner_lr=0.01,
    meta_lr=0.001,
    inner_steps=5
)

# Meta-training
trainer.train(
    dataset='miniImageNet',
    meta_epochs=60000,
    first_order=False
)
```

### Cross-Domain Evaluation
```python
from few_shot import CrossDomainEvaluator

# Evaluate across domains
evaluator = CrossDomainEvaluator(
    model_path='checkpoints/proto_net.pth',
    method='prototypical'
)

results = evaluator.evaluate_cross_domain(
    source_domain='miniImageNet',
    target_domains=['CUB', 'Cars', 'Places', 'Plantae']
)
```

### Few-Shot Detection
```python
from few_shot import FewShotDetector

# Few-shot object detection
detector = FewShotDetector(
    backbone='resnet50',
    method='meta_rcnn'
)

detector.train(
    dataset='COCO',
    n_way=5,
    k_shot=10,
    episodes=10000
)
```

## Project Structure
```
Project 4/
├── main.py                    # Main few-shot learning implementation
├── methods/
│   ├── prototypical.py       # Prototypical Networks
│   ├── maml.py               # Model-Agnostic Meta-Learning
│   ├── relation.py           # Relation Networks
│   ├── matching.py           # Matching Networks
│   └── memory_augmented.py   # Memory-Augmented Networks
├── backbones/
│   ├── conv_nets.py          # Convolutional backbones
│   ├── resnet.py             # ResNet backbones
│   ├── vit.py                # Vision Transformer backbones
│   └── custom.py             # Custom architectures
├── datasets/
│   ├── omniglot.py           # Omniglot dataset
│   ├── mini_imagenet.py      # miniImageNet dataset
│   ├── tiered_imagenet.py    # tieredImageNet dataset
│   ├── cifar_fs.py           # CIFAR-FS dataset
│   └── cross_domain.py       # Cross-domain datasets
├── samplers/
│   ├── episode_sampler.py    # Episode sampling strategies
│   ├── balanced_sampler.py   # Balanced class sampling
│   └── meta_sampler.py       # Meta-learning samplers
├── trainers/
│   ├── meta_trainer.py       # Base meta-learning trainer
│   ├── episodic_trainer.py   # Episodic training loop
│   └── continual_trainer.py  # Continual learning trainer
├── evaluation/
│   ├── few_shot_eval.py      # Few-shot evaluation
│   ├── cross_domain_eval.py  # Cross-domain evaluation
│   └── detection_eval.py     # Few-shot detection eval
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Plotting and visualization
│   ├── augmentation.py       # Data augmentation
│   └── logging.py            # Experiment tracking
└── configs/
    ├── prototypical_config.yaml
    ├── maml_config.yaml
    ├── relation_config.yaml
    └── matching_config.yaml
```

## Key Concepts

### Few-Shot Learning Paradigms
1. **Metric Learning**: Learning similarity functions for classification
2. **Meta-Learning**: Learning to learn from few examples
3. **Transfer Learning**: Adapting pre-trained models to new tasks
4. **Data Augmentation**: Generating more training examples

### Episode Structure
- **Support Set**: Few labeled examples for each class
- **Query Set**: Unlabeled examples for testing within episode
- **N-way K-shot**: N classes with K examples each
- **Meta-Batch**: Multiple episodes processed together

### Optimization Strategies
1. **Gradient-Based**: MAML, Reptile, first-order methods
2. **Distance-Based**: Prototypical, matching, relation networks
3. **Memory-Based**: External memory, episodic memory
4. **Ensemble Methods**: Multiple models and predictions

## Evaluation Protocols

### Standard Benchmarks
- **Omniglot**: 1623 characters, 20 examples each
- **miniImageNet**: 100 classes, 600 images per class
- **tieredImageNet**: Hierarchical subset of ImageNet
- **CIFAR-FS**: Few-shot version of CIFAR-100

### Evaluation Metrics
- **Accuracy**: Classification accuracy on query sets
- **Confidence Intervals**: Statistical significance testing
- **Learning Curves**: Adaptation speed analysis
- **Computational Cost**: Training and inference efficiency

### Cross-Domain Evaluation
- **Domain Gap**: Performance drop across domains
- **Adaptation Speed**: Few-shot adaptation rate
- **Robustness**: Performance under domain shift
- **Transfer Quality**: Knowledge transfer effectiveness

## Advanced Techniques

### Meta-Optimization
- **Gradient-Based Meta-Learning**: MAML, Reptile, iMAML
- **Black-Box Meta-Learning**: Recurrent models, memory networks
- **Hybrid Approaches**: Combining gradient and metric learning
- **Continual Meta-Learning**: Sequential task learning

### Architecture Design
- **Embedding Networks**: Feature extraction for similarity
- **Attention Mechanisms**: Adaptive feature weighting
- **Memory Modules**: External and internal memory
- **Modular Networks**: Compositional architectures

### Training Strategies
- **Curriculum Learning**: Progressive difficulty increase
- **Data Augmentation**: Episode-level augmentations
- **Regularization**: Preventing overfitting in meta-learning
- **Multi-Task Learning**: Joint training on multiple tasks

## Applications

### Computer Vision
- **Image Classification**: Novel class recognition
- **Object Detection**: Few-shot object detection
- **Semantic Segmentation**: Pixel-level few-shot learning
- **Medical Imaging**: Diagnosis with limited data

### Real-World Scenarios
- **Robotics**: Quick adaptation to new environments
- **Autonomous Vehicles**: Recognizing rare objects
- **Manufacturing**: Quality control with few defect examples
- **Agriculture**: Crop disease detection with limited samples

### Research Directions
- **Unsupervised Few-Shot**: Learning without labels
- **Multimodal Few-Shot**: Vision-language learning
- **Federated Few-Shot**: Distributed learning scenarios
- **Continual Few-Shot**: Lifelong learning capabilities

## Performance Benchmarks

### Omniglot (5-way 1-shot)
- Prototypical Networks: ~98.8%
- MAML: ~98.7%
- Matching Networks: ~98.1%
- Relation Networks: ~99.6%

### miniImageNet (5-way 1-shot)
- Prototypical Networks: ~49.4%
- MAML: ~48.7%
- Matching Networks: ~46.6%
- Relation Networks: ~50.4%

### miniImageNet (5-way 5-shot)
- Prototypical Networks: ~68.2%
- MAML: ~63.1%
- Matching Networks: ~60.0%
- Relation Networks: ~65.3%

## Tips for Success

### Data Preparation
1. **Quality over Quantity**: Ensure high-quality few-shot examples
2. **Diverse Support Sets**: Include varied examples per class
3. **Balanced Episodes**: Equal representation across classes
4. **Augmentation Strategy**: Consistent within episodes

### Model Selection
1. **Architecture Choice**: Match backbone to data complexity
2. **Embedding Dimension**: Balance capacity and generalization
3. **Method Selection**: Consider task requirements and constraints
4. **Hyperparameter Tuning**: Careful optimization of meta-parameters

### Training Best Practices
1. **Episode Design**: Realistic episode structure for target task
2. **Validation Protocol**: Separate meta-validation set
3. **Early Stopping**: Prevent overfitting in meta-learning
4. **Multiple Runs**: Average results over several random seeds

## References
- Prototypical Networks: "Prototypical Networks for Few-shot Learning"
- MAML: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
- Relation Networks: "Learning to Compare: Relation Network for Few-Shot Learning"
- Matching Networks: "Matching Networks for One Shot Learning"
- Memory-Augmented: "Meta-Learning with Memory-Augmented Neural Networks"
