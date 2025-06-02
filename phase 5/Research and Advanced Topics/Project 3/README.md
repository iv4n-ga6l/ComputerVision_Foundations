# Project 3: Neural Architecture Search (NAS)

## Overview
This project implements Neural Architecture Search (NAS) techniques to automatically discover optimal neural network architectures for computer vision tasks. NAS automates the design process of neural networks, finding architectures that achieve better performance than manually designed ones.

## Methods Implemented

### 1. Differentiable Architecture Search (DARTS)
- Continuous relaxation of architecture search space
- Gradient-based optimization of architecture parameters
- Memory-efficient one-shot training approach
- Mixed operations with architecture weights

### 2. Progressive Neural Architecture Search (PNAS)
- Progressive search strategy with predictor networks
- Efficiency through early stopping of poor architectures
- Cell-based search space design
- Performance prediction models

### 3. Efficient Neural Architecture Search (ENAS)
- Parameter sharing across architectures
- Controller network for architecture generation
- Reinforcement learning-based search strategy
- Weight sharing for computational efficiency

### 4. PC-DARTS (Partially Connected DARTS)
- Memory-efficient variant of DARTS
- Partial channel connections to reduce memory
- Edge normalization for stable training
- Faster convergence with reduced computational cost

### 5. ProxylessNAS
- Direct search on target hardware platforms
- Latency-aware architecture optimization
- Gradient-based search without proxies
- Mobile-optimized architectures

## Features

### Search Space Design
- **Macro Search Space**: Overall network structure and connections
- **Micro Search Space**: Individual cell/block architectures
- **Operation Pool**: Convolutions, pooling, skip connections, zero operations
- **Topology Search**: Connection patterns and layer arrangements

### Search Strategies
- **Gradient-based**: DARTS, PC-DARTS, ProxylessNAS
- **Evolutionary**: Genetic algorithms for architecture evolution
- **Reinforcement Learning**: Controller networks for architecture generation
- **Bayesian Optimization**: Gaussian processes for efficient search

### Hardware-Aware Search
- **Latency Constraints**: Mobile and edge device optimization
- **Energy Efficiency**: Power consumption considerations
- **Memory Usage**: RAM and parameter count constraints
- **FLOPs Optimization**: Computational complexity reduction

### Multi-Objective Optimization
- **Accuracy vs Efficiency**: Pareto-optimal architectures
- **Performance Trade-offs**: Speed, accuracy, size balance
- **Constraint Handling**: Hard and soft constraint satisfaction
- **Objective Weighting**: Customizable optimization goals

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
- graphviz >= 0.17.0
- networkx >= 2.6.0
- optuna >= 2.10.0
- thop >= 0.1.0
- ptflops >= 0.6.0

## Usage

### DARTS Architecture Search
```python
from nas import DARTSSearcher

# Initialize DARTS searcher
searcher = DARTSSearcher(
    dataset='cifar10',
    search_space='darts_space',
    epochs=50,
    batch_size=64
)

# Search for optimal architecture
best_arch = searcher.search()

# Train discovered architecture
final_model = searcher.train_final_architecture(
    architecture=best_arch,
    epochs=600
)
```

### Progressive NAS
```python
from nas import PNASSearcher

# Progressive search with predictor
searcher = PNASSearcher(
    search_space='pnas_space',
    predictor_epochs=150,
    search_epochs=300
)

# Search with progressive strategy
best_arch = searcher.progressive_search()
```

### Hardware-Aware Search
```python
from nas import ProxylessNASSearcher

# Hardware-aware architecture search
searcher = ProxylessNASSearcher(
    target_platform='mobile',
    latency_constraint=100,  # ms
    accuracy_weight=0.7,
    latency_weight=0.3
)

# Search for mobile-optimized architecture
mobile_arch = searcher.search()
```

### Architecture Evaluation
```python
from nas import ArchitectureEvaluator

# Evaluate discovered architectures
evaluator = ArchitectureEvaluator()

results = evaluator.evaluate_architecture(
    architecture=best_arch,
    dataset='imagenet',
    metrics=['accuracy', 'latency', 'flops', 'params']
)
```

## Project Structure
```
Project 3/
├── main.py                    # Main NAS implementation
├── search_spaces/
│   ├── darts_space.py        # DARTS search space
│   ├── pnas_space.py         # PNAS search space
│   ├── mobile_space.py       # Mobile-optimized space
│   └── custom_space.py       # Custom search spaces
├── searchers/
│   ├── darts.py              # DARTS implementation
│   ├── pnas.py               # PNAS implementation
│   ├── enas.py               # ENAS implementation
│   ├── pc_darts.py           # PC-DARTS implementation
│   └── proxyless.py          # ProxylessNAS implementation
├── models/
│   ├── operations.py         # Primitive operations
│   ├── cells.py              # Architecture cells
│   ├── networks.py           # Complete networks
│   └── supernet.py           # Supernet for weight sharing
├── predictors/
│   ├── performance.py        # Performance predictors
│   ├── latency.py            # Latency predictors
│   └── accuracy.py           # Accuracy predictors
├── optimizers/
│   ├── controllers.py        # RL controllers
│   ├── evolutionary.py       # Evolutionary algorithms
│   └── bayesian.py           # Bayesian optimization
├── utils/
│   ├── visualization.py      # Architecture visualization
│   ├── metrics.py            # Evaluation metrics
│   ├── hardware.py           # Hardware profiling
│   └── encoding.py           # Architecture encoding
└── configs/
    ├── darts_config.yaml
    ├── pnas_config.yaml
    ├── enas_config.yaml
    └── mobile_config.yaml
```

## Key Concepts

### Search Space Design
1. **Operation Primitives**: Basic building blocks (conv, pool, skip)
2. **Cell Structure**: Repeatable computation units
3. **Connection Patterns**: How operations are connected
4. **Hierarchy Levels**: Macro and micro architecture components

### Search Strategies
- **One-shot**: Train supernet once, extract subnetworks
- **Progressive**: Gradually build complex architectures
- **Evolutionary**: Mutation and selection of architectures
- **Gradient-based**: Differentiable architecture parameters

### Efficiency Techniques
1. **Weight Sharing**: Reuse parameters across architectures
2. **Early Stopping**: Terminate poor architectures early
3. **Proxy Tasks**: Use smaller datasets/models for search
4. **Progressive Complexity**: Start simple, add complexity

## Architecture Representation

### Encoding Schemes
- **Adjacency Matrix**: Connection matrix representation
- **String Encoding**: Sequential operation descriptions
- **Graph Representation**: Nodes and edges structure
- **Hierarchical Encoding**: Multi-level architecture description

### Visualization Tools
- **Architecture Graphs**: Visual network topology
- **Performance Plots**: Accuracy vs efficiency scatter plots
- **Search Progress**: Evolution of discovered architectures
- **Operation Statistics**: Usage frequency of operations

## Evaluation Metrics

### Performance Metrics
- **Top-1/Top-5 Accuracy**: Classification performance
- **mIoU**: Semantic segmentation performance
- **mAP**: Object detection performance
- **Perplexity**: Language modeling performance

### Efficiency Metrics
- **Latency**: Inference time on target hardware
- **FLOPs**: Floating point operations count
- **Parameters**: Model size in parameters
- **Memory Usage**: Peak memory consumption
- **Energy**: Power consumption per inference

### Search Metrics
- **Search Time**: Time to find optimal architecture
- **Search Cost**: Computational resources used
- **Convergence**: Speed of search convergence
- **Diversity**: Variety of discovered architectures

## Applications

### Computer Vision Tasks
- **Image Classification**: CIFAR, ImageNet, custom datasets
- **Object Detection**: COCO, Open Images, custom objects
- **Semantic Segmentation**: Cityscapes, ADE20K, medical images
- **Face Recognition**: Large-scale face datasets

### Deployment Scenarios
- **Mobile Devices**: Smartphones and tablets optimization
- **Edge Computing**: IoT and embedded systems
- **Cloud Services**: Scalable inference systems
- **Specialized Hardware**: TPUs, FPGAs, neuromorphic chips

### Research Directions
- **Multi-task NAS**: Single architecture for multiple tasks
- **Few-shot NAS**: Architecture search with limited data
- **Federated NAS**: Distributed architecture search
- **Continual NAS**: Adapting architectures over time

## Advanced Features

### Predictor Networks
- **Performance Prediction**: Estimate accuracy without training
- **Latency Modeling**: Hardware-specific timing prediction
- **Resource Usage**: Memory and computation prediction
- **Transfer Learning**: Cross-domain performance prediction

### Multi-objective Optimization
- **Pareto Frontiers**: Trade-off analysis
- **Constraint Satisfaction**: Hard constraint handling
- **Preference Learning**: User preference incorporation
- **Robust Optimization**: Uncertainty-aware search

### Hardware Integration
- **Profiling Tools**: Automatic hardware characterization
- **Platform Adaptation**: Cross-platform optimization
- **Quantization-aware**: Include quantization in search
- **Pruning Integration**: Combined pruning and architecture search

## Tips for Success

### Search Space Design
1. **Start Simple**: Begin with well-understood operations
2. **Gradual Expansion**: Add complexity incrementally
3. **Domain Knowledge**: Include task-specific operations
4. **Avoid Bias**: Don't over-constrain the search space

### Training Strategies
1. **Proper Validation**: Use separate validation sets
2. **Regularization**: Prevent overfitting in search
3. **Stable Training**: Ensure reproducible results
4. **Resource Management**: Monitor memory and compute usage

### Evaluation Protocol
1. **Multiple Runs**: Average over several search runs
2. **Statistical Testing**: Verify significance of results
3. **Baseline Comparison**: Compare with manual designs
4. **Transfer Evaluation**: Test on different datasets

## References
- DARTS: "Differentiable Architecture Search"
- PNAS: "Progressive Neural Architecture Search"
- ENAS: "Efficient Neural Architecture Search via Parameter Sharing"
- PC-DARTS: "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search"
- ProxylessNAS: "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware"
