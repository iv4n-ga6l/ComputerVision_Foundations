# Project 4: Neural Style Transfer

## Overview
Implement Neural Style Transfer using pre-trained convolutional neural networks to combine the content of one image with the artistic style of another. This project demonstrates how to leverage feature representations from deep networks for artistic image generation.

## Learning Objectives
- Understand how CNNs capture content and style information
- Implement optimization-based style transfer
- Learn about Gram matrices for style representation
- Apply perceptual losses for image generation
- Explore fast neural style transfer approaches

## Key Concepts
- **Content Representation**: High-level feature maps that capture semantic content
- **Style Representation**: Gram matrices of feature maps capturing artistic style
- **Perceptual Loss**: Loss functions based on pre-trained network features
- **Optimization-based Transfer**: Iterative optimization of target image
- **Fast Style Transfer**: Feed-forward networks for real-time style transfer

## Project Structure
```
Project 4/
├── README.md
├── main.py                 # Main style transfer implementation
├── models/
│   ├── vgg_features.py     # VGG feature extractor
│   ├── fast_transfer.py    # Fast style transfer network
│   └── utils.py           # Model utilities
├── style_transfer/
│   ├── optimization.py     # Optimization-based style transfer
│   ├── losses.py          # Content and style losses
│   └── preprocessing.py   # Image preprocessing utilities
├── data/
│   ├── content/           # Content images
│   ├── style/             # Style images
│   └── output/            # Generated images
└── requirements.txt
```

## Key Features
1. **Multiple Transfer Methods**
   - Optimization-based style transfer (Gatys et al.)
   - Fast neural style transfer (Johnson et al.)
   - Arbitrary style transfer

2. **Advanced Features**
   - Color preservation options
   - Style strength control
   - Multi-scale style transfer
   - Real-time video style transfer

3. **Comprehensive Loss Functions**
   - Content loss using VGG features
   - Style loss using Gram matrices
   - Total variation loss for smoothness
   - Perceptual losses

4. **Interactive Interface**
   - Command-line interface
   - Batch processing capabilities
   - Parameter tuning options

## Usage
```bash
# Basic style transfer
python main.py --content content.jpg --style style.jpg --output result.jpg

# Fast style transfer
python main.py --method fast --content content.jpg --style style.jpg

# Optimization-based with custom parameters
python main.py --method optimization --content content.jpg --style style.jpg --iterations 1000 --style_weight 1000000

# Video style transfer
python main.py --input_video video.mp4 --style style.jpg --output_video styled_video.mp4

# Train fast style transfer model
python main.py --mode train --style style.jpg --dataset coco --epochs 2
```

## Implementation Details
- Uses VGG-19 pre-trained network for feature extraction
- Implements both optimization and feed-forward approaches
- Supports various style transfer techniques
- Includes comprehensive evaluation metrics
- Provides tools for batch processing and video transfer
