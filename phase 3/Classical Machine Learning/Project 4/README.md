# Project 4: K-Means Image Segmentation

Implement unsupervised image segmentation using K-Means clustering algorithm.

## Objective
Segment images into meaningful regions using color and spatial information through clustering, demonstrating unsupervised learning in computer vision.

## Key Concepts

### K-Means Clustering:
- Partitions data into k clusters based on feature similarity
- Minimizes within-cluster sum of squared distances
- Iteratively updates cluster centroids
- Commonly used for color quantization and segmentation

### Image Segmentation Applications:
- Object isolation and background removal
- Medical image analysis
- Satellite image processing
- Image compression through color quantization

## Features Implemented
- K-Means clustering with configurable k values
- Color-based and spatial-aware segmentation
- Multiple color spaces (RGB, HSV, LAB)
- Elbow method for optimal k selection
- Interactive segmentation with different algorithms
- Performance comparison between methods

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Expected Results
- Clear segmentation of image regions
- Optimal k value determination
- Comparison of different color spaces
- Real-time interactive segmentation

## Applications
- Medical imaging (tumor detection)
- Remote sensing (land cover classification)
- Quality control in manufacturing
- Content-based image retrieval
