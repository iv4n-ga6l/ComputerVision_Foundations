# Project 2: 3D Reconstruction from Multiple Views

## Overview
Implement 3D reconstruction algorithms to create 3D models from multiple 2D images. This project covers Structure from Motion (SfM), Multi-View Stereo (MVS), and photogrammetry techniques for reconstructing 3D scenes from image sequences.

## Learning Objectives
- Understand fundamental principles of 3D reconstruction
- Implement Structure from Motion (SfM) pipeline
- Learn Multi-View Stereo (MVS) algorithms
- Apply bundle adjustment for optimization
- Create dense 3D point clouds and meshes

## Key Concepts
- **Structure from Motion (SfM)**: Estimating camera poses and 3D structure simultaneously
- **Bundle Adjustment**: Non-linear optimization of camera parameters and 3D points
- **Multi-View Stereo (MVS)**: Dense reconstruction using multiple calibrated views
- **Essential/Fundamental Matrix**: Geometric relationships between views
- **Triangulation**: Computing 3D points from corresponding 2D features

## Project Structure
```
Project 2/
├── README.md
├── main.py                 # Main reconstruction pipeline
├── reconstruction/
│   ├── sfm.py             # Structure from Motion implementation
│   ├── mvs.py             # Multi-View Stereo algorithms
│   ├── bundle_adjustment.py # Bundle adjustment optimization
│   └── triangulation.py   # Point triangulation methods
├── utils/
│   ├── camera.py          # Camera models and calibration
│   ├── feature_matching.py # Advanced feature matching
│   ├── visualization.py   # 3D visualization tools
│   └── io_utils.py        # I/O utilities for 3D data
├── data/
│   ├── images/            # Input image sequences
│   ├── calibration/       # Camera calibration data
│   └── output/            # Reconstructed models
└── requirements.txt
```

## Key Features
1. **Complete SfM Pipeline**
   - Feature detection and matching
   - Camera pose estimation
   - Bundle adjustment optimization
   - Dense reconstruction

2. **Multiple Reconstruction Methods**
   - Incremental SfM
   - Global SfM
   - Hierarchical reconstruction

3. **Advanced Processing**
   - Automatic camera calibration
   - Loop closure detection
   - Mesh generation and texturing
   - Quality assessment metrics

4. **Visualization Tools**
   - Interactive 3D point cloud viewer
   - Camera trajectory visualization
   - Reconstruction quality analysis

## Usage
```bash
# Full reconstruction pipeline
python main.py --input_dir images/ --output_dir reconstruction/ --method sfm

# Structure from Motion only
python main.py --mode sfm --images images/*.jpg --output reconstruction.ply

# Multi-View Stereo densification
python main.py --mode mvs --sparse_reconstruction sparse.ply --images images/ --output dense.ply

# Bundle adjustment optimization
python main.py --mode bundle_adjust --input reconstruction.ply --output optimized.ply

# Mesh generation
python main.py --mode mesh --pointcloud dense.ply --output mesh.obj

# Evaluation with ground truth
python main.py --mode evaluate --reconstruction result.ply --ground_truth gt.ply
```

## Implementation Details
- Uses robust feature matching with geometric verification
- Implements various triangulation methods (DLT, optimal)
- Includes comprehensive bundle adjustment with different parameterizations
- Supports multiple camera models (pinhole, fisheye, etc.)
- Provides extensive visualization and analysis tools
