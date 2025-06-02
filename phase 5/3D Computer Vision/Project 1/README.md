# Project 1: Stereo Vision and Depth Estimation

## Overview
Implement stereo vision algorithms to compute depth maps from stereo image pairs. This project covers fundamental 3D vision concepts including camera calibration, stereo rectification, and disparity estimation.

## Learning Objectives
- Understand stereo vision principles and epipolar geometry
- Implement camera calibration for stereo systems
- Learn stereo rectification algorithms
- Master disparity estimation techniques
- Convert disparity to depth measurements
- Handle stereo matching challenges

## Key Concepts
- **Stereo Vision**: Using two cameras to perceive depth
- **Epipolar Geometry**: Geometric relationship between stereo views
- **Camera Calibration**: Determine intrinsic and extrinsic parameters
- **Stereo Rectification**: Align stereo images for easier matching
- **Disparity Map**: Pixel displacement between stereo views
- **Block Matching**: Local correspondence algorithm

## Implementation Features
- Complete stereo vision pipeline
- Camera calibration using checkerboard patterns
- Stereo rectification implementation
- Multiple disparity estimation algorithms (SAD, SSD, NCC)
- Semi-global matching (SGM) implementation
- Depth map visualization and analysis
- Real-time stereo processing

## Dataset
- KITTI stereo dataset for autonomous driving
- Middlebury stereo datasets for evaluation
- Custom stereo image pairs
- Synthetic datasets with ground truth

## Requirements
See `requirements.txt` for full dependencies.

## Usage
```bash
# Calibrate stereo camera system
python main.py --mode calibrate --left_images calib/left/ --right_images calib/right/

# Compute depth from stereo pair
python main.py --mode depth --left image_left.jpg --right image_right.jpg

# Evaluate on dataset
python main.py --mode evaluate --dataset kitti --subset training

# Real-time stereo processing
python main.py --mode realtime --left_camera 0 --right_camera 1
```

## Expected Results
- Accurate depth estimation for textured regions
- Smooth depth maps with preserved edges
- Real-time processing (>10 FPS) on modern hardware
- Competitive performance on Middlebury benchmark
