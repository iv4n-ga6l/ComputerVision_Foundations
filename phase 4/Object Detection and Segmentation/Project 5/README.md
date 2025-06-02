# Video Object Tracking

Implement various object tracking algorithms for video sequences.

## Project Overview

This project implements multiple object tracking algorithms including classical tracking methods and deep learning-based approaches. It supports single object tracking (SOT) and multiple object tracking (MOT).

## Features

1. **Classical Tracking Algorithms**
   - Kalman Filter tracking
   - Particle Filter tracking
   - Mean Shift tracking
   - CAMShift tracking
   - Lucas-Kanade optical flow

2. **Modern Tracking Algorithms**
   - SORT (Simple Online and Realtime Tracking)
   - DeepSORT with appearance features
   - ByteTrack
   - FairMOT
   - Tracking with detection

3. **Deep Learning Trackers**
   - Siamese networks for tracking
   - Correlation filters
   - LSTM-based tracking
   - Transformer-based tracking

4. **Evaluation Metrics**
   - MOTA (Multiple Object Tracking Accuracy)
   - MOTP (Multiple Object Tracking Precision)
   - IDF1 (ID F1 Score)
   - Track completeness and fragmentation

5. **Real-time Processing**
   - Webcam tracking
   - Video file processing
   - Batch processing
   - Performance optimization

## Architecture

### Tracking Pipeline
1. **Detection**: Object detection in each frame
2. **Association**: Data association between detections and tracks
3. **Tracking**: State estimation and track management
4. **Visualization**: Track visualization and analysis

### Multi-Object Tracking
- Track initialization and termination
- Identity preservation
- Occlusion handling
- Re-identification features

## Usage

### Single Object Tracking
```bash
# Track single object
python main.py --mode sot --video_path video.mp4 --tracker kcf

# Interactive tracking (select object)
python main.py --mode interactive --camera_id 0
```

### Multiple Object Tracking
```bash
# Multiple object tracking
python main.py --mode mot --video_path video.mp4 --detector yolo --tracker sort

# DeepSORT tracking
python main.py --mode mot --tracker deepsort --reid_model osnet

# Real-time tracking
python main.py --mode realtime --camera_id 0 --tracker bytetrack
```

### Evaluation
```bash
# Evaluate on MOT dataset
python main.py --mode eval --dataset_path MOT17 --tracker deepsort

# Compute tracking metrics
python main.py --mode metrics --gt_path ground_truth.txt --pred_path predictions.txt
```

### Training Custom Tracker
```bash
# Train Siamese tracker
python main.py --mode train --tracker siamese --dataset_path tracking_dataset

# Train ReID model
python main.py --mode train_reid --dataset_path market1501
```

## Tracking Algorithms

### SORT (Simple Online and Realtime Tracking)
- Kalman filter for motion prediction
- Hungarian algorithm for data association
- Linear motion model
- Fast and simple implementation

### DeepSORT
- SORT + deep appearance features
- CNN-based Re-ID model
- Improved identity preservation
- Cosine distance matching

### ByteTrack
- Two-step association strategy
- High and low threshold detections
- Improved handling of low-confidence detections
- State-of-the-art performance

### Siamese Tracker
- Siamese CNN for similarity learning
- Template matching approach
- End-to-end trainable
- Good for single object tracking

## Supported Datasets

- **MOT Challenge**: MOT15, MOT16, MOT17, MOT20
- **KITTI Tracking**: Autonomous driving scenarios
- **UA-DETRAC**: Vehicle tracking dataset
- **LaSOT**: Large-scale single object tracking
- **Custom datasets**: Support for custom annotations

## Performance Metrics

### Multiple Object Tracking
- **MOTA**: Overall tracking accuracy
- **MOTP**: Precision of object localization
- **IDF1**: Identity preservation score
- **MT/ML**: Mostly tracked/lost trajectories
- **FP/FN**: False positives/negatives
- **ID Sw**: Identity switches

### Single Object Tracking
- **Success Rate**: Overlap-based evaluation
- **Precision**: Center location error
- **Robustness**: Failure rate analysis

## Real-time Performance

The implementation achieves:
- SORT: 100+ FPS
- DeepSORT: 30-50 FPS
- ByteTrack: 50+ FPS
- Siamese Tracker: 25+ FPS

## Requirements

- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Scipy >= 1.7.0
- Matplotlib >= 3.3.0
- Filterpy >= 1.4.5

## Advanced Features

- **Multi-camera tracking**: Cross-camera re-identification
- **3D tracking**: Stereo camera tracking
- **Online learning**: Adaptive appearance models
- **Track interpolation**: Fill missing detections
- **Uncertainty estimation**: Confidence-aware tracking
