# Project 4: Real-time Video Processing Pipeline

## Overview
This project implements a comprehensive real-time video processing pipeline optimized for edge devices. It includes video streaming, preprocessing, inference, post-processing, and output with performance monitoring and adaptive quality control.

## Features

### Core Components
- **Multi-threaded Video Pipeline**: Separate threads for capture, processing, and output
- **Adaptive Quality Control**: Dynamic resolution and frame rate adjustment based on performance
- **Buffer Management**: Efficient frame buffering with overflow protection
- **Performance Monitoring**: Real-time FPS, latency, and resource usage tracking
- **Multiple Processing Modes**: Object detection, pose estimation, segmentation, style transfer

### Video Processing Features
- **Real-time Object Detection**: YOLO-based detection with confidence filtering
- **Human Pose Estimation**: MediaPipe-based pose detection and tracking
- **Semantic Segmentation**: Real-time segmentation with visualization
- **Style Transfer**: Fast neural style transfer for artistic effects
- **Custom Filters**: Edge detection, blur, color space transformations

### Performance Optimization
- **Frame Skipping**: Intelligent frame dropping under high load
- **Resolution Scaling**: Dynamic resolution adjustment for performance
- **Batch Processing**: Efficient batch inference when possible
- **Memory Management**: Optimized memory usage and garbage collection
- **GPU Acceleration**: CUDA/OpenCL support where available

### Edge Device Support
- **Raspberry Pi**: Optimized for ARM processors with limited resources
- **NVIDIA Jetson**: GPU acceleration with TensorRT optimization
- **Intel NUC**: x86 optimization with OpenVINO inference
- **Mobile Devices**: Android/iOS deployment support

## Requirements
- Python 3.8+
- OpenCV 4.5+
- PyTorch or TensorFlow
- NumPy, threading
- Optional: CUDA, TensorRT, OpenVINO

## Usage

### Basic Usage
```python
from main import VideoProcessor

# Initialize processor
processor = VideoProcessor(
    input_source=0,  # Webcam
    processing_mode='object_detection',
    target_fps=30,
    output_resolution=(640, 480)
)

# Start processing
processor.start()
processor.wait_for_completion()
```

### Configuration Options
```python
config = {
    'input_source': 0,  # 0 for webcam, file path for video
    'processing_mode': 'object_detection',  # detection, pose, segmentation, style
    'target_fps': 30,
    'output_resolution': (640, 480),
    'enable_gpu': True,
    'adaptive_quality': True,
    'buffer_size': 10,
    'confidence_threshold': 0.5
}

processor = VideoProcessor(**config)
```

### Performance Monitoring
```python
# Get real-time statistics
stats = processor.get_performance_stats()
print(f"FPS: {stats['fps']:.2f}")
print(f"Latency: {stats['latency_ms']:.2f}ms")
print(f"CPU Usage: {stats['cpu_percent']:.1f}%")
print(f"Memory Usage: {stats['memory_mb']:.1f}MB")
```

## Project Structure
```
Project 4/
├── main.py              # Main video processing pipeline
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── models/              # Pre-trained models
│   ├── yolo/           # YOLO detection models
│   ├── pose/           # Pose estimation models
│   └── segmentation/   # Segmentation models
├── utils/              # Utility functions
│   ├── video_utils.py  # Video I/O utilities
│   ├── performance.py  # Performance monitoring
│   └── filters.py      # Video filters and effects
└── examples/           # Example usage scripts
    ├── webcam_demo.py  # Webcam processing demo
    ├── file_demo.py    # Video file processing
    └── benchmark.py    # Performance benchmarking
```

## Key Components

### 1. Video Pipeline Architecture
- **Producer Thread**: Captures frames from input source
- **Consumer Thread**: Processes frames with AI models
- **Output Thread**: Displays/saves processed frames
- **Monitor Thread**: Tracks performance metrics

### 2. Adaptive Quality Control
- **Dynamic Resolution**: Adjusts resolution based on processing speed
- **Frame Rate Control**: Drops frames to maintain real-time performance
- **Quality Metrics**: Monitors latency and adjusts parameters automatically

### 3. Processing Modes
- **Object Detection**: Real-time YOLO-based object detection
- **Pose Estimation**: Human pose detection and tracking
- **Segmentation**: Semantic segmentation with color mapping
- **Style Transfer**: Artistic style transfer effects

### 4. Performance Optimization
- **Memory Pooling**: Reuses frame buffers to reduce allocation overhead
- **Batch Processing**: Groups operations for efficiency
- **Hardware Acceleration**: Utilizes GPU/NPU when available
- **Profiling Tools**: Built-in performance analysis

## Performance Targets

### Raspberry Pi 4
- Resolution: 320x240 @ 15 FPS
- Object Detection: 10-15 FPS
- Pose Estimation: 8-12 FPS
- Memory Usage: < 500MB

### NVIDIA Jetson Nano
- Resolution: 640x480 @ 30 FPS
- Object Detection: 20-30 FPS
- Pose Estimation: 15-25 FPS
- Memory Usage: < 1GB

### Desktop/Laptop
- Resolution: 1280x720 @ 60 FPS
- Object Detection: 30-60 FPS
- Pose Estimation: 25-45 FPS
- Memory Usage: < 2GB

## Advanced Features

### 1. Multi-Camera Support
- Simultaneous processing of multiple video streams
- Camera synchronization and calibration
- Stereo vision processing capabilities

### 2. Network Streaming
- RTMP/WebRTC streaming output
- Remote monitoring and control
- Distributed processing support

### 3. Custom Model Integration
- Easy integration of custom trained models
- Model switching during runtime
- A/B testing framework for models

### 4. Edge Analytics
- Local data processing and storage
- Event detection and alerting
- Analytics dashboard and reporting

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download pre-trained models:
```python
python -c "from main import VideoProcessor; VideoProcessor.download_models()"
```

3. Run basic demo:
```bash
python main.py --mode object_detection --source 0
```

4. Run performance benchmark:
```bash
python examples/benchmark.py --device auto
```

## Troubleshooting

### Common Issues
1. **Low FPS**: Enable adaptive quality, reduce resolution
2. **High Memory Usage**: Decrease buffer size, enable garbage collection
3. **GPU Not Detected**: Install proper CUDA/OpenCL drivers
4. **Camera Not Found**: Check device permissions and connections

### Performance Tuning
- Adjust `target_fps` based on hardware capabilities
- Enable `adaptive_quality` for automatic optimization
- Use appropriate `processing_mode` for your use case
- Monitor performance stats and adjust accordingly

## Future Enhancements
- WebAssembly deployment for browsers
- Edge TPU support for Google Coral
- Advanced video analytics (crowd counting, behavior analysis)
- Integration with cloud services for hybrid processing
