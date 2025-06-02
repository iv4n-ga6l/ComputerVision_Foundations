"""
Real-time Video Processing Pipeline for Edge Devices

This module implements a comprehensive real-time video processing pipeline
optimized for edge devices with adaptive quality control, performance monitoring,
and multiple AI processing modes.

Author: Computer Vision Engineer
Date: 2024
"""

import cv2
import numpy as np
import threading
import queue
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from enum import Enum
import json
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Available video processing modes."""
    OBJECT_DETECTION = "object_detection"
    POSE_ESTIMATION = "pose_estimation"
    SEGMENTATION = "segmentation"
    STYLE_TRANSFER = "style_transfer"
    CUSTOM_FILTER = "custom_filter"

@dataclass
class PerformanceStats:
    """Performance statistics container."""
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    processing_time_ms: float = 0.0

class FrameBuffer:
    """Thread-safe frame buffer with overflow protection."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.dropped_frames = 0
    
    def put(self, frame: np.ndarray, timestamp: float) -> bool:
        """Add frame to buffer. Returns False if buffer is full."""
        try:
            self.buffer.put_nowait((frame, timestamp))
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False
    
    def get(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get frame from buffer with timeout."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear(self):
        """Clear all frames from buffer."""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break

class ModelManager:
    """Manages AI models for different processing modes."""
    
    def __init__(self):
        self.models = {}
        self.device = "cpu"
        self._detect_device()
        self._load_models()
    
    def _detect_device(self):
        """Detect available compute devices."""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference")
        except ImportError:
            self.device = "cpu"
            logger.info("PyTorch not available, using CPU")
    
    def _load_models(self):
        """Load pre-trained models for different processing modes."""
        # YOLO for object detection
        self._load_yolo_model()
        
        # MediaPipe for pose estimation
        self._load_pose_model()
        
        # Segmentation model
        self._load_segmentation_model()
        
        # Style transfer model
        self._load_style_model()
    
    def _load_yolo_model(self):
        """Load YOLO object detection model."""
        try:
            # Use OpenCV's DNN module for YOLO
            config_path = "models/yolo/yolov4.cfg"
            weights_path = "models/yolo/yolov4.weights"
            
            if not os.path.exists(weights_path):
                logger.warning("YOLO weights not found, using mock detection")
                self.models[ProcessingMode.OBJECT_DETECTION] = self._mock_detection
                return
            
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            if self.device == "cuda":
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.models[ProcessingMode.OBJECT_DETECTION] = net
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.models[ProcessingMode.OBJECT_DETECTION] = self._mock_detection
    
    def _load_pose_model(self):
        """Load pose estimation model."""
        try:
            import mediapipe as mp
            
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            pose_model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.models[ProcessingMode.POSE_ESTIMATION] = pose_model
            logger.info("MediaPipe Pose model loaded successfully")
            
        except ImportError:
            logger.warning("MediaPipe not available, using mock pose estimation")
            self.models[ProcessingMode.POSE_ESTIMATION] = self._mock_pose
    
    def _load_segmentation_model(self):
        """Load semantic segmentation model."""
        try:
            # Mock segmentation for now
            self.models[ProcessingMode.SEGMENTATION] = self._mock_segmentation
            logger.info("Segmentation model loaded (mock)")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            self.models[ProcessingMode.SEGMENTATION] = self._mock_segmentation
    
    def _load_style_model(self):
        """Load style transfer model."""
        try:
            # Mock style transfer for now
            self.models[ProcessingMode.STYLE_TRANSFER] = self._mock_style_transfer
            logger.info("Style transfer model loaded (mock)")
        except Exception as e:
            logger.error(f"Failed to load style transfer model: {e}")
            self.models[ProcessingMode.STYLE_TRANSFER] = self._mock_style_transfer
    
    def _mock_detection(self, frame: np.ndarray) -> np.ndarray:
        """Mock object detection for demonstration."""
        h, w = frame.shape[:2]
        
        # Draw mock bounding boxes
        cv2.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), 2)
        cv2.putText(frame, "Person: 0.85", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (300, 100), (450, 200), (255, 0, 0), 2)
        cv2.putText(frame, "Car: 0.72", (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def _mock_pose(self, frame: np.ndarray) -> np.ndarray:
        """Mock pose estimation for demonstration."""
        h, w = frame.shape[:2]
        
        # Draw mock skeleton
        keypoints = [
            (w//2, h//4),      # Head
            (w//2, h//3),      # Neck
            (w//3, h//2),      # Left shoulder
            (2*w//3, h//2),    # Right shoulder
            (w//2, 2*h//3),    # Hip center
            (w//3, 3*h//4),    # Left hip
            (2*w//3, 3*h//4),  # Right hip
        ]
        
        # Draw keypoints
        for point in keypoints:
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
        
        # Draw connections
        connections = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)]
        for start_idx, end_idx in connections:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
        
        return frame
    
    def _mock_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Mock segmentation for demonstration."""
        # Create a simple color-based segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color (rough approximation)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply color overlay
        overlay = frame.copy()
        overlay[skin_mask > 0] = [0, 255, 0]  # Green for detected skin
        
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return result
    
    def _mock_style_transfer(self, frame: np.ndarray) -> np.ndarray:
        """Mock style transfer for demonstration."""
        # Apply artistic effect (edge enhancement + color modification)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Enhance colors
        enhanced = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        
        # Apply color transformation for artistic effect
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.5  # Increase saturation
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return np.clip(result, 0, 255).astype(np.uint8)

class AdaptiveQualityController:
    """Controls video quality based on performance metrics."""
    
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.current_resolution = None
        self.current_fps = target_fps
        self.performance_history = []
        self.adjustment_cooldown = 0
        
        # Quality levels (width, height, quality_factor)
        self.quality_levels = [
            (1920, 1080, 1.0),  # Full HD
            (1280, 720, 0.8),   # HD
            (854, 480, 0.6),    # SD
            (640, 360, 0.4),    # Low
            (426, 240, 0.2),    # Very Low
        ]
        
        self.current_level = 1  # Start with HD
    
    def update_performance(self, stats: PerformanceStats):
        """Update performance metrics and adjust quality if needed."""
        self.performance_history.append(stats.fps)
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # Adjust quality based on performance
        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return
        
        avg_fps = np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else stats.fps
        
        # If FPS is too low, reduce quality
        if avg_fps < self.target_fps * 0.8 and self.current_level < len(self.quality_levels) - 1:
            self.current_level += 1
            self.adjustment_cooldown = 10
            logger.info(f"Reducing quality to level {self.current_level}")
        
        # If FPS is consistently high, increase quality
        elif avg_fps > self.target_fps * 1.2 and self.current_level > 0:
            self.current_level -= 1
            self.adjustment_cooldown = 10
            logger.info(f"Increasing quality to level {self.current_level}")
    
    def get_current_resolution(self) -> Tuple[int, int]:
        """Get current recommended resolution."""
        return self.quality_levels[self.current_level][:2]
    
    def get_quality_factor(self) -> float:
        """Get current quality factor."""
        return self.quality_levels[self.current_level][2]

class PerformanceMonitor:
    """Monitors system performance and video processing metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.dropped_frames = 0
        self.processing_times = []
        self.process = psutil.Process()
    
    def update_frame_stats(self, processing_time: float, dropped: bool = False):
        """Update frame processing statistics."""
        self.frame_count += 1
        if dropped:
            self.dropped_frames += 1
        else:
            self.processing_times.append(processing_time)
            
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate FPS
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Get system metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        return PerformanceStats(
            fps=fps,
            latency_ms=avg_processing_time * 1000,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            frames_processed=self.frame_count,
            frames_dropped=self.dropped_frames,
            processing_time_ms=avg_processing_time * 1000
        )

class VideoProcessor:
    """Main video processing pipeline with adaptive quality control."""
    
    def __init__(self, 
                 input_source: Any = 0,
                 processing_mode: str = "object_detection",
                 target_fps: float = 30.0,
                 output_resolution: Tuple[int, int] = (640, 480),
                 enable_gpu: bool = True,
                 adaptive_quality: bool = True,
                 buffer_size: int = 10,
                 confidence_threshold: float = 0.5,
                 output_path: Optional[str] = None):
        
        self.input_source = input_source
        self.processing_mode = ProcessingMode(processing_mode)
        self.target_fps = target_fps
        self.output_resolution = output_resolution
        self.enable_gpu = enable_gpu
        self.adaptive_quality = adaptive_quality
        self.confidence_threshold = confidence_threshold
        self.output_path = output_path
        
        # Initialize components
        self.input_buffer = FrameBuffer(buffer_size)
        self.output_buffer = FrameBuffer(buffer_size)
        self.model_manager = ModelManager()
        self.quality_controller = AdaptiveQualityController(target_fps) if adaptive_quality else None
        self.performance_monitor = PerformanceMonitor()
        
        # Threading components
        self.capture_thread = None
        self.processing_thread = None
        self.output_thread = None
        self.monitor_thread = None
        self.running = False
        
        # Video capture/writer
        self.cap = None
        self.writer = None
        
        # Statistics
        self.stats = PerformanceStats()
        
        logger.info(f"VideoProcessor initialized with mode: {processing_mode}")
    
    def start(self):
        """Start the video processing pipeline."""
        try:
            self._initialize_capture()
            self._initialize_output()
            
            self.running = True
            
            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.output_thread = threading.Thread(target=self._output_loop, daemon=True)
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            
            self.capture_thread.start()
            self.processing_thread.start()
            self.output_thread.start()
            self.monitor_thread.start()
            
            logger.info("Video processing pipeline started")
            
        except Exception as e:
            logger.error(f"Failed to start video processor: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the video processing pipeline."""
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2.0)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Cleanup resources
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        
        cv2.destroyAllWindows()
        logger.info("Video processing pipeline stopped")
    
    def _initialize_capture(self):
        """Initialize video capture."""
        if isinstance(self.input_source, int):
            # Webcam
            self.cap = cv2.VideoCapture(self.input_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.output_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.output_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        else:
            # Video file
            self.cap = cv2.VideoCapture(self.input_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.input_source}")
        
        logger.info(f"Video capture initialized: {self.input_source}")
    
    def _initialize_output(self):
        """Initialize video output writer if needed."""
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.target_fps, self.output_resolution
            )
            logger.info(f"Video writer initialized: {self.output_path}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.input_source, str):
                    # End of video file
                    break
                else:
                    continue
            
            # Resize frame if needed
            if self.quality_controller and self.adaptive_quality:
                target_resolution = self.quality_controller.get_current_resolution()
                if frame.shape[:2][::-1] != target_resolution:
                    frame = cv2.resize(frame, target_resolution)
            
            timestamp = time.time()
            
            # Add to buffer
            if not self.input_buffer.put(frame, timestamp):
                self.performance_monitor.update_frame_stats(0, dropped=True)
        
        logger.info("Capture loop finished")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        model = self.model_manager.models.get(self.processing_mode)
        
        while self.running:
            frame_data = self.input_buffer.get(timeout=0.1)
            if frame_data is None:
                continue
            
            frame, timestamp = frame_data
            start_time = time.time()
            
            try:
                # Process frame based on mode
                if self.processing_mode == ProcessingMode.OBJECT_DETECTION:
                    processed_frame = self._process_object_detection(frame, model)
                elif self.processing_mode == ProcessingMode.POSE_ESTIMATION:
                    processed_frame = self._process_pose_estimation(frame, model)
                elif self.processing_mode == ProcessingMode.SEGMENTATION:
                    processed_frame = self._process_segmentation(frame, model)
                elif self.processing_mode == ProcessingMode.STYLE_TRANSFER:
                    processed_frame = self._process_style_transfer(frame, model)
                else:
                    processed_frame = frame  # Pass through
                
                processing_time = time.time() - start_time
                self.performance_monitor.update_frame_stats(processing_time)
                
                # Add to output buffer
                self.output_buffer.put(processed_frame, timestamp)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                # Pass through original frame on error
                self.output_buffer.put(frame, timestamp)
        
        logger.info("Processing loop finished")
    
    def _output_loop(self):
        """Main output loop running in separate thread."""
        while self.running:
            frame_data = self.output_buffer.get(timeout=0.1)
            if frame_data is None:
                continue
            
            frame, timestamp = frame_data
            
            # Display frame
            cv2.imshow('Video Processing', frame)
            
            # Write to file if enabled
            if self.writer:
                self.writer.write(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(f'frame_{int(time.time())}.jpg', frame)
                logger.info("Frame saved")
        
        logger.info("Output loop finished")
    
    def _monitor_loop(self):
        """Performance monitoring loop."""
        while self.running:
            time.sleep(1.0)  # Update every second
            
            # Get current stats
            self.stats = self.performance_monitor.get_stats()
            
            # Update quality controller
            if self.quality_controller:
                self.quality_controller.update_performance(self.stats)
            
            # Log stats periodically
            if int(time.time()) % 10 == 0:  # Every 10 seconds
                self._log_performance_stats()
        
        logger.info("Monitor loop finished")
    
    def _process_object_detection(self, frame: np.ndarray, model) -> np.ndarray:
        """Process frame with object detection."""
        if callable(model):
            return model(frame)
        
        # YOLO detection using OpenCV DNN
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input to the model
        model.setInput(blob)
        
        # Run inference
        outputs = model.forward()
        
        # Process outputs (simplified)
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        # Draw bounding boxes
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Object: {confidence:.2f}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def _process_pose_estimation(self, frame: np.ndarray, model) -> np.ndarray:
        """Process frame with pose estimation."""
        if callable(model):
            return model(frame)
        
        # MediaPipe pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(frame_rgb)
        
        if results.pose_landmarks:
            self.model_manager.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.model_manager.mp_pose.POSE_CONNECTIONS
            )
        
        return frame
    
    def _process_segmentation(self, frame: np.ndarray, model) -> np.ndarray:
        """Process frame with segmentation."""
        if callable(model):
            return model(frame)
        
        # Placeholder for actual segmentation model
        return frame
    
    def _process_style_transfer(self, frame: np.ndarray, model) -> np.ndarray:
        """Process frame with style transfer."""
        if callable(model):
            return model(frame)
        
        # Placeholder for actual style transfer model
        return frame
    
    def _log_performance_stats(self):
        """Log current performance statistics."""
        logger.info(
            f"Performance - FPS: {self.stats.fps:.2f}, "
            f"Latency: {self.stats.latency_ms:.2f}ms, "
            f"CPU: {self.stats.cpu_percent:.1f}%, "
            f"Memory: {self.stats.memory_mb:.1f}MB, "
            f"Dropped: {self.stats.frames_dropped}"
        )
    
    def get_performance_stats(self) -> PerformanceStats:
        """Get current performance statistics."""
        return self.stats
    
    def wait_for_completion(self):
        """Wait for processing to complete."""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    @staticmethod
    def download_models():
        """Download required pre-trained models."""
        logger.info("Downloading models...")
        
        # Create models directory
        os.makedirs("models/yolo", exist_ok=True)
        os.makedirs("models/pose", exist_ok=True)
        os.makedirs("models/segmentation", exist_ok=True)
        
        # Download YOLO weights (placeholder)
        logger.info("YOLO models should be downloaded from official sources")
        logger.info("MediaPipe models will be downloaded automatically")
        
        logger.info("Model download completed")

def create_demo_config() -> Dict[str, Any]:
    """Create demonstration configuration."""
    return {
        'input_source': 0,  # Webcam
        'processing_mode': 'object_detection',
        'target_fps': 30,
        'output_resolution': (640, 480),
        'enable_gpu': True,
        'adaptive_quality': True,
        'buffer_size': 10,
        'confidence_threshold': 0.5,
        'output_path': None
    }

def run_benchmark(config: Dict[str, Any], duration: int = 30):
    """Run performance benchmark."""
    logger.info(f"Starting {duration}s benchmark...")
    
    processor = VideoProcessor(**config)
    
    try:
        processor.start()
        
        # Collect stats for benchmark duration
        start_time = time.time()
        stats_history = []
        
        while time.time() - start_time < duration and processor.running:
            time.sleep(1.0)
            stats = processor.get_performance_stats()
            stats_history.append(stats)
            
            logger.info(
                f"Benchmark [{int(time.time() - start_time)}s] - "
                f"FPS: {stats.fps:.2f}, CPU: {stats.cpu_percent:.1f}%, "
                f"Memory: {stats.memory_mb:.1f}MB"
            )
        
        # Calculate benchmark results
        if stats_history:
            avg_fps = np.mean([s.fps for s in stats_history])
            avg_cpu = np.mean([s.cpu_percent for s in stats_history])
            avg_memory = np.mean([s.memory_mb for s in stats_history])
            total_dropped = stats_history[-1].frames_dropped
            
            logger.info("=== BENCHMARK RESULTS ===")
            logger.info(f"Average FPS: {avg_fps:.2f}")
            logger.info(f"Average CPU Usage: {avg_cpu:.1f}%")
            logger.info(f"Average Memory Usage: {avg_memory:.1f}MB")
            logger.info(f"Total Dropped Frames: {total_dropped}")
            logger.info(f"Drop Rate: {(total_dropped / stats_history[-1].frames_processed * 100):.2f}%")
        
    finally:
        processor.stop()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Real-time Video Processing Pipeline')
    parser.add_argument('--mode', choices=['object_detection', 'pose_estimation', 'segmentation', 'style_transfer'],
                        default='object_detection', help='Processing mode')
    parser.add_argument('--source', default=0, help='Video source (0 for webcam, path for file)')
    parser.add_argument('--fps', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--resolution', default='640x480', help='Resolution (WxH)')
    parser.add_argument('--output', help='Output video file path')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--duration', type=int, default=30, help='Benchmark duration in seconds')
    parser.add_argument('--adaptive', action='store_true', default=True, help='Enable adaptive quality')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        if isinstance(args.source, str) and args.source.isdigit():
            args.source = int(args.source)
    except:
        pass
    
    width, height = map(int, args.resolution.split('x'))
    
    # Create configuration
    config = {
        'input_source': args.source,
        'processing_mode': args.mode,
        'target_fps': args.fps,
        'output_resolution': (width, height),
        'enable_gpu': True,
        'adaptive_quality': args.adaptive,
        'buffer_size': 10,
        'confidence_threshold': 0.5,
        'output_path': args.output
    }
    
    if args.benchmark:
        run_benchmark(config, args.duration)
    else:
        # Run normal processing
        processor = VideoProcessor(**config)
        
        try:
            processor.start()
            processor.wait_for_completion()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            processor.stop()

if __name__ == "__main__":
    main()
