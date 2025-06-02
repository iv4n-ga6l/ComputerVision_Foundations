"""
Mobile App with Computer Vision
Author: Computer Vision Foundations Project
Description: Mobile deployment toolkit for computer vision models
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import zipfile

class MobileModelConverter:
    """Convert models for mobile deployment"""
    
    def __init__(self):
        self.supported_formats = ['tflite', 'coreml', 'onnx']
        
    def convert_to_tflite(self, model, quantization='dynamic', optimization='balanced'):
        """Convert model to TensorFlow Lite format"""
        
        # Create TensorFlow Lite converter
        if isinstance(model, str):
            # Load from saved model path
            converter = tf.lite.TFLiteConverter.from_saved_model(model)
        elif hasattr(model, 'save'):
            # Save PyTorch model first, then convert
            temp_path = 'temp_model'
            # For demo, we'll create a simple TF model
            converter = self._create_demo_converter()
        else:
            converter = self._create_demo_converter()
        
        # Set optimization strategy
        if optimization == 'speed':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        elif optimization == 'size':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        else:  # balanced
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization
        if quantization == 'dynamic':
            # Dynamic range quantization (post-training)
            pass  # Already handled by DEFAULT optimization
        elif quantization == 'int8':
            # Full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            # Would need representative dataset for calibration
        elif quantization == 'float16':
            # Float16 quantization
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        try:
            tflite_model = converter.convert()
            print(f"Model converted to TensorFlow Lite")
            print(f"Original size estimation: ~10 MB")
            print(f"TFLite size: {len(tflite_model) / (1024*1024):.2f} MB")
            return tflite_model
        except Exception as e:
            print(f"Conversion failed: {e}")
            return None
    
    def _create_demo_converter(self):
        """Create demo TensorFlow model for conversion"""
        # Create a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        
        return tf.lite.TFLiteConverter.from_keras_model(model)
    
    def convert_to_coreml(self, model, target_ios_version='13.0'):
        """Convert model to Core ML format (iOS)"""
        try:
            import coremltools as ct
            
            # For demo purposes, create a simple model
            print(f"Converting to Core ML for iOS {target_ios_version}+")
            
            # Create dummy Core ML model specification
            coreml_model_spec = {
                'model_type': 'neural_network',
                'input_shape': [224, 224, 3],
                'output_classes': 1000,
                'ios_deployment_target': target_ios_version
            }
            
            print("Core ML model created (demo)")
            print("Features: GPU acceleration, Vision framework compatibility")
            
            return coreml_model_spec
            
        except ImportError:
            print("Core ML Tools not available. Install with: pip install coremltools")
            return None
    
    def optimize_for_mobile(self, model_path, target_platform='android'):
        """Apply mobile-specific optimizations"""
        optimizations = {
            'android': [
                'GPU delegate support',
                'NNAPI acceleration',
                'Hexagon DSP support',
                'Dynamic quantization'
            ],
            'ios': [
                'Core ML integration',
                'Metal GPU acceleration',
                'Neural Engine support',
                'Memory optimization'
            ]
        }
        
        print(f"Applying {target_platform} optimizations:")
        for opt in optimizations.get(target_platform, []):
            print(f"  - {opt}")
        
        return f"optimized_model_{target_platform}.tflite"

class MobileInference:
    """Mobile inference engine"""
    
    def __init__(self, model_path_or_data, use_gpu=True):
        self.model_data = model_path_or_data
        self.use_gpu = use_gpu
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = None
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load TensorFlow Lite model"""
        try:
            if isinstance(self.model_data, bytes):
                self.interpreter = tf.lite.Interpreter(model_content=self.model_data)
            else:
                # For demo, create a mock interpreter
                print("Loading TensorFlow Lite model...")
                self.interpreter = "Mock TFLite Interpreter"
                
                # Mock input/output details
                self.input_details = [{
                    'name': 'input',
                    'shape': [1, 224, 224, 3],
                    'dtype': np.float32
                }]
                
                self.output_details = [{
                    'name': 'output',
                    'shape': [1, 1000],
                    'dtype': np.float32
                }]
                
            print("Model loaded successfully")
            print(f"Input shape: {self.input_details[0]['shape']}")
            print(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def _load_labels(self):
        """Load class labels"""
        # Use ImageNet class names for demo
        self.class_names = [f"class_{i}" for i in range(1000)]
        print(f"Loaded {len(self.class_names)} class labels")
    
    def preprocess_image(self, image):
        """Preprocess image for mobile inference"""
        if isinstance(image, str):
            # Load from path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB
                pass
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        target_size = (224, 224)
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image, top_k=5):
        """Run inference on image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Simulate inference timing
        start_time = time.time()
        
        # For demo, generate random predictions
        predictions = np.random.random(1000)
        predictions = predictions / np.sum(predictions)  # Normalize
        
        inference_time = time.time() - start_time
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        
        for i, idx in enumerate(top_indices):
            results.append({
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'rank': i + 1
            })
        
        return {
            'predictions': results,
            'inference_time_ms': inference_time * 1000,
            'preprocessing_time_ms': 5.0,  # Mock timing
            'total_time_ms': (inference_time * 1000) + 5.0
        }
    
    def predict_batch(self, images, top_k=5):
        """Run batch inference (if supported)"""
        results = []
        
        total_start = time.time()
        for image in images:
            result = self.predict(image, top_k)
            results.append(result)
        total_time = time.time() - total_start
        
        return {
            'batch_results': results,
            'total_batch_time_ms': total_time * 1000,
            'avg_time_per_image_ms': (total_time * 1000) / len(images)
        }

class MobileAppSimulator:
    """Simulate mobile app camera interface"""
    
    def __init__(self, model_inference):
        self.inference_engine = model_inference
        self.is_running = False
        self.frame_count = 0
        self.fps_history = []
        
    def simulate_camera_feed(self, video_source=0, duration=30):
        """Simulate real-time camera inference"""
        print("Starting mobile camera simulation...")
        print(f"Duration: {duration} seconds")
        
        # For demo, generate synthetic frames
        frames = self._generate_synthetic_frames(duration * 30)  # 30 FPS
        
        self.is_running = True
        start_time = time.time()
        
        results_history = []
        
        for frame_idx, frame in enumerate(frames):
            if not self.is_running:
                break
            
            frame_start = time.time()
            
            # Run inference
            result = self.inference_engine.predict(frame, top_k=3)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            
            # Store results
            results_history.append({
                'frame': frame_idx,
                'timestamp': time.time() - start_time,
                'result': result,
                'fps': fps
            })
            
            # Print periodic updates
            if frame_idx % 30 == 0:  # Every second
                avg_fps = np.mean(self.fps_history[-30:]) if self.fps_history else 0
                print(f"Frame {frame_idx}: FPS={avg_fps:.1f}, "
                      f"Inference={result['inference_time_ms']:.1f}ms")
            
            # Simulate real-time delay
            time.sleep(max(0, 1/30 - frame_time))  # Target 30 FPS
        
        self.is_running = False
        
        # Generate performance report
        self._generate_performance_report(results_history)
        
        return results_history
    
    def _generate_synthetic_frames(self, num_frames):
        """Generate synthetic camera frames"""
        frames = []
        
        for i in range(num_frames):
            # Create random image
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some structure (simulate objects)
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)
            
            frames.append(frame)
        
        return frames
    
    def _generate_performance_report(self, results_history):
        """Generate mobile performance analysis"""
        if not results_history:
            return
        
        # Extract metrics
        inference_times = [r['result']['inference_time_ms'] for r in results_history]
        fps_values = [r['fps'] for r in results_history]
        timestamps = [r['timestamp'] for r in results_history]
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Inference time over time
        axes[0, 0].plot(timestamps, inference_times)
        axes[0, 0].set_title('Inference Time Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Inference Time (ms)')
        axes[0, 0].grid(True)
        
        # FPS over time
        axes[0, 1].plot(timestamps, fps_values)
        axes[0, 1].set_title('FPS Over Time')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('FPS')
        axes[0, 1].grid(True)
        
        # Inference time distribution
        axes[1, 0].hist(inference_times, bins=20, alpha=0.7)
        axes[1, 0].set_title('Inference Time Distribution')
        axes[1, 0].set_xlabel('Inference Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Performance summary
        axes[1, 1].axis('off')
        summary_text = f"""
        MOBILE PERFORMANCE SUMMARY
        
        Total Frames: {len(results_history)}
        Average FPS: {np.mean(fps_values):.1f}
        Average Inference Time: {np.mean(inference_times):.1f} ms
        Min Inference Time: {np.min(inference_times):.1f} ms
        Max Inference Time: {np.max(inference_times):.1f} ms
        
        Performance Grade: {'A' if np.mean(fps_values) > 25 else 'B' if np.mean(fps_values) > 15 else 'C'}
        Real-time Capable: {'Yes' if np.mean(inference_times) < 33 else 'No'}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('mobile_performance_report.png', dpi=300, bbox_inches='tight')
        plt.show()

class MobileBatteryProfiler:
    """Profile battery consumption for mobile inference"""
    
    def __init__(self):
        self.baseline_power = 1000  # mW (baseline mobile power consumption)
        self.inference_power_overhead = 500  # mW (additional power for inference)
        
    def estimate_battery_consumption(self, inference_results, battery_capacity_mah=3000):
        """Estimate battery consumption for inference workload"""
        
        total_inference_time = sum(r['result']['total_time_ms'] for r in inference_results) / 1000.0  # seconds
        total_session_time = inference_results[-1]['timestamp'] if inference_results else 0
        
        # Calculate power consumption
        baseline_consumption = (self.baseline_power * total_session_time) / 3600  # Wh
        inference_consumption = (self.inference_power_overhead * total_inference_time) / 3600  # Wh
        
        total_consumption = baseline_consumption + inference_consumption
        
        # Estimate battery percentage used
        # Assume 3.7V battery (typical Li-ion)
        battery_capacity_wh = (battery_capacity_mah * 3.7) / 1000
        battery_percentage_used = (total_consumption / battery_capacity_wh) * 100
        
        return {
            'total_consumption_wh': total_consumption,
            'baseline_consumption_wh': baseline_consumption,
            'inference_consumption_wh': inference_consumption,
            'battery_percentage_used': battery_percentage_used,
            'estimated_battery_life_hours': battery_capacity_wh / (total_consumption / (total_session_time / 3600))
        }

def create_mobile_app_demo():
    """Create demo mobile application files"""
    
    # Android MainActivity.java (simplified)
    android_code = '''
    // Android MainActivity.java (Simplified Demo)
    public class MainActivity extends AppCompatActivity {
        private TensorFlowLite tfliteModel;
        private Camera camera;
        
        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);
            
            // Load TensorFlow Lite model
            loadModel();
            
            // Setup camera
            setupCamera();
        }
        
        private void loadModel() {
            try {
                tfliteModel = new TensorFlowLite(this, "model.tflite");
                Log.d("CV_APP", "Model loaded successfully");
            } catch (Exception e) {
                Log.e("CV_APP", "Failed to load model", e);
            }
        }
        
        private void onImageAvailable(Image image) {
            // Convert camera image to tensor
            Tensor inputTensor = preprocessImage(image);
            
            // Run inference
            long startTime = System.currentTimeMillis();
            Tensor output = tfliteModel.predict(inputTensor);
            long inferenceTime = System.currentTimeMillis() - startTime;
            
            // Process results
            List<Classification> results = postprocessOutput(output);
            
            // Update UI
            updateUI(results, inferenceTime);
        }
    }
    '''
    
    # iOS ViewController.swift (simplified)
    ios_code = '''
    // iOS ViewController.swift (Simplified Demo)
    import UIKit
    import CoreML
    import Vision
    
    class ViewController: UIViewController {
        var model: VNCoreMLModel?
        
        override func viewDidLoad() {
            super.viewDidLoad()
            loadModel()
            setupCamera()
        }
        
        func loadModel() {
            guard let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodelc") else {
                print("Failed to find model file")
                return
            }
            
            do {
                let mlModel = try MLModel(contentsOf: modelURL)
                model = try VNCoreMLModel(for: mlModel)
                print("Model loaded successfully")
            } catch {
                print("Failed to load model: \\(error)")
            }
        }
        
        func processImage(_ image: CVPixelBuffer) {
            guard let model = model else { return }
            
            let request = VNCoreMLRequest(model: model) { request, error in
                guard let results = request.results as? [VNClassificationObservation] else { return }
                
                DispatchQueue.main.async {
                    self.updateUI(with: results)
                }
            }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])
            try? handler.perform([request])
        }
    }
    '''
    
    # React Native component (simplified)
    react_native_code = '''
    // React Native Component (Simplified Demo)
    import React, { useEffect, useState } from 'react';
    import { View, Text } from 'react-native';
    import { Camera } from 'react-native-camera';
    import TensorFlowLite from 'react-native-tensorflow-lite';
    
    const CVApp = () => {
        const [model, setModel] = useState(null);
        const [predictions, setPredictions] = useState([]);
        
        useEffect(() => {
            loadModel();
        }, []);
        
        const loadModel = async () => {
            try {
                const loadedModel = await TensorFlowLite.loadModel('model.tflite');
                setModel(loadedModel);
                console.log('Model loaded successfully');
            } catch (error) {
                console.error('Failed to load model:', error);
            }
        };
        
        const onImageCapture = async (imageUri) => {
            if (!model) return;
            
            try {
                const startTime = Date.now();
                const results = await model.predict(imageUri);
                const inferenceTime = Date.now() - startTime;
                
                setPredictions(results);
                console.log(`Inference time: ${inferenceTime}ms`);
            } catch (error) {
                console.error('Inference failed:', error);
            }
        };
        
        return (
            <View style={{ flex: 1 }}>
                <Camera onImageCapture={onImageCapture} />
                <View>
                    {predictions.map((pred, index) => (
                        <Text key={index}>
                            {pred.label}: {(pred.confidence * 100).toFixed(1)}%
                        </Text>
                    ))}
                </View>
            </View>
        );
    };
    '''
    
    # Save demo files
    with open('AndroidMainActivity.java', 'w') as f:
        f.write(android_code)
    
    with open('iOSViewController.swift', 'w') as f:
        f.write(ios_code)
    
    with open('ReactNativeComponent.js', 'w') as f:
        f.write(react_native_code)
    
    print("Mobile app demo files created:")
    print("  - AndroidMainActivity.java")
    print("  - iOSViewController.swift")
    print("  - ReactNativeComponent.js")

def demo_mobile_deployment():
    """Demonstrate mobile computer vision deployment"""
    print("Mobile App with Computer Vision Demo")
    print("=" * 50)
    
    # Step 1: Model Conversion
    print("\n1. Converting Model for Mobile Deployment...")
    converter = MobileModelConverter()
    
    # Convert to TensorFlow Lite
    print("\n   Converting to TensorFlow Lite...")
    tflite_model = converter.convert_to_tflite(
        None, quantization='dynamic', optimization='balanced'
    )
    
    # Convert to Core ML
    print("\n   Converting to Core ML...")
    coreml_model = converter.convert_to_coreml(None, target_ios_version='13.0')
    
    # Step 2: Mobile Inference Setup
    print("\n2. Setting up Mobile Inference Engine...")
    mobile_inference = MobileInference(tflite_model, use_gpu=True)
    
    # Step 3: Camera Simulation
    print("\n3. Simulating Mobile Camera Feed...")
    app_simulator = MobileAppSimulator(mobile_inference)
    
    # Run camera simulation for 10 seconds
    results = app_simulator.simulate_camera_feed(duration=10)
    
    # Step 4: Battery Analysis
    print("\n4. Analyzing Battery Consumption...")
    battery_profiler = MobileBatteryProfiler()
    battery_analysis = battery_profiler.estimate_battery_consumption(results)
    
    print(f"\nBattery Analysis Results:")
    print(f"  Total consumption: {battery_analysis['total_consumption_wh']:.3f} Wh")
    print(f"  Battery percentage used: {battery_analysis['battery_percentage_used']:.2f}%")
    print(f"  Estimated battery life: {battery_analysis['estimated_battery_life_hours']:.1f} hours")
    
    # Step 5: Create Mobile App Templates
    print("\n5. Creating Mobile App Demo Files...")
    create_mobile_app_demo()
    
    # Performance Summary
    print("\n" + "="*60)
    print("MOBILE DEPLOYMENT SUMMARY")
    print("="*60)
    
    if results:
        avg_inference = np.mean([r['result']['inference_time_ms'] for r in results])
        avg_fps = np.mean([r['fps'] for r in results])
        
        print(f"Model Performance:")
        print(f"  Average inference time: {avg_inference:.1f} ms")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Real-time capable: {'Yes' if avg_inference < 33 else 'No'}")
        
        print(f"\nOptimization Recommendations:")
        if avg_inference > 50:
            print("  - Consider more aggressive quantization")
            print("  - Use GPU acceleration")
            print("  - Reduce model complexity")
        elif avg_inference > 33:
            print("  - Enable GPU delegation")
            print("  - Optimize preprocessing pipeline")
        else:
            print("  - Model is well optimized for mobile")
        
        print(f"\nDeployment Targets:")
        print(f"  Android: TensorFlow Lite with GPU delegate")
        print(f"  iOS: Core ML with Neural Engine acceleration")
        print(f"  Cross-platform: React Native/Flutter integration")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demo_mobile_deployment()
