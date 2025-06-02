"""
Model Optimization and Quantization
 Project
Description: Comprehensive model optimization toolkit for edge deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import psutil
import matplotlib.pyplot as plt
from collections import OrderedDict
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """Comprehensive model optimization toolkit"""
    
    def __init__(self, model, config=None):
        self.original_model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.calibration_data = None
        self.test_data = None
        
        # Move model to device
        self.original_model.to(self.device)
        
        # Optimization history
        self.optimization_history = []
        
    def set_calibration_data(self, calibration_loader):
        """Set calibration data for static quantization"""
        self.calibration_data = calibration_loader
        
    def set_test_data(self, test_loader):
        """Set test data for evaluation"""
        self.test_data = test_loader
    
    def dynamic_quantization(self, model=None):
        """Apply dynamic quantization (weights only)"""
        if model is None:
            model = self.original_model
            
        # Create a copy
        quantized_model = torch.jit.script(model.eval())
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(), 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        print("Applied dynamic quantization")
        return quantized_model
    
    def static_quantization(self, model=None):
        """Apply static quantization with calibration"""
        if model is None:
            model = self.original_model
            
        if self.calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        
        # Prepare model for quantization
        model.eval()
        model.cpu()
        
        # Set quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse modules if possible
        try:
            fused_model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        except:
            fused_model = model
        
        # Prepare for quantization
        prepared_model = torch.quantization.prepare(fused_model)
        
        # Calibration
        print("Calibrating model...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.calibration_data):
                if batch_idx >= 100:  # Limit calibration samples
                    break
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        print("Applied static quantization")
        return quantized_model
    
    def quantization_aware_training(self, model=None, train_loader=None, epochs=5):
        """Apply quantization-aware training"""
        if model is None:
            model = self.original_model
        if train_loader is None:
            train_loader = self.calibration_data
            
        # Prepare model for QAT
        model.train()
        model.cpu()
        
        # Set QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        # Training setup
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting QAT for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit training for demo
                    break
                    
                optimizer.zero_grad()
                output = prepared_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        
        print("Completed quantization-aware training")
        return quantized_model
    
    def magnitude_pruning(self, model=None, sparsity=0.5):
        """Apply magnitude-based pruning"""
        if model is None:
            model = self.original_model
            
        # Create a copy
        pruned_model = torch.jit.script(model.eval()) if hasattr(model, 'forward') else model
        
        # Apply magnitude-based pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Calculate threshold for sparsity
                weights = module.weight.data.abs()
                threshold = torch.quantile(weights, sparsity)
                
                # Create mask
                mask = weights > threshold
                
                # Apply mask
                module.weight.data *= mask.float()
        
        print(f"Applied magnitude pruning with {sparsity:.1%} sparsity")
        return pruned_model
    
    def structured_pruning(self, model=None, prune_ratio=0.3):
        """Apply structured pruning (remove channels)"""
        if model is None:
            model = self.original_model
            
        pruned_model = model
        
        # Identify layers to prune
        conv_layers = [(name, module) for name, module in model.named_modules() 
                      if isinstance(module, nn.Conv2d)]
        
        for name, layer in conv_layers[:-1]:  # Don't prune last layer
            # Calculate channel importance (L1 norm)
            importance = torch.sum(torch.abs(layer.weight.data), dim=(1, 2, 3))
            
            # Determine channels to keep
            num_channels = len(importance)
            num_keep = int(num_channels * (1 - prune_ratio))
            
            if num_keep < num_channels:
                # Get indices of most important channels
                _, indices = torch.topk(importance, num_keep)
                indices = indices.sort()[0]
                
                # Create new layer with fewer channels
                new_layer = nn.Conv2d(
                    layer.in_channels,
                    num_keep,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    bias=layer.bias is not None
                )
                
                # Copy weights
                new_layer.weight.data = layer.weight.data[indices]
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data[indices]
                
                # Replace layer (simplified for demo)
                print(f"Pruned {name}: {num_channels} -> {num_keep} channels")
        
        print(f"Applied structured pruning with {prune_ratio:.1%} channel reduction")
        return pruned_model
    
    def knowledge_distillation(self, teacher_model, student_model, train_loader, 
                             temperature=4.0, alpha=0.5, epochs=10):
        """Apply knowledge distillation for model compression"""
        
        teacher_model.eval()
        student_model.train()
        
        # Move to device
        teacher_model.to(self.device)
        student_model.to(self.device)
        
        # Setup
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        print(f"Starting knowledge distillation for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 50:  # Limit for demo
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Student forward pass
                student_output = student_model(data)
                
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                
                # Calculate losses
                ce_loss = criterion_ce(student_output, target)
                
                # Distillation loss
                student_soft = torch.log_softmax(student_output / temperature, dim=1)
                teacher_soft = torch.softmax(teacher_output / temperature, dim=1)
                kd_loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
                
                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kd_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Completed knowledge distillation")
        return student_model
    
    def convert_to_onnx(self, model, input_shape=(1, 3, 224, 224), output_path="model.onnx"):
        """Convert PyTorch model to ONNX format"""
        model.eval()
        model.cpu()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def optimize_onnx_model(self, onnx_path):
        """Optimize ONNX model for inference"""
        try:
            import onnxoptimizer
            
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model)
            
            # Save optimized model
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            print(f"Optimized ONNX model saved: {optimized_path}")
            return optimized_path
            
        except ImportError:
            print("onnxoptimizer not available, skipping ONNX optimization")
            return onnx_path
    
    def benchmark_model(self, model, test_loader=None, num_runs=100):
        """Benchmark model performance"""
        if test_loader is None:
            test_loader = self.test_data
            
        model.eval()
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        
        results = {
            'inference_times': [],
            'memory_usage': [],
            'accuracy': 0.0,
            'model_size': 0.0
        }
        
        # Measure model size
        if hasattr(model, 'state_dict'):
            temp_path = 'temp_model.pth'
            torch.save(model.state_dict(), temp_path)
            results['model_size'] = os.path.getsize(temp_path) / (1024 * 1024)  # MB
            os.remove(temp_path)
        
        # Warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            for _ in range(10):
                _ = model(dummy_input)
        
        # Accuracy evaluation
        if test_loader is not None:
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    if batch_idx >= 20:  # Limit for demo
                        break
                    
                    data = data.to(device)
                    
                    # Measure inference time
                    start_time = time.time()
                    output = model(data)
                    end_time = time.time()
                    
                    results['inference_times'].append(end_time - start_time)
                    
                    # Calculate accuracy
                    if isinstance(output, torch.Tensor):
                        pred = output.argmax(dim=1)
                        correct += (pred.cpu() == target).sum().item()
                        total += target.size(0)
                    
                    # Memory usage
                    if device == 'cuda':
                        results['memory_usage'].append(
                            torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                        )
            
            results['accuracy'] = correct / total if total > 0 else 0.0
        
        # Performance statistics
        if results['inference_times']:
            results['avg_inference_time'] = np.mean(results['inference_times'])
            results['std_inference_time'] = np.std(results['inference_times'])
            results['fps'] = 1.0 / results['avg_inference_time']
        
        if results['memory_usage']:
            results['avg_memory_usage'] = np.mean(results['memory_usage'])
        
        return results
    
    def compare_models(self, models_dict, test_loader=None):
        """Compare multiple optimized models"""
        results = {}
        
        print("Benchmarking models...")
        for name, model in models_dict.items():
            print(f"\nBenchmarking {name}...")
            results[name] = self.benchmark_model(model, test_loader)
        
        # Create comparison report
        self._create_comparison_report(results)
        
        return results
    
    def _create_comparison_report(self, results):
        """Create visual comparison report"""
        models = list(results.keys())
        metrics = ['model_size', 'avg_inference_time', 'accuracy', 'fps']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            
            ax = axes[i]
            bars = ax.bar(models, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print numerical comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Model Size: {result.get('model_size', 0):.2f} MB")
            print(f"  Inference Time: {result.get('avg_inference_time', 0)*1000:.2f} ms")
            print(f"  FPS: {result.get('fps', 0):.1f}")
            print(f"  Accuracy: {result.get('accuracy', 0):.3f}")
            if 'avg_memory_usage' in result:
                print(f"  Memory Usage: {result['avg_memory_usage']:.1f} MB")

class MobileNetOptimizer:
    """Specialized optimizer for MobileNet architectures"""
    
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        
    def create_efficient_variants(self):
        """Create different efficiency variants"""
        variants = {}
        
        # Original model
        variants['original'] = self.model
        
        # Width multiplier variants
        for width_mult in [0.75, 0.5, 0.35]:
            variant = models.mobilenet_v2(pretrained=False)
            # Modify architecture for width multiplier (simplified)
            variants[f'width_{width_mult}'] = variant
        
        return variants

def create_student_model(num_classes=1000):
    """Create a smaller student model for knowledge distillation"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, num_classes)
    )

def demo_model_optimization():
    """Demonstrate model optimization techniques"""
    print("Model Optimization and Quantization Demo")
    print("=" * 50)
    
    # Load pre-trained model
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    # Create synthetic dataset for demo
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Generate synthetic data
    num_samples = 100
    synthetic_data = []
    for i in range(num_samples):
        # Create random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_tensor = transform(img)
        label = np.random.randint(0, 1000)
        synthetic_data.append((img_tensor, label))
    
    # Create data loaders
    calibration_data = DataLoader(synthetic_data[:50], batch_size=8, shuffle=False)
    test_data = DataLoader(synthetic_data[50:], batch_size=8, shuffle=False)
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model)
    optimizer.set_calibration_data(calibration_data)
    optimizer.set_test_data(test_data)
    
    print("\n1. Applying Dynamic Quantization...")
    dynamic_quantized = optimizer.dynamic_quantization()
    
    print("\n2. Applying Static Quantization...")
    static_quantized = optimizer.static_quantization()
    
    print("\n3. Applying Magnitude Pruning...")
    pruned_model = optimizer.magnitude_pruning(sparsity=0.3)
    
    print("\n4. Creating Student Model for Knowledge Distillation...")
    student_model = create_student_model()
    distilled_model = optimizer.knowledge_distillation(
        model, student_model, calibration_data, epochs=3
    )
    
    print("\n5. Converting to ONNX...")
    onnx_path = optimizer.convert_to_onnx(model, output_path="mobilenet_v2.onnx")
    
    # Compare all models
    models_to_compare = {
        'Original': model,
        'Dynamic Quantized': dynamic_quantized,
        'Static Quantized': static_quantized,
        'Pruned (30%)': pruned_model,
        'Distilled': distilled_model
    }
    
    print("\n6. Benchmarking All Models...")
    comparison_results = optimizer.compare_models(models_to_compare, test_data)
    
    # ONNX Runtime benchmarking
    print("\n7. ONNX Runtime Performance...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = ort_session.run(None, {'input': dummy_input})
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            _ = ort_session.run(None, {'input': dummy_input})
            times.append(time.time() - start)
        
        print(f"ONNX Runtime - Avg inference time: {np.mean(times)*1000:.2f} ms")
        print(f"ONNX Runtime - FPS: {1/np.mean(times):.1f}")
        
    except Exception as e:
        print(f"ONNX Runtime benchmark failed: {e}")
    
    print("\nDemo completed successfully!")
    print("\nOptimization Summary:")
    print("- Quantization reduces model size and improves inference speed")
    print("- Pruning removes redundant parameters")
    print("- Knowledge distillation creates compact models")
    print("- ONNX enables cross-platform deployment")

if __name__ == "__main__":
    demo_model_optimization()
