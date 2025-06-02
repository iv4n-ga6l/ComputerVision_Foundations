"""
Convolutional Neural Network from Scratch
=========================================

Implementation of a complete CNN without using high-level frameworks, including
convolution layers, pooling layers, and backpropagation through all components.


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
from pathlib import Path

class ConvolutionLayer:
    """Convolutional layer implementation"""
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.weights = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        
        self.biases = np.zeros((output_channels, 1))
        
        # For storing gradients
        self.dW = None
        self.db = None
        
        # For storing intermediate values
        self.input_data = None
        self.output_data = None
    
    def add_padding(self, data, padding):
        """Add zero padding to input data"""
        if padding == 0:
            return data
        
        if len(data.shape) == 3:  # Single image
            return np.pad(data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:  # Batch of images
            return np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    def forward(self, input_data):
        """Forward pass through convolution layer"""
        self.input_data = input_data
        
        # Add padding if needed
        padded_input = self.add_padding(input_data, self.padding)
        
        if len(input_data.shape) == 3:  # Single image
            batch_size = 1
            padded_input = padded_input.reshape(1, *padded_input.shape)
        else:
            batch_size = input_data.shape[0]
        
        # Calculate output dimensions
        input_height, input_width = padded_input.shape[2], padded_input.shape[3]
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.output_channels):
                for i in range(0, output_height * self.stride, self.stride):
                    for j in range(0, output_width * self.stride, self.stride):
                        output_i = i // self.stride
                        output_j = j // self.stride
                        
                        # Extract region
                        region = padded_input[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                        
                        # Convolution operation
                        output[b, f, output_i, output_j] = np.sum(region * self.weights[f]) + self.biases[f]
        
        if len(input_data.shape) == 3:
            output = output.squeeze(0)
        
        self.output_data = output
        return output
    
    def backward(self, grad_output):
        """Backward pass through convolution layer"""
        if len(self.input_data.shape) == 3:  # Single image
            batch_size = 1
            input_data = self.input_data.reshape(1, *self.input_data.shape)
            grad_output = grad_output.reshape(1, *grad_output.shape)
        else:
            batch_size = self.input_data.shape[0]
            input_data = self.input_data
        
        # Add padding to input
        padded_input = self.add_padding(input_data, self.padding)
        
        # Initialize gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)
        grad_input = np.zeros_like(padded_input)
        
        output_height, output_width = grad_output.shape[2], grad_output.shape[3]
        
        # Compute gradients
        for b in range(batch_size):
            for f in range(self.output_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Input region indices
                        start_i = i * self.stride
                        end_i = start_i + self.kernel_size
                        start_j = j * self.stride
                        end_j = start_j + self.kernel_size
                        
                        # Extract region
                        region = padded_input[b, :, start_i:end_i, start_j:end_j]
                        
                        # Gradient w.r.t. weights
                        self.dW[f] += grad_output[b, f, i, j] * region
                        
                        # Gradient w.r.t. input
                        grad_input[b, :, start_i:end_i, start_j:end_j] += \
                            grad_output[b, f, i, j] * self.weights[f]
        
        # Gradient w.r.t. biases
        self.db = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1)
        
        # Remove padding from gradient
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        if len(self.input_data.shape) == 3:
            grad_input = grad_input.squeeze(0)
        
        return grad_input

class MaxPoolingLayer:
    """Max pooling layer implementation"""
    
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
        # For storing intermediate values
        self.input_data = None
        self.mask = None
    
    def forward(self, input_data):
        """Forward pass through max pooling layer"""
        self.input_data = input_data
        
        if len(input_data.shape) == 3:  # Single image
            batch_size = 1
            input_data = input_data.reshape(1, *input_data.shape)
        else:
            batch_size = input_data.shape[0]
        
        channels, input_height, input_width = input_data.shape[1], input_data.shape[2], input_data.shape[3]
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        # Initialize output and mask
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.mask = np.zeros_like(input_data)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Input region indices
                        start_i = i * self.stride
                        end_i = start_i + self.pool_size
                        start_j = j * self.stride
                        end_j = start_j + self.pool_size
                        
                        # Extract region
                        region = input_data[b, c, start_i:end_i, start_j:end_j]
                        
                        # Find max value and its position
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val
                        
                        # Create mask for backpropagation
                        max_mask = (region == max_val)
                        self.mask[b, c, start_i:end_i, start_j:end_j] = max_mask
        
        if len(self.input_data.shape) == 3:
            output = output.squeeze(0)
        
        return output
    
    def backward(self, grad_output):
        """Backward pass through max pooling layer"""
        if len(self.input_data.shape) == 3:  # Single image
            grad_output = grad_output.reshape(1, *grad_output.shape)
        
        grad_input = np.zeros_like(self.input_data if len(self.input_data.shape) == 4 else self.input_data.reshape(1, *self.input_data.shape))
        
        batch_size, channels = grad_input.shape[0], grad_input.shape[1]
        output_height, output_width = grad_output.shape[2], grad_output.shape[3]
        
        # Distribute gradients
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Input region indices
                        start_i = i * self.stride
                        end_i = start_i + self.pool_size
                        start_j = j * self.stride
                        end_j = start_j + self.pool_size
                        
                        # Distribute gradient only to max positions
                        grad_input[b, c, start_i:end_i, start_j:end_j] += \
                            grad_output[b, c, i, j] * self.mask[b, c, start_i:end_i, start_j:end_j]
        
        if len(self.input_data.shape) == 3:
            grad_input = grad_input.squeeze(0)
        
        return grad_input

class ReLULayer:
    """ReLU activation layer"""
    
    def __init__(self):
        self.input_data = None
    
    def forward(self, input_data):
        """Forward pass through ReLU"""
        self.input_data = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output):
        """Backward pass through ReLU"""
        return grad_output * (self.input_data > 0)

class FlattenLayer:
    """Flatten layer to convert 2D feature maps to 1D"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_data):
        """Forward pass through flatten layer"""
        self.input_shape = input_data.shape
        
        if len(input_data.shape) == 3:  # Single image
            return input_data.flatten()
        else:  # Batch of images
            return input_data.reshape(input_data.shape[0], -1)
    
    def backward(self, grad_output):
        """Backward pass through flatten layer"""
        return grad_output.reshape(self.input_shape)

class DenseLayer:
    """Fully connected layer"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # For storing gradients
        self.dW = None
        self.db = None
        
        # For storing intermediate values
        self.input_data = None
    
    def forward(self, input_data):
        """Forward pass through dense layer"""
        self.input_data = input_data
        
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        output = np.dot(input_data, self.weights) + self.biases
        
        if output.shape[0] == 1:
            output = output.flatten()
        
        return output
    
    def backward(self, grad_output):
        """Backward pass through dense layer"""
        if len(self.input_data.shape) == 1:
            input_data = self.input_data.reshape(1, -1)
            grad_output = grad_output.reshape(1, -1)
        else:
            input_data = self.input_data
        
        # Compute gradients
        self.dW = np.dot(input_data.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        
        if len(self.input_data.shape) == 1:
            grad_input = grad_input.flatten()
        
        return grad_input

class SoftmaxLayer:
    """Softmax activation layer"""
    
    def __init__(self):
        self.output = None
    
    def forward(self, input_data):
        """Forward pass through softmax"""
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Subtract max for numerical stability
        shifted_input = input_data - np.max(input_data, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        if self.output.shape[0] == 1:
            self.output = self.output.flatten()
        
        return self.output
    
    def backward(self, grad_output):
        """Backward pass through softmax (combined with cross-entropy loss)"""
        return grad_output

class CNN:
    """Complete CNN implementation"""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        self.losses = []
        
        # Build default architecture
        self.build_model()
    
    def build_model(self):
        """Build CNN architecture"""
        # Conv Layer 1: 32 filters, 3x3 kernel
        self.layers.append(ConvolutionLayer(1, 32, 3, stride=1, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(2, stride=2))
        
        # Conv Layer 2: 64 filters, 3x3 kernel
        self.layers.append(ConvolutionLayer(32, 64, 3, stride=1, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(2, stride=2))
        
        # Flatten and Dense layers
        self.layers.append(FlattenLayer())
        
        # Calculate flattened size (for MNIST 28x28)
        flattened_size = 64 * 7 * 7  # After two 2x2 pooling operations: 28->14->7
        
        self.layers.append(DenseLayer(flattened_size, 128))
        self.layers.append(ReLULayer())
        self.layers.append(DenseLayer(128, self.num_classes))
        self.layers.append(SoftmaxLayer())
    
    def forward(self, input_data):
        """Forward pass through entire network"""
        current_input = input_data
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        return current_input
    
    def backward(self, y_true, y_pred):
        """Backward pass through entire network"""
        # Compute loss gradient (cross-entropy + softmax)
        if len(y_true.shape) == 0:  # Single sample
            grad_output = y_pred.copy()
            grad_output[y_true] -= 1
        else:  # Batch
            grad_output = y_pred.copy()
            grad_output[np.arange(len(y_true)), y_true] -= 1
            grad_output /= len(y_true)
        
        # Propagate gradients backward
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        epsilon = 1e-15  # Small value to prevent log(0)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if len(y_true.shape) == 0:  # Single sample
            return -np.log(y_pred_clipped[y_true])
        else:  # Batch
            return -np.mean(np.log(y_pred_clipped[np.arange(len(y_true)), y_true]))
    
    def update_weights(self, learning_rate):
        """Update weights using gradient descent"""
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learning_rate * layer.dW
                layer.biases -= learning_rate * layer.db
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.001, batch_size=32):
        """Train the CNN"""
        print(f"Training CNN for {epochs} epochs...")
        
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]
                
                # Process each sample in batch
                batch_loss = 0
                for i in range(len(batch_X)):
                    # Forward pass
                    y_pred = self.forward(batch_X[i])
                    
                    # Compute loss
                    loss = self.compute_loss(batch_y[i], y_pred)
                    batch_loss += loss
                    
                    # Backward pass
                    self.backward(batch_y[i], y_pred)
                    
                    # Check prediction
                    if np.argmax(y_pred) == batch_y[i]:
                        correct_predictions += 1
                
                # Update weights
                self.update_weights(learning_rate)
                epoch_loss += batch_loss
            
            # Calculate metrics
            avg_loss = epoch_loss / n_samples
            train_accuracy = correct_predictions / n_samples
            
            # Validation accuracy
            val_accuracy = self.evaluate(X_val, y_val)
            
            # Store metrics
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return train_losses, train_accuracies, val_accuracies
    
    def evaluate(self, X_test, y_test):
        """Evaluate the CNN"""
        correct_predictions = 0
        
        for i in range(len(X_test)):
            y_pred = self.forward(X_test[i])
            if np.argmax(y_pred) == y_test[i]:
                correct_predictions += 1
        
        return correct_predictions / len(X_test)
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        
        for i in range(len(X)):
            y_pred = self.forward(X[i])
            predictions.append(np.argmax(y_pred))
        
        return np.array(predictions)

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    # Reshape to (n_samples, height, width)
    X = X.reshape(-1, 28, 28)
    
    # Add channel dimension (n_samples, channels, height, width)
    X = X.reshape(-1, 1, 28, 28)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Use subset for faster training (remove this for full dataset)
    subset_size = 5000
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    
    X_val = X_val[:1000]
    y_val = y_val[:1000]
    
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def visualize_filters(cnn, layer_idx=0):
    """Visualize learned convolutional filters"""
    conv_layer = None
    current_idx = 0
    
    for layer in cnn.layers:
        if isinstance(layer, ConvolutionLayer):
            if current_idx == layer_idx:
                conv_layer = layer
                break
            current_idx += 1
    
    if conv_layer is None:
        print("No convolutional layer found at specified index")
        return
    
    filters = conv_layer.weights
    n_filters = min(16, filters.shape[0])  # Show first 16 filters
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_filters):
        # Get filter for first input channel
        filter_img = filters[i, 0]
        
        axes[i].imshow(filter_img, cmap='gray')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters, 16):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(cnn, input_image):
    """Visualize feature maps from convolutional layers"""
    current_input = input_image
    conv_outputs = []
    
    # Forward pass and collect conv layer outputs
    for layer in cnn.layers:
        current_input = layer.forward(current_input)
        if isinstance(layer, ConvolutionLayer):
            conv_outputs.append(current_input.copy())
    
    # Visualize feature maps from first conv layer
    if len(conv_outputs) > 0:
        feature_maps = conv_outputs[0]
        n_maps = min(16, feature_maps.shape[0])
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_maps):
            axes[i].imshow(feature_maps[i], cmap='gray')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_maps, 16):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def plot_training_history(train_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for CNN from scratch demonstration"""
    print("=== CNN from Scratch Implementation ===")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    
    # Create CNN
    input_shape = X_train.shape[1:]  # (channels, height, width)
    num_classes = len(np.unique(y_train))
    
    cnn = CNN(input_shape, num_classes)
    
    print(f"\nCNN Architecture:")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of layers: {len(cnn.layers)}")
    
    # Train model
    start_time = time.time()
    train_losses, train_accuracies, val_accuracies = cnn.train(
        X_train, y_train, X_val, y_val,
        epochs=5,  # Small number for demonstration
        learning_rate=0.001,
        batch_size=32
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = cnn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = cnn.predict(X_test[:100])  # Predict first 100 samples
    
    print("\nClassification Report (first 100 test samples):")
    print(classification_report(y_test[:100], y_pred))
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    
    # Visualize filters
    print("\nVisualizing learned filters...")
    visualize_filters(cnn, layer_idx=0)
    
    # Visualize feature maps
    print("\nVisualizing feature maps for a sample image...")
    sample_image = X_test[0]
    visualize_feature_maps(cnn, sample_image)
    
    # Show sample predictions
    plt.figure(figsize=(15, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i, 0], cmap='gray')
        plt.title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCNN from scratch implementation completed!")
    print("\nKey Components Implemented:")
    print("- Convolutional layers with forward/backward pass")
    print("- Max pooling layers")
    print("- ReLU activation")
    print("- Fully connected layers")
    print("- Softmax output layer")
    print("- Cross-entropy loss")
    print("- Gradient descent optimization")
    print("- Complete backpropagation algorithm")

if __name__ == "__main__":
    main()
