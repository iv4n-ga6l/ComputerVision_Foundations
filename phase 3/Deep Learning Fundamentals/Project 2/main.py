"""
Image Classification with Multi-Layer Perceptron (MLP)
======================================================

Train a multi-layer perceptron on CIFAR-10 dataset for image classification.
Demonstrates practical application of neural networks to computer vision tasks.


"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import argparse

class MLPImageClassifier:
    """Multi-Layer Perceptron for image classification"""
    
    def __init__(self, input_shape, num_classes, hidden_layers=[512, 256, 128]):
        """
        Initialize MLP classifier
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of output classes
            hidden_layers (list): List of hidden layer sizes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.model = None
        self.history = None
        
    def build_model(self, dropout_rate=0.3, batch_norm=True):
        """
        Build MLP model architecture
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            batch_norm (bool): Whether to use batch normalization
        """
        model = keras.Sequential([
            layers.Flatten(input_shape=self.input_shape[1:])  # Flatten images
        ])
        
        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation='relu'))
            
            if batch_norm:
                model.add(layers.BatchNormalization())
            
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """
        Compile the model
        
        Args:
            optimizer (str): Optimizer type
            learning_rate (float): Learning rate
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if recorded)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Final accuracy comparison
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        axes[1, 1].bar(['Training', 'Validation'], [final_train_acc, final_val_acc])
        axes[1, 1].set_title('Final Accuracy Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate([final_train_acc, final_val_acc]):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def visualize_predictions(self, X_test, y_true, y_pred, class_names, num_samples=16):
        """Visualize sample predictions"""
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            # Display image
            if len(X_test[idx].shape) == 3:  # Color image
                axes[i].imshow(X_test[idx])
            else:  # Grayscale
                axes[i].imshow(X_test[idx], cmap='gray')
            
            # Set title with prediction
            true_label = class_names[y_true[idx]]
            pred_label = class_names[y_pred[idx]]
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def load_cifar10_data():
    """Load and preprocess CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    
    return (X_train, y_train), (X_test, y_test), class_names

def load_fashion_mnist_data():
    """Load and preprocess Fashion-MNIST dataset"""
    print("Loading Fashion-MNIST dataset...")
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Normalize and add channel dimension
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(class_names)}")
    
    return (X_train, y_train), (X_test, y_test), class_names

def data_augmentation_demo(X_train, y_train):
    """Demonstrate data augmentation techniques"""
    print("\nDemonstrating data augmentation...")
    
    # Create data generator with augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Show original vs augmented images
    sample_image = X_train[0:1]
    
    plt.figure(figsize=(15, 3))
    
    # Original image
    plt.subplot(1, 6, 1)
    plt.imshow(sample_image[0])
    plt.title('Original')
    plt.axis('off')
    
    # Generate augmented versions
    datagen.fit(sample_image)
    for i, batch in enumerate(datagen.flow(sample_image, batch_size=1)):
        plt.subplot(1, 6, i + 2)
        plt.imshow(batch[0])
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
        
        if i >= 4:  # Show 5 augmented versions
            break
    
    plt.tight_layout()
    plt.show()
    
    return datagen

def compare_optimizers(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """Compare different optimizers"""
    print("\nComparing different optimizers...")
    
    optimizers = ['adam', 'sgd', 'rmsprop']
    results = {}
    
    for optimizer in optimizers:
        print(f"\nTraining with {optimizer.upper()} optimizer...")
        
        # Create new model for each optimizer
        classifier = MLPImageClassifier(input_shape, num_classes, hidden_layers=[256, 128])
        classifier.build_model(dropout_rate=0.3)
        classifier.compile_model(optimizer=optimizer)
        
        # Train for fewer epochs for comparison
        history = classifier.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=128)
        
        # Store results
        results[optimizer] = {
            'final_accuracy': history.history['val_accuracy'][-1],
            'history': history.history
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for optimizer in optimizers:
        plt.plot(results[optimizer]['history']['val_accuracy'], 
                label=f'{optimizer.upper()}')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for optimizer in optimizers:
        plt.plot(results[optimizer]['history']['val_loss'], 
                label=f'{optimizer.upper()}')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("\nOptimizer comparison results:")
    for optimizer in optimizers:
        acc = results[optimizer]['final_accuracy']
        print(f"{optimizer.upper()}: {acc:.4f}")

def main():
    """Main function for MLP image classification demo"""
    parser = argparse.ArgumentParser(description='MLP Image Classification')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'fashion_mnist'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size')
    parser.add_argument('--hidden_layers', nargs='+', type=int, 
                       default=[512, 256, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--compare_optimizers', action='store_true',
                       help='Compare different optimizers')
    parser.add_argument('--data_augmentation', action='store_true',
                       help='Show data augmentation demo')
    
    args = parser.parse_args()
    
    print("=== MLP Image Classification ===")
    
    # Load dataset
    if args.dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test), class_names = load_cifar10_data()
    else:
        (X_train, y_train), (X_test, y_test), class_names = load_fashion_mnist_data()
    
    # Split training data for validation
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    input_shape = X_train.shape
    num_classes = len(class_names)
    
    # Data augmentation demo
    if args.data_augmentation:
        datagen = data_augmentation_demo(X_train, y_train)
    
    # Compare optimizers
    if args.compare_optimizers:
        compare_optimizers(X_train, y_train, X_val, y_val, input_shape, num_classes)
        return
    
    # Create and train model
    print(f"\nBuilding MLP with hidden layers: {args.hidden_layers}")
    classifier = MLPImageClassifier(input_shape, num_classes, args.hidden_layers)
    
    # Build model
    model = classifier.build_model(dropout_rate=args.dropout_rate)
    print(f"Model created with {model.count_params():,} parameters")
    
    # Print model summary
    model.summary()
    
    # Compile model
    classifier.compile_model(optimizer=args.optimizer, learning_rate=args.learning_rate)
    
    # Train model
    print(f"\nTraining model for {args.epochs} epochs...")
    classifier.train(X_train, y_train, X_val, y_val, 
                    epochs=args.epochs, batch_size=args.batch_size)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = classifier.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, results['predictions'], 
                              target_names=class_names))
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(y_test, results['predictions'], class_names)
    
    # Visualize predictions
    classifier.visualize_predictions(X_test, y_test, results['predictions'], class_names)
    
    # Save model
    model_path = f'mlp_{args.dataset}_model.h5'
    classifier.save_model(model_path)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
