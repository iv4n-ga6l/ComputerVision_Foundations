"""
Optimization Algorithms Comparison
==================================

Compare different optimization algorithms for neural network training, including
SGD, Adam, RMSprop, and custom implementations with various learning strategies.

Author: Computer Vision Foundations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from pathlib import Path

class CustomOptimizer:
    """Base class for custom optimizer implementations"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.t = 0  # Time step
    
    def update(self, weights, gradients):
        """Update weights based on gradients"""
        raise NotImplementedError

class SGDOptimizer(CustomOptimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, weights, gradients):
        """Update weights using SGD with momentum"""
        updated_weights = {}
        
        for key in weights:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(weights[key])
            
            # Update velocity
            self.velocity[key] = (self.momentum * self.velocity[key] - 
                                self.learning_rate * gradients[key])
            
            # Update weights
            updated_weights[key] = weights[key] + self.velocity[key]
        
        return updated_weights

class AdamOptimizer(CustomOptimizer):
    """Adam optimizer implementation"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
    
    def update(self, weights, gradients):
        """Update weights using Adam algorithm"""
        self.t += 1
        updated_weights = {}
        
        for key in weights:
            if key not in self.m:
                self.m[key] = np.zeros_like(weights[key])
                self.v[key] = np.zeros_like(weights[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update weights
            updated_weights[key] = (weights[key] - 
                                  self.learning_rate * m_corrected / 
                                  (np.sqrt(v_corrected) + self.epsilon))
        
        return updated_weights

class RMSpropOptimizer(CustomOptimizer):
    """RMSprop optimizer implementation"""
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, weights, gradients):
        """Update weights using RMSprop algorithm"""
        updated_weights = {}
        
        for key in weights:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(weights[key])
            
            # Update cache
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (gradients[key] ** 2)
            
            # Update weights
            updated_weights[key] = (weights[key] - 
                                  self.learning_rate * gradients[key] / 
                                  (np.sqrt(self.cache[key]) + self.epsilon))
        
        return updated_weights

class OptimizationExperiment:
    """Class to run optimization experiments"""
    
    def __init__(self, model_type='classification'):
        self.model_type = model_type
        self.results = {}
    
    def create_simple_model(self, input_dim, output_dim, hidden_dims=[64, 32]):
        """Create a simple neural network model"""
        model = keras.Sequential()
        model.add(layers.Dense(hidden_dims[0], activation='relu', input_dim=input_dim))
        
        for dim in hidden_dims[1:]:
            model.add(layers.Dense(dim, activation='relu'))
        
        if self.model_type == 'classification':
            if output_dim == 1:
                model.add(layers.Dense(output_dim, activation='sigmoid'))
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
        else:  # regression
            model.add(layers.Dense(output_dim, activation='linear'))
        
        return model
    
    def compile_model(self, model, optimizer_name, learning_rate=0.001):
        """Compile model with specified optimizer"""
        if optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        if self.model_type == 'classification':
            loss = 'sparse_categorical_crossentropy' if len(np.unique(model.output_shape[-1])) > 2 else 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def run_experiment(self, X_train, y_train, X_val, y_val, 
                      optimizers=['sgd', 'adam', 'rmsprop'], 
                      learning_rates=[0.001, 0.01, 0.1],
                      epochs=100, batch_size=32):
        """Run optimization experiment"""
        
        input_dim = X_train.shape[1]
        if self.model_type == 'classification':
            output_dim = len(np.unique(y_train))
        else:
            output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        print(f"Running experiment with {len(optimizers)} optimizers and {len(learning_rates)} learning rates")
        
        for optimizer_name in optimizers:
            self.results[optimizer_name] = {}
            
            for lr in learning_rates:
                print(f"Training with {optimizer_name} (lr={lr})...")
                
                # Create and compile model
                model = self.create_simple_model(input_dim, output_dim)
                model = self.compile_model(model, optimizer_name, lr)
                
                # Train model
                start_time = time.time()
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                training_time = time.time() - start_time
                
                # Store results
                self.results[optimizer_name][lr] = {
                    'history': history.history,
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'training_time': training_time,
                    'model': model
                }
                
                if self.model_type == 'classification':
                    self.results[optimizer_name][lr]['final_train_acc'] = history.history['accuracy'][-1]
                    self.results[optimizer_name][lr]['final_val_acc'] = history.history['val_accuracy'][-1]
    
    def plot_convergence_curves(self):
        """Plot convergence curves for different optimizers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].set_title('Training Loss')
        axes[0, 1].set_title('Validation Loss')
        
        if self.model_type == 'classification':
            axes[1, 0].set_title('Training Accuracy')
            axes[1, 1].set_title('Validation Accuracy')
        else:
            axes[1, 0].set_title('Training MAE')
            axes[1, 1].set_title('Validation MAE')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (optimizer, opt_results) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            
            # Find best learning rate based on validation loss
            best_lr = min(opt_results.keys(), 
                         key=lambda lr: opt_results[lr]['final_val_loss'])
            
            history = opt_results[best_lr]['history']
            
            # Plot loss curves
            axes[0, 0].plot(history['loss'], label=f'{optimizer} (lr={best_lr})', 
                          color=color, linestyle='-')
            axes[0, 1].plot(history['val_loss'], label=f'{optimizer} (lr={best_lr})', 
                          color=color, linestyle='-')
            
            # Plot accuracy/MAE curves
            if self.model_type == 'classification':
                axes[1, 0].plot(history['accuracy'], label=f'{optimizer} (lr={best_lr})', 
                              color=color, linestyle='-')
                axes[1, 1].plot(history['val_accuracy'], label=f'{optimizer} (lr={best_lr})', 
                              color=color, linestyle='-')
            else:
                axes[1, 0].plot(history['mae'], label=f'{optimizer} (lr={best_lr})', 
                              color=color, linestyle='-')
                axes[1, 1].plot(history['val_mae'], label=f'{optimizer} (lr={best_lr})', 
                              color=color, linestyle='-')
        
        # Set labels and legends
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_rate_sensitivity(self):
        """Plot learning rate sensitivity for each optimizer"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        optimizers = list(self.results.keys())
        learning_rates = list(self.results[optimizers[0]].keys())
        
        # Plot final validation loss vs learning rate
        for optimizer in optimizers:
            val_losses = [self.results[optimizer][lr]['final_val_loss'] 
                         for lr in learning_rates]
            axes[0].plot(learning_rates, val_losses, 'o-', label=optimizer)
        
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Final Validation Loss')
        axes[0].set_xscale('log')
        axes[0].set_title('Learning Rate Sensitivity - Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot training time vs learning rate
        for optimizer in optimizers:
            training_times = [self.results[optimizer][lr]['training_time'] 
                             for lr in learning_rates]
            axes[1].plot(learning_rates, training_times, 'o-', label=optimizer)
        
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_xscale('log')
        axes[1].set_title('Training Time vs Learning Rate')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_table(self):
        """Print summary table of results"""
        print("\n" + "="*80)
        print("OPTIMIZATION ALGORITHM COMPARISON SUMMARY")
        print("="*80)
        
        print(f"{'Optimizer':<12} {'Learning Rate':<15} {'Val Loss':<12} {'Train Time':<12}", end="")
        if self.model_type == 'classification':
            print(f" {'Val Accuracy':<12}")
        else:
            print()
        
        print("-"*80)
        
        for optimizer in self.results:
            for lr in self.results[optimizer]:
                result = self.results[optimizer][lr]
                print(f"{optimizer:<12} {lr:<15.4f} {result['final_val_loss']:<12.4f} "
                      f"{result['training_time']:<12.2f}s", end="")
                
                if self.model_type == 'classification':
                    print(f" {result['final_val_acc']:<12.4f}")
                else:
                    print()
        
        print("-"*80)
        
        # Best results
        print("\nBEST RESULTS:")
        best_optimizer = None
        best_lr = None
        best_val_loss = float('inf')
        
        for optimizer in self.results:
            for lr in self.results[optimizer]:
                val_loss = self.results[optimizer][lr]['final_val_loss']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_optimizer = optimizer
                    best_lr = lr
        
        best_result = self.results[best_optimizer][best_lr]
        print(f"Best Optimizer: {best_optimizer}")
        print(f"Best Learning Rate: {best_lr}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        if self.model_type == 'classification':
            print(f"Best Validation Accuracy: {best_result['final_val_acc']:.4f}")

def load_classification_data():
    """Load classification dataset"""
    print("Loading classification dataset (Digits)...")
    
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_regression_data():
    """Load regression dataset"""
    print("Loading regression dataset...")
    
    # Generate synthetic regression data
    X, y = make_regression(n_samples=2000, n_features=20, noise=0.1, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compare_custom_optimizers():
    """Compare custom optimizer implementations with simple example"""
    print("\n=== Custom Optimizer Implementations Demo ===")
    
    # Simple quadratic function: f(x) = (x-2)^2 + (y-1)^2
    def quadratic_function(x, y):
        return (x - 2)**2 + (y - 1)**2
    
    def quadratic_gradients(x, y):
        dx = 2 * (x - 2)
        dy = 2 * (y - 1)
        return dx, dy
    
    # Initialize optimizers
    optimizers = {
        'SGD': SGDOptimizer(learning_rate=0.1),
        'SGD+Momentum': SGDOptimizer(learning_rate=0.1, momentum=0.9),
        'Adam': AdamOptimizer(learning_rate=0.1),
        'RMSprop': RMSpropOptimizer(learning_rate=0.1)
    }
    
    # Starting point
    start_x, start_y = 0.0, 0.0
    n_iterations = 100
    
    # Track optimization paths
    paths = {}
    
    for name, optimizer in optimizers.items():
        path = [(start_x, start_y)]
        x, y = start_x, start_y
        
        for i in range(n_iterations):
            # Compute gradients
            dx, dy = quadratic_gradients(x, y)
            
            # Create weights and gradients dictionaries for optimizer
            weights = {'x': np.array([x]), 'y': np.array([y])}
            gradients = {'x': np.array([dx]), 'y': np.array([dy])}
            
            # Update using optimizer
            updated = optimizer.update(weights, gradients)
            x, y = updated['x'][0], updated['y'][0]
            
            path.append((x, y))
        
        paths[name] = path
    
    # Plot optimization paths
    plt.figure(figsize=(12, 8))
    
    # Create contour plot of the function
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = quadratic_function(X, Y)
    
    plt.contour(X, Y, Z, levels=20, alpha=0.5)
    
    # Plot optimization paths
    colors = ['blue', 'red', 'green', 'orange']
    for i, (name, path) in enumerate(paths.items()):
        x_path = [point[0] for point in path]
        y_path = [point[1] for point in path]
        
        plt.plot(x_path, y_path, 'o-', color=colors[i], 
                label=name, markersize=3, linewidth=2)
        
        # Mark starting and ending points
        plt.plot(x_path[0], y_path[0], 's', color=colors[i], markersize=8)
        plt.plot(x_path[-1], y_path[-1], '*', color=colors[i], markersize=12)
    
    # Mark global minimum
    plt.plot(2, 1, 'ko', markersize=10, label='Global Minimum')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization Paths Comparison\n(Quadratic Function: f(x,y) = (x-2)² + (y-1)²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    # Print final results
    print("\nFinal positions after optimization:")
    for name, path in paths.items():
        final_x, final_y = path[-1]
        final_value = quadratic_function(final_x, final_y)
        distance_to_optimum = np.sqrt((final_x - 2)**2 + (final_y - 1)**2)
        
        print(f"{name:15}: ({final_x:.4f}, {final_y:.4f}), "
              f"f = {final_value:.6f}, "
              f"distance = {distance_to_optimum:.6f}")

def main():
    """Main function for optimization algorithms comparison"""
    print("=== Optimization Algorithms Comparison ===")
    
    # Custom optimizers demo
    compare_custom_optimizers()
    
    # Classification experiment
    print("\n=== Classification Task Comparison ===")
    X_train, X_val, X_test, y_train, y_val, y_test = load_classification_data()
    
    classification_exp = OptimizationExperiment(model_type='classification')
    classification_exp.run_experiment(
        X_train, y_train, X_val, y_val,
        optimizers=['sgd', 'adam', 'rmsprop', 'adagrad'],
        learning_rates=[0.001, 0.01, 0.1],
        epochs=50,
        batch_size=32
    )
    
    classification_exp.plot_convergence_curves()
    classification_exp.plot_learning_rate_sensitivity()
    classification_exp.print_summary_table()
    
    # Regression experiment
    print("\n=== Regression Task Comparison ===")
    X_train, X_val, X_test, y_train, y_val, y_test = load_regression_data()
    
    regression_exp = OptimizationExperiment(model_type='regression')
    regression_exp.run_experiment(
        X_train, y_train, X_val, y_val,
        optimizers=['sgd', 'adam', 'rmsprop', 'adagrad'],
        learning_rates=[0.001, 0.01, 0.1],
        epochs=50,
        batch_size=32
    )
    
    regression_exp.plot_convergence_curves()
    regression_exp.plot_learning_rate_sensitivity()
    regression_exp.print_summary_table()
    
    print("\nOptimization algorithms comparison completed!")
    print("\nKey Insights:")
    print("- Adam generally converges faster and more stable")
    print("- SGD with momentum can achieve better final performance")
    print("- RMSprop works well for non-stationary objectives")
    print("- Learning rate is crucial for all optimizers")
    print("- Different optimizers may be optimal for different tasks")

if __name__ == "__main__":
    main()
