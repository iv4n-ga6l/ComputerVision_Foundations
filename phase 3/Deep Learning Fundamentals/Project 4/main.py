"""
Regularization Techniques in Deep Learning
==========================================

Implement and compare various regularization techniques including dropout, 
batch normalization, L1/L2 regularization, and early stopping to prevent overfitting.

Author: Computer Vision Foundations
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from pathlib import Path

class RegularizationExperiment:
    """Class to experiment with different regularization techniques"""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128]):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.models = {}
        self.histories = {}
    
    def create_baseline_model(self):
        """Create baseline model without regularization"""
        model = keras.Sequential([
            layers.Dense(self.hidden_dims[0], activation='relu', input_dim=self.input_dim),
            layers.Dense(self.hidden_dims[1], activation='relu'),
            layers.Dense(self.hidden_dims[2], activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_dropout_model(self, dropout_rates=[0.3, 0.4, 0.5]):
        """Create model with dropout regularization"""
        model = keras.Sequential([
            layers.Dense(self.hidden_dims[0], activation='relu', input_dim=self.input_dim),
            layers.Dropout(dropout_rates[0]),
            layers.Dense(self.hidden_dims[1], activation='relu'),
            layers.Dropout(dropout_rates[1]),
            layers.Dense(self.hidden_dims[2], activation='relu'),
            layers.Dropout(dropout_rates[2]),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_batch_norm_model(self):
        """Create model with batch normalization"""
        model = keras.Sequential([
            layers.Dense(self.hidden_dims[0], input_dim=self.input_dim),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(self.hidden_dims[1]),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(self.hidden_dims[2]),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_l1_l2_model(self, l1_reg=0.01, l2_reg=0.01):
        """Create model with L1 and L2 regularization"""
        model = keras.Sequential([
            layers.Dense(self.hidden_dims[0], activation='relu', 
                        input_dim=self.input_dim,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.Dense(self.hidden_dims[1], activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.Dense(self.hidden_dims[2], activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def create_combined_model(self, dropout_rate=0.3, l1_reg=0.001, l2_reg=0.001):
        """Create model with combined regularization techniques"""
        model = keras.Sequential([
            layers.Dense(self.hidden_dims[0], input_dim=self.input_dim,
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(self.hidden_dims[1],
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(self.hidden_dims[2],
                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(dropout_rate),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def train_models(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train all models with different regularization techniques"""
        
        # Model configurations
        model_configs = {
            'Baseline': self.create_baseline_model,
            'Dropout': lambda: self.create_dropout_model([0.3, 0.4, 0.5]),
            'BatchNorm': self.create_batch_norm_model,
            'L1_L2': lambda: self.create_l1_l2_model(0.001, 0.001),
            'Combined': lambda: self.create_combined_model(0.3, 0.0005, 0.0005)
        }
        
        for name, model_func in model_configs.items():
            print(f"\nTraining {name} model...")
            
            # Create and compile model
            model = model_func()
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Store results
            self.models[name] = model
            self.histories[name] = history.history
            
            # Print final results
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            print(f"{name} - Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}")
    
    def plot_training_curves(self):
        """Plot training curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, history) in enumerate(self.histories.items()):
            color = colors[i % len(colors)]
            
            # Training accuracy
            axes[0, 0].plot(history['accuracy'], label=name, color=color)
            
            # Validation accuracy
            axes[0, 1].plot(history['val_accuracy'], label=name, color=color)
            
            # Training loss
            axes[1, 0].plot(history['loss'], label=name, color=color)
            
            # Validation loss
            axes[1, 1].plot(history['val_loss'], label=name, color=color)
        
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_overfitting_analysis(self):
        """Analyze overfitting for each model"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(self.histories.keys())
        train_accuracies = []
        val_accuracies = []
        overfitting_gaps = []
        
        for name in model_names:
            history = self.histories[name]
            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            overfitting_gap = final_train_acc - final_val_acc
            
            train_accuracies.append(final_train_acc)
            val_accuracies.append(final_val_acc)
            overfitting_gaps.append(overfitting_gap)
        
        # Final accuracy comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, train_accuracies, width, label='Training', alpha=0.8)
        axes[0].bar(x + width/2, val_accuracies, width, label='Validation', alpha=0.8)
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Final Training vs Validation Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Overfitting gap
        colors = ['red' if gap > 0.05 else 'green' for gap in overfitting_gaps]
        bars = axes[1].bar(model_names, overfitting_gaps, color=colors, alpha=0.7)
        
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Overfitting Gap (Train - Val Accuracy)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, gap in zip(bars, overfitting_gaps):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{gap:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test data"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            # Get predictions
            y_pred = np.argmax(model.predict(X_test), axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate loss
            test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
            
            results[name] = {
                'accuracy': accuracy,
                'loss': test_loss,
                'predictions': y_pred
            }
            
            print(f"{name:12}: Accuracy = {accuracy:.4f}, Loss = {test_loss:.4f}")
        
        print("="*60)
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest Model: {best_model}")
        print(f"Best Accuracy: {results[best_model]['accuracy']:.4f}")
        
        return results

class DropoutAnalysis:
    """Specific analysis for dropout regularization"""
    
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def compare_dropout_rates(self, X_train, y_train, X_val, y_val, 
                             dropout_rates=[0.0, 0.2, 0.3, 0.5, 0.7, 0.8]):
        """Compare different dropout rates"""
        results = {}
        
        print("\nComparing different dropout rates...")
        
        for rate in dropout_rates:
            print(f"Training with dropout rate: {rate}")
            
            # Create model
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_dim=self.input_dim),
                layers.Dropout(rate),
                layers.Dense(128, activation='relu'),
                layers.Dropout(rate),
                layers.Dense(64, activation='relu'),
                layers.Dropout(rate),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            results[rate] = {
                'history': history.history,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'overfitting_gap': history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
            }
        
        self.plot_dropout_comparison(results)
        return results
    
    def plot_dropout_comparison(self, results):
        """Plot dropout rate comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        dropout_rates = list(results.keys())
        train_accs = [results[rate]['final_train_acc'] for rate in dropout_rates]
        val_accs = [results[rate]['final_val_acc'] for rate in dropout_rates]
        overfitting_gaps = [results[rate]['overfitting_gap'] for rate in dropout_rates]
        
        # Training vs Validation accuracy
        axes[0].plot(dropout_rates, train_accs, 'o-', label='Training', linewidth=2)
        axes[0].plot(dropout_rates, val_accs, 'o-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Dropout Rate')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy vs Dropout Rate')
        axes[0].legend()
        axes[0].grid(True)
        
        # Overfitting gap
        axes[1].plot(dropout_rates, overfitting_gaps, 'ro-', linewidth=2)
        axes[1].set_xlabel('Dropout Rate')
        axes[1].set_ylabel('Overfitting Gap (Train - Val)')
        axes[1].set_title('Overfitting vs Dropout Rate')
        axes[1].grid(True)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Validation accuracy curves for different dropout rates
        for rate in [0.0, 0.3, 0.5, 0.8]:
            if rate in results:
                history = results[rate]['history']
                axes[2].plot(history['val_accuracy'], label=f'Dropout {rate}')
        
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Validation Accuracy')
        axes[2].set_title('Training Curves for Different Dropout Rates')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

class BatchNormalizationAnalysis:
    """Specific analysis for batch normalization"""
    
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def compare_with_without_batchnorm(self, X_train, y_train, X_val, y_val):
        """Compare models with and without batch normalization"""
        print("\nComparing models with and without batch normalization...")
        
        # Model without batch normalization
        model_without = keras.Sequential([
            layers.Dense(512, activation='relu', input_dim=self.input_dim),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Model with batch normalization
        model_with = keras.Sequential([
            layers.Dense(512, input_dim=self.input_dim),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        models = {'Without BatchNorm': model_without, 'With BatchNorm': model_with}
        histories = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            histories[name] = history.history
        
        self.plot_batchnorm_comparison(histories)
        return histories
    
    def plot_batchnorm_comparison(self, histories):
        """Plot batch normalization comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for name, history in histories.items():
            color = 'blue' if 'Without' in name else 'red'
            
            axes[0, 0].plot(history['accuracy'], label=name, color=color)
            axes[0, 1].plot(history['val_accuracy'], label=name, color=color)
            axes[1, 0].plot(history['loss'], label=name, color=color)
            axes[1, 1].plot(history['val_loss'], label=name, color=color)
        
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def create_overfitting_dataset():
    """Create a dataset prone to overfitting"""
    print("Creating dataset prone to overfitting...")
    
    # Generate complex classification data with noise
    X, y = make_classification(
        n_samples=1000,  # Small dataset
        n_features=50,   # Many features
        n_informative=10,
        n_redundant=10,
        n_classes=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    # Add noise features
    noise = np.random.randn(X.shape[0], 20) * 0.1
    X = np.concatenate([X, noise], axis=1)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function for regularization techniques demonstration"""
    print("=== Regularization Techniques in Deep Learning ===")
    
    # Create overfitting-prone dataset
    X_train, X_val, X_test, y_train, y_val, y_test = create_overfitting_dataset()
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Main regularization experiment
    print("\n=== Main Regularization Experiment ===")
    experiment = RegularizationExperiment(input_dim, num_classes)
    experiment.train_models(X_train, y_train, X_val, y_val, epochs=100)
    
    # Plot results
    experiment.plot_training_curves()
    experiment.plot_overfitting_analysis()
    
    # Evaluate on test set
    test_results = experiment.evaluate_models(X_test, y_test)
    
    # Dropout analysis
    print("\n=== Dropout Rate Analysis ===")
    dropout_analysis = DropoutAnalysis(input_dim, num_classes)
    dropout_results = dropout_analysis.compare_dropout_rates(X_train, y_train, X_val, y_val)
    
    # Batch normalization analysis
    print("\n=== Batch Normalization Analysis ===")
    batchnorm_analysis = BatchNormalizationAnalysis(input_dim, num_classes)
    batchnorm_results = batchnorm_analysis.compare_with_without_batchnorm(X_train, y_train, X_val, y_val)
    
    # Summary
    print("\n" + "="*80)
    print("REGULARIZATION TECHNIQUES SUMMARY")
    print("="*80)
    print("\n1. DROPOUT:")
    print("   - Prevents overfitting by randomly setting neurons to zero")
    print("   - Optimal rate typically between 0.2-0.5")
    print("   - Higher rates may hurt performance")
    
    print("\n2. BATCH NORMALIZATION:")
    print("   - Normalizes inputs to each layer")
    print("   - Accelerates training and acts as regularizer")
    print("   - Reduces internal covariate shift")
    
    print("\n3. L1/L2 REGULARIZATION:")
    print("   - L1: Promotes sparsity in weights")
    print("   - L2: Prevents large weights")
    print("   - Controls model complexity")
    
    print("\n4. EARLY STOPPING:")
    print("   - Stops training when validation performance stops improving")
    print("   - Prevents overfitting to training data")
    print("   - Simple and effective technique")
    
    print("\n5. COMBINED APPROACHES:")
    print("   - Multiple techniques often work better together")
    print("   - Balance between regularization and model capacity")
    print("   - Experiment to find optimal combination")
    
    print("\nRegularization techniques comparison completed!")

if __name__ == "__main__":
    main()
