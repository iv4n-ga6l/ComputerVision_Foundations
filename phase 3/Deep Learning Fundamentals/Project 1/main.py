"""
Neural Network from Scratch
===========================

Complete implementation of a multi-layer perceptron without using deep learning frameworks.
Includes forward propagation, backpropagation, and various optimization techniques.


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

class ActivationFunction:
    """Base class for activation functions"""
    
    @staticmethod
    def forward(x):
        raise NotImplementedError
    
    @staticmethod
    def backward(x):
        raise NotImplementedError

class ReLU(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x):
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def backward(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(x):
        return 1 - np.tanh(x) ** 2

class Softmax(ActivationFunction):
    @staticmethod
    def forward(x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x):
        # For softmax, the derivative is computed in the loss function
        return np.ones_like(x)

class Layer:
    """Fully connected layer"""
    
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # Set activation function
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # For storing intermediate values during forward pass
        self.z = None
        self.a = None
        self.input = None
        
        # For gradients
        self.dW = None
        self.db = None

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, da):
        m = self.input.shape[0]
        
        # Compute gradients
        dz = da * self.activation.backward(self.z)
        self.dW = (1/m) * np.dot(self.input.T, dz)
        self.db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        # Gradient w.r.t. input (for previous layer)
        da_prev = np.dot(dz, self.weights.T)
        
        return da_prev

class NeuralNetwork:
    """Multi-layer perceptron implementation"""
    
    def __init__(self, layers_config, learning_rate=0.01):
        """
        Initialize neural network
        
        Args:
            layers_config: List of tuples (input_size, output_size, activation)
            learning_rate: Learning rate for optimization
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.costs = []
        
        # Create layers
        for config in layers_config:
            layer = Layer(*config)
            self.layers.append(layer)
    
    def forward(self, X):
        """Forward propagation through all layers"""
        current_input = X
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        return current_input
    
    def compute_cost(self, y_pred, y_true, loss_type='cross_entropy'):
        """Compute loss function"""
        m = y_true.shape[0]
        
        if loss_type == 'cross_entropy':
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            
            if y_true.shape[1] == 1:  # Binary classification
                cost = -(1/m) * np.sum(y_true * np.log(y_pred_clipped) + 
                                      (1 - y_true) * np.log(1 - y_pred_clipped))
            else:  # Multi-class classification
                cost = -(1/m) * np.sum(y_true * np.log(y_pred_clipped))
        
        elif loss_type == 'mse':
            cost = (1/(2*m)) * np.sum((y_pred - y_true) ** 2)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return cost
    
    def backward(self, y_pred, y_true, loss_type='cross_entropy'):
        """Backward propagation through all layers"""
        m = y_true.shape[0]
        
        # Compute gradient of loss w.r.t. output
        if loss_type == 'cross_entropy':
            if y_true.shape[1] == 1:  # Binary classification
                da = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
            else:  # Multi-class classification
                da = -(y_true / y_pred)
        elif loss_type == 'mse':
            da = (y_pred - y_true)
        
        # Propagate gradients backward through layers
        current_da = da
        for layer in reversed(self.layers):
            current_da = layer.backward(current_da)
    
    def update_parameters(self, optimizer='sgd', momentum=0.9):
        """Update parameters using specified optimizer"""
        if not hasattr(self, 'velocity'):
            self.velocity = [{'dW': np.zeros_like(layer.weights), 
                             'db': np.zeros_like(layer.biases)} 
                            for layer in self.layers]
        
        for i, layer in enumerate(self.layers):
            if optimizer == 'sgd':
                layer.weights -= self.learning_rate * layer.dW
                layer.biases -= self.learning_rate * layer.db
            
            elif optimizer == 'momentum':
                # Update velocity
                self.velocity[i]['dW'] = momentum * self.velocity[i]['dW'] + (1 - momentum) * layer.dW
                self.velocity[i]['db'] = momentum * self.velocity[i]['db'] + (1 - momentum) * layer.db
                
                # Update parameters
                layer.weights -= self.learning_rate * self.velocity[i]['dW']
                layer.biases -= self.learning_rate * self.velocity[i]['db']
    
    def train(self, X, y, epochs=1000, batch_size=32, validation_data=None, 
              loss_type='cross_entropy', optimizer='sgd', verbose=True):
        """Train the neural network"""
        
        m = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward propagation
                y_pred = self.forward(X_batch)
                
                # Compute cost
                batch_cost = self.compute_cost(y_pred, y_batch, loss_type)
                epoch_cost += batch_cost
                num_batches += 1
                
                # Backward propagation
                self.backward(y_pred, y_batch, loss_type)
                
                # Update parameters
                self.update_parameters(optimizer)
            
            # Average cost for epoch
            avg_cost = epoch_cost / num_batches
            self.costs.append(avg_cost)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {avg_cost:.6f}")
                
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_pred = self.forward(X_val)
                    val_accuracy = self.evaluate(X_val, y_val)
                    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        
        if y_pred.shape[1] == 1:  # Binary classification
            return (y_pred > 0.5).astype(int)
        else:  # Multi-class classification
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward(X)
    
    def evaluate(self, X, y):
        """Evaluate model accuracy"""
        predictions = self.predict(X)
        
        if y.shape[1] > 1:  # One-hot encoded
            y_true = np.argmax(y, axis=1)
        else:  # Already in class format
            y_true = y.flatten()
        
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        return accuracy_score(y_true, predictions)
    
    def plot_cost(self):
        """Plot training cost over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.costs)
        plt.title('Training Cost Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath):
        """Save model parameters"""
        model_data = {
            'layers_config': [(layer.input_size, layer.output_size, 'relu') for layer in self.layers],
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.biases for layer in self.layers],
            'learning_rate': self.learning_rate,
            'costs': self.costs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct layers
        self.layers = []
        for i, (weights, biases) in enumerate(zip(model_data['weights'], model_data['biases'])):
            input_size, output_size = weights.shape
            layer = Layer(input_size, output_size)
            layer.weights = weights
            layer.biases = biases
            self.layers.append(layer)
        
        self.learning_rate = model_data['learning_rate']
        self.costs = model_data['costs']
        
        print(f"Model loaded from {filepath}")

def prepare_data_for_classification():
    """Prepare sample classification dataset"""
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

def prepare_data_for_regression():
    """Prepare sample regression dataset"""
    # Generate synthetic regression data
    np.random.seed(42)
    m = 1000
    X = np.random.randn(m, 2)
    y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(m) * 0.1
    y = y.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def demo_classification():
    """Demonstrate neural network on classification task"""
    print("=== Classification Demo (Digits Dataset) ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_classification()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    # Define network architecture
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = y_train.shape[1]
    
    layers_config = [
        (input_size, hidden_size, 'relu'),
        (hidden_size, 64, 'relu'),
        (64, output_size, 'softmax')
    ]
    
    # Create and train model
    model = NeuralNetwork(layers_config, learning_rate=0.01)
    
    print("Training neural network...")
    model.train(X_train, y_train, epochs=1000, batch_size=32, 
                validation_data=(X_test, y_test),
                loss_type='cross_entropy', optimizer='momentum')
    
    # Evaluate model
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training cost
    model.plot_cost()
    
    # Save model
    model.save_model('classification_model.pkl')
    
    return model

def demo_regression():
    """Demonstrate neural network on regression task"""
    print("\n=== Regression Demo ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_regression()
    
    print(f"Training data shape: {X_train.shape}")
    
    # Define network architecture
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    
    layers_config = [
        (input_size, hidden_size, 'relu'),
        (hidden_size, 32, 'relu'),
        (32, output_size, 'sigmoid')  # Linear output for regression
    ]
    
    # Create and train model
    model = NeuralNetwork(layers_config, learning_rate=0.01)
    
    print("Training neural network...")
    model.train(X_train, y_train, epochs=500, batch_size=32,
                loss_type='mse', optimizer='momentum')
    
    # Make predictions
    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    
    # Calculate MSE
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)
    
    print(f"\nResults:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted (Test Set)')
    
    plt.subplot(1, 2, 2)
    model.plot_cost()
    
    plt.tight_layout()
    plt.show()
    
    return model

def main():
    """Main function to run neural network demos"""
    print("Neural Network from Scratch - Demo")
    print("=" * 40)
    
    # Run classification demo
    classification_model = demo_classification()
    
    # Run regression demo
    regression_model = demo_regression()
    
    print("\nDemo completed! Models saved.")
    print("Check the generated plots to see training progress.")

if __name__ == "__main__":
    main()
