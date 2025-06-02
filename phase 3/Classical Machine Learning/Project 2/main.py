"""
SVM Image Classification with HOG Features
==========================================

This project implements a complete pipeline for image classification using:
- HOG (Histogram of Oriented Gradients) feature extraction
- SVM (Support Vector Machine) classification
- Cross-validation and model evaluation


"""

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class HOGSVMClassifier:
    def __init__(self, hog_params=None):
        """
        Initialize the HOG-SVM classifier
        
        Args:
            hog_params (dict): HOG descriptor parameters
        """
        # Default HOG parameters
        default_hog = {
            'winSize': (64, 64),
            'blockSize': (16, 16),
            'blockStride': (8, 8),
            'cellSize': (8, 8),
            'nbins': 9
        }
        
        if hog_params:
            default_hog.update(hog_params)
        
        self.hog = cv2.HOGDescriptor(
            _winSize=default_hog['winSize'],
            _blockSize=default_hog['blockSize'],
            _blockStride=default_hog['blockStride'],
            _cellSize=default_hog['cellSize'],
            _nbins=default_hog['nbins']
        )
        
        self.svm = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.is_trained = False
    
    def extract_hog_features(self, image):
        """
        Extract HOG features from an image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Flattened HOG feature vector
        """
        # Resize image to standard size
        image_resized = cv2.resize(image, self.hog.winSize)
        
        # Convert to grayscale if needed
        if len(image_resized.shape) == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_resized
        
        # Extract HOG features
        features = self.hog.compute(image_gray)
        return features.flatten()
    
    def visualize_hog(self, image):
        """
        Visualize HOG features for an image
        
        Args:
            image (np.ndarray): Input image
        """
        # Resize and convert to grayscale
        image_resized = cv2.resize(image, self.hog.winSize)
        if len(image_resized.shape) == 3:
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image_resized
        
        # Compute HOG features and visualization
        features, hog_image = cv2.HOGDescriptor.compute(
            self.hog, image_gray, visualize=True
        )
        
        # Display results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_resized, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Visualization')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.plot(features[:100])  # Plot first 100 features
        plt.title('HOG Feature Vector (first 100)')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        
        plt.tight_layout()
        plt.show()
    
    def load_dataset(self, dataset_path):
        """
        Load images from directory structure
        
        Args:
            dataset_path (str): Path to dataset directory
            
        Returns:
            tuple: Features array, labels array, class names
        """
        features = []
        labels = []
        self.class_names = []
        
        dataset_path = Path(dataset_path)
        
        print("Loading dataset...")
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_names.append(class_name)
                print(f"Loading class: {class_name}")
                
                for image_file in class_dir.glob('*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image = cv2.imread(str(image_file))
                        
                        if image is not None:
                            hog_features = self.extract_hog_features(image)
                            features.append(hog_features)
                            labels.append(class_name)
        
        print(f"Loaded {len(features)} images from {len(self.class_names)} classes")
        return np.array(features), np.array(labels)
    
    def train(self, X, y, test_size=0.2, use_grid_search=True):
        """
        Train the SVM classifier
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Proportion of test data
            use_grid_search (bool): Whether to use grid search for hyperparameters
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training SVM classifier...")
        
        if use_grid_search:
            # Grid search for best hyperparameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            
            self.svm = GridSearchCV(
                SVC(random_state=42),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:
            self.svm = SVC(kernel='rbf', random_state=42)
        
        # Train the model
        self.svm.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        if use_grid_search:
            print(f"Best parameters: {self.svm.best_params_}")
        
        # Classification report
        target_names = [self.class_names[i] for i in range(len(self.class_names))]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def predict(self, image):
        """
        Predict class for a single image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: Predicted class name and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_hog_features(image).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.svm.predict(features_scaled)[0]
        probabilities = self.svm.predict_proba(features_scaled)[0] if hasattr(self.svm, 'predict_proba') else None
        
        # Convert back to class name
        class_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return class_name, probabilities
    
    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'hog_params': {
                'winSize': self.hog.winSize,
                'blockSize': self.hog.blockSize,
                'blockStride': self.hog.blockStride,
                'cellSize': self.hog.cellSize,
                'nbins': self.hog.nbins
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.class_names = model_data['class_names']
        
        # Reconstruct HOG descriptor
        hog_params = model_data['hog_params']
        self.hog = cv2.HOGDescriptor(
            _winSize=hog_params['winSize'],
            _blockSize=hog_params['blockSize'],
            _blockStride=hog_params['blockStride'],
            _cellSize=hog_params['cellSize'],
            _nbins=hog_params['nbins']
        )
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    print("Creating sample dataset...")
    
    # Create basic geometric shapes as sample data
    os.makedirs('sample_dataset/circles', exist_ok=True)
    os.makedirs('sample_dataset/rectangles', exist_ok=True)
    os.makedirs('sample_dataset/triangles', exist_ok=True)
    
    # Generate sample images (circles, rectangles, triangles)
    for i in range(50):
        # Circle
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(img, (32, 32), 20, (255, 255, 255), -1)
        cv2.imwrite(f'sample_dataset/circles/circle_{i}.png', img)
        
        # Rectangle
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(img, (15, 15), (49, 49), (255, 255, 255), -1)
        cv2.imwrite(f'sample_dataset/rectangles/rectangle_{i}.png', img)
        
        # Triangle
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        points = np.array([[32, 10], [10, 50], [54, 50]], np.int32)
        cv2.fillPoly(img, [points], (255, 255, 255))
        cv2.imwrite(f'sample_dataset/triangles/triangle_{i}.png', img)
    
    print("Sample dataset created in 'sample_dataset' directory")

def main():
    """Main function to demonstrate HOG-SVM classification"""
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists('sample_dataset'):
        create_sample_dataset()
    
    # Initialize classifier
    classifier = HOGSVMClassifier()
    
    # Load dataset
    try:
        X, y = classifier.load_dataset('sample_dataset')
        
        # Show HOG visualization for first image
        sample_image = cv2.imread('sample_dataset/circles/circle_0.png')
        if sample_image is not None:
            print("\nVisualizing HOG features for sample image...")
            classifier.visualize_hog(sample_image)
        
        # Train the classifier
        accuracy = classifier.train(X, y, use_grid_search=True)
        
        # Save the model
        classifier.save_model('hog_svm_model.pkl')
        
        # Test prediction on a single image
        test_image = cv2.imread('sample_dataset/circles/circle_0.png')
        if test_image is not None:
            predicted_class, probabilities = classifier.predict(test_image)
            print(f"\nPrediction for test image: {predicted_class}")
            if probabilities is not None:
                for i, prob in enumerate(probabilities):
                    print(f"{classifier.class_names[i]}: {prob:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your dataset follows the required directory structure")

if __name__ == "__main__":
    main()
