"""
Random Forest Texture Classification
=====================================

This project implements texture classification using Random Forest with:
- Multiple texture descriptors (LBP, GLCM, Gabor, Wavelet)
- Feature combination and selection
- Ensemble learning with Random Forest
- Real-time texture analysis


"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy import ndimage
import pywt
import os
import pickle
import seaborn as sns
from pathlib import Path

class TextureClassifier:
    def __init__(self, n_estimators=100):
        """
        Initialize Random Forest texture classifier
        
        Args:
            n_estimators (int): Number of trees in the forest
        """
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.class_names = []
        self.is_trained = False
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """
        Extract Local Binary Pattern features
        
        Args:
            image (np.ndarray): Grayscale image
            radius (int): Radius of circle
            n_points (int): Number of points on circle
            
        Returns:
            np.ndarray: LBP histogram features
        """
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                              range=(0, n_points + 2), density=True)
        return hist
    
    def extract_glcm_features(self, image, distances=[5], angles=[0, 45, 90, 135]):
        """
        Extract Gray-Level Co-occurrence Matrix features
        
        Args:
            image (np.ndarray): Grayscale image
            distances (list): Pixel distances
            angles (list): Angles in degrees
            
        Returns:
            np.ndarray: GLCM texture features
        """
        # Convert angles to radians
        angles_rad = [np.radians(angle) for angle in angles]
        
        # Compute GLCM
        glcm = graycomatrix(image, distances, angles_rad, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.flatten())
        
        return np.array(features)
    
    def extract_gabor_features(self, image, frequencies=[0.1, 0.3, 0.5], 
                              angles=[0, 45, 90, 135]):
        """
        Extract Gabor filter features
        
        Args:
            image (np.ndarray): Grayscale image
            frequencies (list): Spatial frequencies
            angles (list): Orientations in degrees
            
        Returns:
            np.ndarray: Gabor filter responses
        """
        features = []
        
        for freq in frequencies:
            for angle in angles:
                theta = np.radians(angle)
                kernel = gabor_kernel(freq, theta=theta)
                
                # Apply filter
                filtered = ndimage.convolve(image, kernel, mode='wrap')
                
                # Extract statistics
                features.extend([
                    np.mean(np.abs(filtered)),
                    np.var(filtered)
                ])
        
        return np.array(features)
    
    def extract_wavelet_features(self, image, wavelet='db4', levels=3):
        """
        Extract wavelet transform features
        
        Args:
            image (np.ndarray): Grayscale image
            wavelet (str): Wavelet type
            levels (int): Decomposition levels
            
        Returns:
            np.ndarray: Wavelet features
        """
        features = []
        
        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet, level=levels)
        
        # Extract statistics from each subband
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Approximation coefficients
                features.extend([
                    np.mean(coeff),
                    np.var(coeff),
                    np.std(coeff)
                ])
            else:  # Detail coefficients (LH, HL, HH)
                for subband in coeff:
                    features.extend([
                        np.mean(np.abs(subband)),
                        np.var(subband),
                        np.std(subband)
                    ])
        
        return np.array(features)
    
    def extract_all_features(self, image):
        """
        Extract all texture features from an image
        
        Args:
            image (np.ndarray): Input image (color or grayscale)
            
        Returns:
            np.ndarray: Combined feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Extract different types of features
        lbp_features = self.extract_lbp_features(gray)
        glcm_features = self.extract_glcm_features(gray)
        gabor_features = self.extract_gabor_features(gray)
        wavelet_features = self.extract_wavelet_features(gray)
        
        # Combine all features
        all_features = np.concatenate([
            lbp_features,
            glcm_features,
            gabor_features,
            wavelet_features
        ])
        
        return all_features
    
    def load_dataset(self, dataset_path, target_size=(128, 128)):
        """
        Load texture dataset from directory structure
        
        Args:
            dataset_path (str): Path to dataset directory
            target_size (tuple): Target image size
            
        Returns:
            tuple: Features array and labels array
        """
        features = []
        labels = []
        self.class_names = []
        
        dataset_path = Path(dataset_path)
        
        print("Loading texture dataset...")
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_names.append(class_name)
                print(f"Loading textures for class: {class_name}")
                
                for image_file in class_dir.glob('*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image = cv2.imread(str(image_file))
                        
                        if image is not None:
                            # Resize image
                            image_resized = cv2.resize(image, target_size)
                            
                            # Extract features
                            texture_features = self.extract_all_features(image_resized)
                            
                            features.append(texture_features)
                            labels.append(class_name)
        
        print(f"Loaded {len(features)} texture samples from {len(self.class_names)} classes")
        
        return np.array(features), np.array(labels)
    
    def train(self, X, y, test_size=0.2):
        """
        Train the Random Forest classifier
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            test_size (float): Proportion of test data
            
        Returns:
            float: Test accuracy
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
        
        print("Training Random Forest classifier...")
        
        # Train the model
        self.rf_classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_classifier, X_train_scaled, y_train, 
                                   cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Evaluate on test set
        y_pred = self.rf_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Feature importance
        self.analyze_feature_importance()
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return accuracy
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing features")
        
        # Get feature importance
        importance = self.rf_classifier.feature_importances_
        
        # Create feature names
        if not self.feature_names:
            n_lbp = 26  # LBP features
            n_glcm = 16  # GLCM features (4 properties × 4 angles)
            n_gabor = 24  # Gabor features (3 freq × 4 angles × 2 stats)
            n_wavelet = len(importance) - n_lbp - n_glcm - n_gabor
            
            self.feature_names = (
                [f'LBP_{i}' for i in range(n_lbp)] +
                [f'GLCM_{i}' for i in range(n_glcm)] +
                [f'Gabor_{i}' for i in range(n_gabor)] +
                [f'Wavelet_{i}' for i in range(n_wavelet)]
            )
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot top features
        n_top = min(20, len(importance))
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {n_top} Feature Importances')
        plt.bar(range(n_top), importance[indices[:n_top]])
        plt.xticks(range(n_top), [self.feature_names[i] for i in indices[:n_top]], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Print top features
        print(f"\nTop {n_top} most important features:")
        for i in range(n_top):
            idx = indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:15s}: {importance[idx]:.4f}")
    
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
        Predict texture class for a single image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: Predicted class name and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_all_features(image).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.rf_classifier.predict(features_scaled)[0]
        probabilities = self.rf_classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Convert back to class name
        class_name = self.label_encoder.inverse_transform([prediction])[0]
        
        return class_name, confidence
    
    def visualize_texture_features(self, image):
        """
        Visualize different texture features for an image
        
        Args:
            image (np.ndarray): Input image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute different representations
        lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        
        # Gabor filter (example with one filter)
        kernel = gabor_kernel(0.3, theta=0)
        gabor_response = ndimage.convolve(gray, kernel, mode='wrap')
        
        # Wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db4')
        cA, (cH, cV, cD) = coeffs
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lbp, cmap='gray')
        axes[0, 1].set_title('Local Binary Pattern')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.abs(gabor_response), cmap='gray')
        axes[0, 2].set_title('Gabor Filter Response')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(cA, cmap='gray')
        axes[1, 0].set_title('Wavelet Approximation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cH, cmap='gray')
        axes[1, 1].set_title('Wavelet Horizontal Detail')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(cV, cmap='gray')
        axes[1, 2].set_title('Wavelet Vertical Detail')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'rf_classifier': self.rf_classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_classifier = model_data['rf_classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.class_names = model_data['class_names']
        self.feature_names = model_data['feature_names']
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def create_sample_textures():
    """Create sample texture images for demonstration"""
    print("Creating sample texture dataset...")
    
    # Create directories
    os.makedirs('sample_textures/brick', exist_ok=True)
    os.makedirs('sample_textures/wood', exist_ok=True)
    os.makedirs('sample_textures/fabric', exist_ok=True)
    os.makedirs('sample_textures/metal', exist_ok=True)
    
    # Generate synthetic textures
    size = (128, 128)
    
    for i in range(20):
        # Brick texture (rectangular patterns)
        brick = np.random.randint(100, 150, size, dtype=np.uint8)
        for y in range(0, size[0], 20):
            for x in range(0, size[1], 40):
                cv2.rectangle(brick, (x, y), (x+35, y+15), 80, 2)
        cv2.imwrite(f'sample_textures/brick/brick_{i}.jpg', brick)
        
        # Wood texture (vertical lines with noise)
        wood = np.random.randint(80, 120, size, dtype=np.uint8)
        for x in range(0, size[1], 5):
            wood[:, x:x+2] = np.random.randint(60, 90)
        cv2.imwrite(f'sample_textures/wood/wood_{i}.jpg', wood)
        
        # Fabric texture (crosshatch pattern)
        fabric = np.random.randint(120, 180, size, dtype=np.uint8)
        for i_line in range(0, size[0], 10):
            fabric[i_line:i_line+2, :] = np.random.randint(90, 110)
        for j_line in range(0, size[1], 10):
            fabric[:, j_line:j_line+2] = np.random.randint(90, 110)
        cv2.imwrite(f'sample_textures/fabric/fabric_{i}.jpg', fabric)
        
        # Metal texture (smooth with some scratches)
        metal = np.random.randint(150, 200, size, dtype=np.uint8)
        for _ in range(5):
            y = np.random.randint(0, size[0])
            cv2.line(metal, (0, y), (size[1], y), 100, 1)
        cv2.imwrite(f'sample_textures/metal/metal_{i}.jpg', metal)
    
    print("Sample texture dataset created in 'sample_textures' directory")

def main():
    """Main function for texture classification demo"""
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists('sample_textures'):
        create_sample_textures()
    
    # Initialize classifier
    classifier = TextureClassifier(n_estimators=100)
    
    try:
        # Load dataset
        X, y = classifier.load_dataset('sample_textures')
        
        if len(X) < 10:
            print("Dataset too small. Please add more texture images.")
            return
        
        # Visualize texture features for first sample
        sample_image = cv2.imread('sample_textures/brick/brick_0.jpg')
        if sample_image is not None:
            print("\nVisualizing texture features for sample image...")
            classifier.visualize_texture_features(sample_image)
        
        # Train the classifier
        accuracy = classifier.train(X, y)
        
        # Save the model
        classifier.save_model('texture_classifier.pkl')
        
        # Test prediction on a sample image
        test_image = cv2.imread('sample_textures/wood/wood_0.jpg')
        if test_image is not None:
            predicted_class, confidence = classifier.predict(test_image)
            print(f"\nPrediction for test image: {predicted_class} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your dataset follows the required directory structure")

if __name__ == "__main__":
    main()
