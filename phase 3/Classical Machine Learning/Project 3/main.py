"""
PCA Face Recognition using Eigenfaces
=====================================

This project implements the classic Eigenfaces method for face recognition using:
- Principal Component Analysis (PCA) for dimensionality reduction
- k-Nearest Neighbors (k-NN) for classification
- Haar cascade for face detection
- Real-time recognition capabilities

Author: Computer Vision Foundations
"""

import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

class EigenfaceRecognizer:
    def __init__(self, n_components=50, face_size=(100, 100)):
        """
        Initialize the Eigenface recognizer
        
        Args:
            n_components (int): Number of principal components to keep
            face_size (tuple): Target size for face images (width, height)
        """
        self.n_components = n_components
        self.face_size = face_size
        self.pca = PCA(n_components=n_components)
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.mean_face = None
        self.eigenfaces = None
        self.face_labels = []
        self.label_names = []
        self.is_trained = False
    
    def detect_face(self, image):
        """
        Detect and extract face from image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray or None: Detected face image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, self.face_size)
            return face_resized
        
        return None
    
    def load_dataset(self, dataset_path):
        """
        Load face dataset from directory structure
        
        Args:
            dataset_path (str): Path to dataset directory
            
        Returns:
            tuple: Face images array and corresponding labels
        """
        faces = []
        labels = []
        self.label_names = []
        
        dataset_path = Path(dataset_path)
        
        print("Loading face dataset...")
        label_id = 0
        
        for person_dir in dataset_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                self.label_names.append(person_name)
                print(f"Loading faces for: {person_name}")
                
                for image_file in person_dir.glob('*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image = cv2.imread(str(image_file))
                        
                        if image is not None:
                            face = self.detect_face(image)
                            if face is not None:
                                faces.append(face.flatten())
                                labels.append(label_id)
                
                label_id += 1
        
        print(f"Loaded {len(faces)} face images from {len(self.label_names)} people")
        return np.array(faces), np.array(labels)
    
    def train(self, X, y, test_size=0.2):
        """
        Train the eigenface recognizer
        
        Args:
            X (np.ndarray): Face images (flattened)
            y (np.ndarray): Labels
            test_size (float): Proportion of test data
            
        Returns:
            float: Test accuracy
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("Computing PCA and eigenfaces...")
        
        # Fit PCA on training data
        X_train_pca = self.pca.fit_transform(X_train)
        
        # Store mean face and eigenfaces
        self.mean_face = self.pca.mean_.reshape(self.face_size)
        self.eigenfaces = self.pca.components_.reshape(
            self.n_components, self.face_size[1], self.face_size[0]
        )
        
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Train classifier in PCA space
        self.classifier.fit(X_train_pca, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        X_test_pca = self.pca.transform(X_test)
        y_pred = self.classifier.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_names))
        
        return accuracy
    
    def visualize_eigenfaces(self, n_faces=10):
        """
        Visualize the top eigenfaces
        
        Args:
            n_faces (int): Number of eigenfaces to display
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before visualization")
        
        n_faces = min(n_faces, self.n_components)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Show mean face
        axes[0].imshow(self.mean_face, cmap='gray')
        axes[0].set_title('Mean Face')
        axes[0].axis('off')
        
        # Show eigenfaces
        for i in range(1, n_faces):
            axes[i].imshow(self.eigenfaces[i-1], cmap='gray')
            axes[i].set_title(f'Eigenface {i}')
            axes[i].axis('off')
        
        plt.suptitle('Mean Face and Top Eigenfaces')
        plt.tight_layout()
        plt.show()
    
    def analyze_components(self, max_components=100):
        """
        Analyze the effect of different numbers of components on accuracy
        
        Args:
            max_components (int): Maximum number of components to test
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            print("Please run train() method first")
            return
        
        components_range = range(5, min(max_components, self.X_train.shape[1]), 5)
        accuracies = []
        
        print("Analyzing optimal number of components...")
        
        for n_comp in components_range:
            pca_temp = PCA(n_components=n_comp)
            X_train_pca = pca_temp.fit_transform(self.X_train)
            X_test_pca = pca_temp.transform(self.X_test)
            
            clf_temp = KNeighborsClassifier(n_neighbors=3)
            clf_temp.fit(X_train_pca, self.y_train)
            y_pred = clf_temp.predict(X_test_pca)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            accuracies.append(accuracy)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(components_range, accuracies, 'b-o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Accuracy')
        plt.title('Recognition Accuracy vs Number of Components')
        plt.grid(True)
        plt.show()
        
        # Find optimal number
        optimal_idx = np.argmax(accuracies)
        optimal_components = list(components_range)[optimal_idx]
        optimal_accuracy = accuracies[optimal_idx]
        
        print(f"Optimal number of components: {optimal_components}")
        print(f"Best accuracy: {optimal_accuracy:.4f}")
    
    def predict(self, image):
        """
        Predict the identity of a face in an image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: Predicted person name and confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        face = self.detect_face(image)
        if face is None:
            return None, 0.0
        
        # Project face to eigenspace
        face_vector = face.flatten().reshape(1, -1)
        face_pca = self.pca.transform(face_vector)
        
        # Predict using k-NN
        prediction = self.classifier.predict(face_pca)[0]
        probabilities = self.classifier.predict_proba(face_pca)[0]
        confidence = np.max(probabilities)
        
        person_name = self.label_names[prediction]
        
        return person_name, confidence
    
    def reconstruct_face(self, face_image, n_components=None):
        """
        Reconstruct a face using eigenfaces
        
        Args:
            face_image (np.ndarray): Input face image
            n_components (int): Number of components to use for reconstruction
            
        Returns:
            np.ndarray: Reconstructed face image
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before reconstruction")
        
        if n_components is None:
            n_components = self.n_components
        
        # Project to eigenspace and back
        face_vector = face_image.flatten().reshape(1, -1)
        face_pca = self.pca.transform(face_vector)
        
        # Use only specified number of components
        face_pca_truncated = face_pca.copy()
        face_pca_truncated[:, n_components:] = 0
        
        # Reconstruct
        reconstructed = self.pca.inverse_transform(face_pca_truncated)
        reconstructed_image = reconstructed.reshape(self.face_size)
        
        return reconstructed_image
    
    def real_time_recognition(self):
        """
        Perform real-time face recognition using webcam
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before real-time recognition")
        
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time face recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Recognize face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, self.face_size)
                
                person_name, confidence = self.predict(frame)
                
                if person_name and confidence > 0.6:
                    label = f"{person_name}: {confidence:.2f}"
                    color = (0, 255, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                
                # Put label
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, color, 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'pca': self.pca,
            'classifier': self.classifier,
            'label_names': self.label_names,
            'n_components': self.n_components,
            'face_size': self.face_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.classifier = model_data['classifier']
        self.label_names = model_data['label_names']
        self.n_components = model_data['n_components']
        self.face_size = model_data['face_size']
        
        # Reconstruct eigenfaces
        self.mean_face = self.pca.mean_.reshape(self.face_size)
        self.eigenfaces = self.pca.components_.reshape(
            self.n_components, self.face_size[1], self.face_size[0]
        )
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def create_sample_dataset():
    """Create a sample dataset using webcam"""
    print("Creating sample face dataset...")
    print("This will capture faces from your webcam.")
    print("Press 's' to save a face, 'n' for next person, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    person_id = 0
    photo_count = 0
    max_photos_per_person = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
        
        cv2.putText(frame, f"Person {person_id+1}, Photo {photo_count+1}/{max_photos_per_person}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save, 'n' for next person, 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Capture Faces', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(faces) > 0:
            # Save face
            person_dir = f'sample_faces/person_{person_id+1}'
            os.makedirs(person_dir, exist_ok=True)
            
            (x, y, w, h) = faces[0]  # Take first face
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            cv2.imwrite(f'{person_dir}/face_{photo_count+1}.jpg', face_resized)
            print(f"Saved face {photo_count+1} for person {person_id+1}")
            
            photo_count += 1
            
            if photo_count >= max_photos_per_person:
                print(f"Completed capturing for person {person_id+1}")
                person_id += 1
                photo_count = 0
        
        elif key == ord('n'):
            # Next person
            if photo_count > 0:
                person_id += 1
                photo_count = 0
                print(f"Moving to person {person_id+1}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset creation completed. {person_id+1} people captured.")

def main():
    """Main function to demonstrate eigenface recognition"""
    
    # Initialize recognizer
    recognizer = EigenfaceRecognizer(n_components=50)
    
    # Check if dataset exists, if not create sample
    if not os.path.exists('sample_faces'):
        print("No dataset found. Creating sample dataset...")
        create_sample_dataset()
    
    try:
        # Load dataset
        X, y = recognizer.load_dataset('sample_faces')
        
        if len(X) < 10:
            print("Dataset too small. Please add more face images.")
            return
        
        # Train the recognizer
        accuracy = recognizer.train(X, y)
        
        # Visualize eigenfaces
        recognizer.visualize_eigenfaces()
        
        # Save model
        recognizer.save_model('eigenface_model.pkl')
        
        # Start real-time recognition
        print("\nStarting real-time face recognition...")
        recognizer.real_time_recognition()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have face images in the correct directory structure")

if __name__ == "__main__":
    main()
