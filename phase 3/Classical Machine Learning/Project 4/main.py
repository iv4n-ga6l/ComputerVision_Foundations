"""
K-Means Image Segmentation
==========================

This project implements image segmentation using K-Means clustering with:
- Multiple color space support (RGB, HSV, LAB)
- Spatial-aware clustering
- Optimal k selection using elbow method
- Interactive segmentation interface


"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from pathlib import Path

class KMeansSegmentation:
    def __init__(self, color_space='RGB'):
        """
        Initialize K-Means segmentation
        
        Args:
            color_space (str): Color space to use ('RGB', 'HSV', 'LAB')
        """
        self.color_space = color_space
        self.original_image = None
        self.processed_image = None
        self.segmented_image = None
        self.labels = None
        self.centers = None
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert color space
        if self.color_space == 'HSV':
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'LAB':
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        else:  # RGB
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
    
    def find_optimal_k(self, max_k=10, include_spatial=False):
        """
        Find optimal number of clusters using elbow method
        
        Args:
            max_k (int): Maximum number of clusters to test
            include_spatial (bool): Include spatial coordinates as features
            
        Returns:
            int: Optimal number of clusters
        """
        if self.processed_image is None:
            raise ValueError("Please load an image first")
        
        # Prepare data
        data = self._prepare_data(include_spatial)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        print("Finding optimal k...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow plot
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal k: {optimal_k}")
        
        return optimal_k
    
    def _prepare_data(self, include_spatial=False):
        """
        Prepare data for clustering
        
        Args:
            include_spatial (bool): Include spatial coordinates
            
        Returns:
            np.ndarray: Prepared data matrix
        """
        h, w, c = self.processed_image.shape
        
        # Reshape image to pixel array
        data = self.processed_image.reshape((-1, c))
        
        if include_spatial:
            # Add spatial coordinates
            coords = np.array([[i, j] for i in range(h) for j in range(w)])
            # Normalize spatial coordinates
            coords = coords / np.array([h, w])
            data = np.hstack([data, coords])
        
        return data.astype(np.float32)
    
    def segment(self, k, include_spatial=False, random_state=42):
        """
        Perform K-Means segmentation
        
        Args:
            k (int): Number of clusters
            include_spatial (bool): Include spatial information
            random_state (int): Random seed
            
        Returns:
            np.ndarray: Segmented image
        """
        if self.processed_image is None:
            raise ValueError("Please load an image first")
        
        # Prepare data
        data = self._prepare_data(include_spatial)
        
        # Perform K-Means clustering
        print(f"Performing K-Means clustering with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.labels = kmeans.fit_predict(data)
        self.centers = kmeans.cluster_centers_
        
        # Create segmented image using cluster centers
        h, w, c = self.processed_image.shape
        
        if include_spatial:
            # Use only color information from centers for reconstruction
            color_centers = self.centers[:, :c]
        else:
            color_centers = self.centers
        
        segmented_data = color_centers[self.labels]
        self.segmented_image = segmented_data.reshape((h, w, c)).astype(np.uint8)
        
        return self.segmented_image
    
    def create_mask(self, cluster_id):
        """
        Create binary mask for specific cluster
        
        Args:
            cluster_id (int): Cluster ID to create mask for
            
        Returns:
            np.ndarray: Binary mask
        """
        if self.labels is None:
            raise ValueError("Please perform segmentation first")
        
        h, w = self.processed_image.shape[:2]
        mask = (self.labels == cluster_id).reshape((h, w)).astype(np.uint8) * 255
        
        return mask
    
    def visualize_clusters(self):
        """Visualize individual clusters"""
        if self.labels is None:
            raise ValueError("Please perform segmentation first")
        
        h, w = self.processed_image.shape[:2]
        n_clusters = len(np.unique(self.labels))
        
        # Create subplot grid
        cols = min(4, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_clusters):
            row, col = i // cols, i % cols
            
            # Create mask for this cluster
            mask = self.create_mask(i)
            
            # Apply mask to original image
            masked_image = self.original_image.copy()
            masked_image[mask == 0] = 0
            
            axes[row, col].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'Cluster {i}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_clusters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_color_spaces(self, k=4):
        """Compare segmentation across different color spaces"""
        if self.original_image is None:
            raise ValueError("Please load an image first")
        
        color_spaces = ['RGB', 'HSV', 'LAB']
        results = {}
        
        original_color_space = self.color_space
        
        for cs in color_spaces:
            print(f"Segmenting in {cs} color space...")
            self.color_space = cs
            
            # Convert to appropriate color space
            if cs == 'HSV':
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            elif cs == 'LAB':
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
            else:  # RGB
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Segment
            segmented = self.segment(k)
            
            # Convert back to RGB for display
            if cs == 'HSV':
                results[cs] = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
            elif cs == 'LAB':
                results[cs] = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
            else:
                results[cs] = segmented
        
        # Restore original color space
        self.color_space = original_color_space
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        for i, (cs, result) in enumerate(results.items()):
            row, col = (i + 1) // 2, (i + 1) % 2
            axes[row, col].imshow(result)
            axes[row, col].set_title(f'K-Means ({cs})')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return results

def create_sample_image():
    """Create a sample image for demonstration"""
    # Create a simple synthetic image with distinct regions
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Sky (blue)
    img[:60, :] = [135, 206, 235]
    
    # Grass (green)
    img[140:, :] = [34, 139, 34]
    
    # Tree (brown trunk, green leaves)
    img[60:140, 140:160] = [139, 69, 19]  # trunk
    img[40:100, 120:180] = [0, 128, 0]   # leaves
    
    # House (red)
    img[100:140, 50:120] = [220, 20, 60]
    
    # Sun (yellow)
    cv2.circle(img, (250, 30), 20, (255, 255, 0), -1)
    
    cv2.imwrite('sample_image.jpg', img)
    print("Sample image created: sample_image.jpg")
    
    return img

def main():
    """Main function for K-Means segmentation demo"""
    parser = argparse.ArgumentParser(description='K-Means Image Segmentation')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters')
    parser.add_argument('--color_space', type=str, default='RGB', 
                       choices=['RGB', 'HSV', 'LAB'], help='Color space')
    parser.add_argument('--spatial', action='store_true', 
                       help='Include spatial information')
    parser.add_argument('--find_k', action='store_true', 
                       help='Find optimal k using elbow method')
    
    args = parser.parse_args()
    
    # Create sample image if none provided
    if not args.image:
        if not Path('sample_image.jpg').exists():
            create_sample_image()
        args.image = 'sample_image.jpg'
    
    # Initialize segmentation
    segmenter = KMeansSegmentation(color_space=args.color_space)
    
    try:
        # Load image
        segmenter.load_image(args.image)
        print(f"Loaded image: {args.image}")
        
        # Find optimal k if requested
        if args.find_k:
            optimal_k = segmenter.find_optimal_k(include_spatial=args.spatial)
            args.k = optimal_k
        
        # Perform segmentation
        segmented = segmenter.segment(args.k, include_spatial=args.spatial)
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(segmenter.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmented image
        if args.color_space == 'RGB':
            display_segmented = segmented
        elif args.color_space == 'HSV':
            display_segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
        else:  # LAB
            display_segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
        
        axes[1].imshow(display_segmented)
        axes[1].set_title(f'K-Means (k={args.k}, {args.color_space})')
        axes[1].axis('off')
        
        # Cluster labels visualization
        h, w = segmenter.original_image.shape[:2]
        labels_img = segmenter.labels.reshape((h, w))
        axes[2].imshow(labels_img, cmap='viridis')
        axes[2].set_title('Cluster Labels')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Visualize individual clusters
        segmenter.visualize_clusters()
        
        # Compare color spaces
        print("\nComparing different color spaces...")
        segmenter.compare_color_spaces(k=args.k)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
