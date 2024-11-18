"""
Implement a small project that sorts a list of images based on dominant colors.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import shutil

class ImageColorSorter:
    def __init__(self, input_folder: str, output_folder: str):
        """
        Initialize the image sorter with input and output folder paths.
        
        Args:
            input_folder (str): Path to folder containing input images
            output_folder (str): Path to folder where sorted images will be saved
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Supported image extensions
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
    def get_dominant_color(self, image_path: str) -> Tuple[float, float, float]:
        """
        Extract the dominant color from an image using k-means clustering.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: HSV values of the dominant color
        """
        # Read image and convert to RGB
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32
        pixels = np.float32(pixels)
        
        # Define criteria for k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        k = 5  # Number of clusters
        
        # Perform k-means clustering
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique_labels[np.argmax(counts)]
        
        # Get the color of the dominant cluster
        dominant_color = centers[dominant_cluster]
        
        # Convert RGB to HSV for better color sorting
        dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
        
        return tuple(dominant_color_hsv)
    
    def sort_images(self):
        """
        Sort all images in the input folder based on their dominant colors.
        Creates a new folder structure organized by hue ranges.
        """
        # Get all image files
        image_files = [
            f for f in self.input_folder.iterdir()
            if f.suffix.lower() in self.image_extensions
        ]
        
        # Process each image
        image_colors = []
        for image_file in image_files:
            try:
                dominant_color = self.get_dominant_color(str(image_file))
                image_colors.append((image_file, dominant_color))
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # Sort images by hue
        image_colors.sort(key=lambda x: x[1][0])
        
        # Create color range folders and copy images
        for idx, (image_file, color) in enumerate(image_colors, 1):
            # Create a folder name based on hue range
            hue = color[0]
            hue_range = int(hue / 30) * 30  # Group by 30-degree hue intervals
            folder_name = f"hue_{hue_range:03d}-{hue_range+30:03d}"
            
            # Create folder
            color_folder = self.output_folder / folder_name
            color_folder.mkdir(exist_ok=True)
            
            # Copy image to new location with index prefix
            new_name = f"{idx:03d}_{image_file.name}"
            shutil.copy2(image_file, color_folder / new_name)
            
            print(f"Processed {image_file.name} -> {folder_name}")

def main():
    input_folder = "input_images"
    output_folder = "sorted_images"
    
    sorter = ImageColorSorter(input_folder, output_folder)
    sorter.sort_images()

if __name__ == "__main__":
    main()