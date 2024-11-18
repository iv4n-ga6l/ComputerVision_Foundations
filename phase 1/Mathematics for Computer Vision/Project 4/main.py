"""
Calculate and visualize histograms of image pixel intensity using Python.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def calculate_pixel_intensity_histogram(image_path):
    """
    Calculate and visualize pixel intensity histogram for a given image.
    
    Parameters:
    image_path (str): Path to the input image file
    """
    # Read the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    # Calculate histogram
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Subplot 2: Pixel Intensity Histogram
    plt.subplot(1, 2, 2)
    plt.bar(bins[:-1], hist, width=1, color='black', alpha=0.7)
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_pixel_intensity_histogram('horse.jpg')