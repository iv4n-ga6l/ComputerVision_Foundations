"""
Implement an edge detection algorithm using the Sobel operator and compare it with the Canny edge detection method.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge_detection(image):
    """
    Perform edge detection using Sobel operator
    
    Args:
    image (numpy.ndarray): Input grayscale image
    
    Returns:
    numpy.ndarray: Edge-detected image
    """
    # Compute Sobel derivatives in x and y directions
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude of gradients
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255 range
    sobel_edges = (sobel_edges / sobel_edges.max() * 255).astype(np.uint8)
    
    return sobel_edges

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Perform edge detection using Canny algorithm
    
    Args:
    image (numpy.ndarray): Input grayscale image
    low_threshold (int): Lower threshold for edge detection
    high_threshold (int): Higher threshold for edge detection
    
    Returns:
    numpy.ndarray: Edge-detected image
    """
    # Apply Canny edge detection
    canny_edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return canny_edges

def compare_edge_detection(image_path):
    """
    Compare Sobel and Canny edge detection methods
    
    Args:
    image_path (str): Path to input image
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Sobel edge detection
    sobel_edges = sobel_edge_detection(image)
    
    # Apply Canny edge detection
    canny_edges = canny_edge_detection(image)
    
    # Visualize results
    plt.figure(figsize=(15,5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Sobel Edge Detection')
    plt.imshow(sobel_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Canny Edge Detection')
    plt.imshow(canny_edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


compare_edge_detection('horse.jpg')