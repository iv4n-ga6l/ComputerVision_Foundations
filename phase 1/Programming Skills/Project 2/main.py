"""
Implement a basic image filter to blur, sharpen, and detect edges.
"""

import cv2
import numpy as np

def apply_blur(image, kernel_size=(5,5)):
    """
    Apply Gaussian blur to an image
    Parameters:
        image: Input image
        kernel_size: Tuple of (height, width) for the Gaussian kernel
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_sharpen(image):
    """
    Sharpen an image using an unsharp masking technique
    Parameters:
        image: Input image
    Returns:
        Sharpened image
    """
    # Create the sharpening kernel
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    
    # Apply the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def detect_edges(image, threshold1=100, threshold2=200):
    """
    Detect edges in an image using Canny edge detection
    Parameters:
        image: Input image
        threshold1: Lower threshold for edge detection
        threshold2: Upper threshold for edge detection
    Returns:
        Image with detected edges
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

def main():
    # Read an image
    image = cv2.imread('horse.jpg')
    if image is None:
        print("Error: Could not read the image")
        return
    
    # Apply filters
    blurred = apply_blur(image)
    sharpened = apply_sharpen(image)
    edges = detect_edges(image)
    
    # Save results
    cv2.imwrite('blurred.jpg', blurred)
    cv2.imwrite('sharpened.jpg', sharpened)
    cv2.imwrite('edges.jpg', edges)
    
    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Sharpened', sharpened)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()