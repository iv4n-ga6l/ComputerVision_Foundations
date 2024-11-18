"""
Create a histogram equalization script to improve the contrast of grayscale images.
"""

import cv2

def histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image to enhance contrast.
    
    Args:
        image (numpy.ndarray): Input grayscale image
    
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)
    
    return equalized_image

def process_image(input_path, output_path):
    """
    Read an image, apply histogram equalization, and save the result.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the equalized image
    """
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded successfully
    if image is None:
        raise ValueError(f"Unable to read image from {input_path}")
    
    # Apply histogram equalization
    equalized_image = histogram_equalization(image)
    
    # Save the equalized image
    cv2.imwrite(output_path, equalized_image)
    
    # Display original and equalized images
    cv2.imshow('Original Image', image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_image_path = 'horse.jpg'
    output_image_path = 'equalized_image.jpg'
    process_image(input_image_path, output_image_path)