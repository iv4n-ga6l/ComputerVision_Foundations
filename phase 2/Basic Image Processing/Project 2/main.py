"""
Create a Python program to convert images between RGB, HSV, and grayscale color spaces.
"""

import cv2

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        numpy.ndarray: Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    return image

def convert_to_rgb(image):
    """
    Convert image to RGB color space.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Image in RGB color space
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_hsv(image):
    """
    Convert image to HSV color space.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Image in HSV color space
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def save_image(image, output_path):
    """
    Save an image to the specified path.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the image
    """
    cv2.imwrite(output_path, image)

def main():
    # Load an image
    input_image = load_image('horse.jpg')
    
    # Convert to different color spaces
    rgb_image = convert_to_rgb(input_image)
    hsv_image = convert_to_hsv(input_image)
    grayscale_image = convert_to_grayscale(input_image)
    
    # Save converted images
    save_image(rgb_image, 'output_rgb.jpg')
    save_image(hsv_image, 'output_hsv.jpg')
    save_image(grayscale_image, 'output_grayscale.jpg')

if __name__ == '__main__':
    main()