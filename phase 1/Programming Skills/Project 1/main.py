"""
Create a Python program that reads an image, converts it to grayscale, and saves it using OpenCV.
"""

import cv2
import sys

def convert_to_grayscale(input_path, output_path):
    """
    Convert an image to grayscale using OpenCV.
    
    Parameters:
    input_path (str): Path to the input image
    output_path (str): Path where the grayscale image will be saved
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(input_path)
        
        # Check if image was successfully loaded
        if image is None:
            print(f"Error: Could not load image at {input_path}")
            return False
        
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Save the grayscale image
        cv2.imwrite(output_path, grayscale)
        print(f"Successfully saved grayscale image to {output_path}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image_path> <output_image_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_to_grayscale(input_path, output_path)

if __name__ == "__main__":
    main()