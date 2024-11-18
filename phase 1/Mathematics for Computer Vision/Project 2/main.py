"""
Implement a visualization of 2D geometric transformations (scaling, rotation, and translation) on images.
"""

import cv2
import numpy as np

def load_and_prepare_image(image_path, size=(400, 400)):
    """Load and resize image to a standard size."""
    img = cv2.imread(image_path)
    if img is None:
        # Create a sample image if no image is provided
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (250, 250), (0, 255, 0), -1)
        cv2.circle(img, (200, 200), 50, (255, 0, 0), 3)
    return cv2.resize(img, size)

def scale_image(img, scale_x, scale_y):
    """Scale the image by given factors."""
    height, width = img.shape[:2]
    scaled_width = int(width * scale_x)
    scaled_height = int(height * scale_y)
    
    # Create transformation matrix
    scaling_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    
    # Apply affine transformation
    scaled_img = cv2.warpAffine(img, scaling_matrix, (scaled_width, scaled_height))
    return scaled_img

def rotate_image(img, angle_degrees, scale=1.0):
    """Rotate the image by given angle in degrees."""
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    
    # Apply affine transformation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def translate_image(img, tx, ty):
    """Translate the image by tx, ty pixels."""
    height, width = img.shape[:2]
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    # Apply affine transformation
    translated_img = cv2.warpAffine(img, translation_matrix, (width, height))
    return translated_img

def visualize_transformations(image_path=None):
    """Demonstrate various geometric transformations on an image."""
    # Load or create image
    original_img = load_and_prepare_image(image_path)
    
    # Create windows for different transformations
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Scaled', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Rotated', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Translated', cv2.WINDOW_NORMAL)
    
    # Initialize transformation parameters
    scale_x, scale_y = 1.0, 1.0
    rotation_angle = 0
    tx, ty = 0, 0
    
    while True:
        # Apply transformations
        scaled_img = scale_image(original_img, scale_x, scale_y)
        rotated_img = rotate_image(original_img, rotation_angle)
        translated_img = translate_image(original_img, tx, ty)
        
        # Display results
        cv2.imshow('Original', original_img)
        cv2.imshow('Scaled', scaled_img)
        cv2.imshow('Rotated', rotated_img)
        cv2.imshow('Translated', translated_img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Scaling controls
        if key == ord('q'):  # Increase X scale
            scale_x += 0.1
        elif key == ord('a'):  # Decrease X scale
            scale_x = max(0.1, scale_x - 0.1)
        elif key == ord('w'):  # Increase Y scale
            scale_y += 0.1
        elif key == ord('s'):  # Decrease Y scale
            scale_y = max(0.1, scale_y - 0.1)
        
        # Rotation controls
        elif key == ord('e'):  # Rotate clockwise
            rotation_angle -= 5
        elif key == ord('r'):  # Rotate counter-clockwise
            rotation_angle += 5
        
        # Translation controls
        elif key == ord('d'):  # Translate right
            tx += 10
        elif key == ord('f'):  # Translate left
            tx -= 10
        elif key == ord('g'):  # Translate down
            ty += 10
        elif key == ord('h'):  # Translate up
            ty -= 10
        
        # Reset controls
        elif key == ord('z'):  # Reset all transformations
            scale_x, scale_y = 1.0, 1.0
            rotation_angle = 0
            tx, ty = 0, 0
        
        # Exit
        elif key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_transformations()