"""
Build an application that detects and highlights the largest contour in a series of images or video frames.
"""

import cv2
import numpy as np

def preprocess_image(image, blur_kernel=(5,5), threshold_method=cv2.THRESH_BINARY):
    """
    Preprocess image for contour detection
    
    Args:
        image (numpy.ndarray): Input image
        blur_kernel (tuple): Gaussian blur kernel size
        threshold_method (int): Thresholding method
    
    Returns:
        numpy.ndarray: Preprocessed binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        threshold_method, 
        11, 2
    )
    
    return binary

def find_largest_contour(binary_image):
    """
    Find the largest contour in a binary image
    
    Args:
        binary_image (numpy.ndarray): Binary input image
    
    Returns:
        numpy.ndarray: Largest contour
    """
    # Find contours
    contours, _ = cv2.findContours(
        binary_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Return largest contour by area
    return max(contours, key=cv2.contourArea) if contours else None

def highlight_largest_contour(original_image, contour):
    """
    Highlight the largest contour on the original image
    
    Args:
        original_image (numpy.ndarray): Original color image
        contour (numpy.ndarray): Largest contour
    
    Returns:
        numpy.ndarray: Image with highlighted contour
    """
    # Create a copy of the image
    result = original_image.copy()
    
    # Draw the contour
    cv2.drawContours(
        result, 
        [contour], 
        -1,  # Draw all contours
        (0, 255, 0),  # Green color
        3  # Thickness
    )
    
    # Optional: Fill the contour with semi-transparent color
    mask = np.zeros(original_image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    color_mask = np.zeros_like(original_image)
    color_mask[mask == 255] = (0, 255, 0)  # Green fill
    
    # Blend the color mask
    result = cv2.addWeighted(result, 0.7, color_mask, 0.3, 0)
    
    return result

def process_image(image_path, output_path=None):
    """
    Process an image to detect and highlight the largest contour
    
    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save output image
    
    Returns:
        numpy.ndarray: Processed image with largest contour highlighted
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Preprocess
    binary = preprocess_image(image)
    
    # Find largest contour
    largest_contour = find_largest_contour(binary)
    
    if largest_contour is None:
        print(f"No contours found in {image_path}")
        return image
    
    # Highlight contour
    result = highlight_largest_contour(image, largest_contour)
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def process_video(input_video_path, output_video_path):
    """
    Process a video to detect and highlight largest contour in each frame
    
    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to save output video
    """
    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Preprocess
        binary = preprocess_image(frame)
        
        # Find largest contour
        largest_contour = find_largest_contour(binary)
        
        if largest_contour is not None:
            # Highlight contour
            frame = highlight_largest_contour(frame, largest_contour)
        
        # Write frame
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()


if __name__ == "__main__":
    # Process single image
    process_image('horse.jpg', 'output_image.jpg')
    
    # Process video
    process_video('video.mp4', 'output_video.mp4')