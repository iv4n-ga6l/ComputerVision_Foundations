"""
Develop a tool to extract and draw contours on objects in an image using thresholding techniques.
"""

import cv2

def extract_contours(image_path, threshold_method='binary', block_size=11, c_value=2):
    """
    Extract and draw contours on an image using various thresholding techniques.
    
    Parameters:
    - image_path: Path to the input image
    - threshold_method: Thresholding technique ('binary', 'adaptive', 'otsu')
    - block_size: Block size for adaptive thresholding (must be odd)
    - c_value: Constant subtracted from mean for adaptive thresholding
    
    Returns:
    - contour_image: Image with contours drawn
    - contours: List of detected contours
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply appropriate thresholding
    if threshold_method == 'binary':
        # Simple binary thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    elif threshold_method == 'otsu':
        # Otsu's thresholding method
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif threshold_method == 'adaptive':
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            c_value
        )
    
    else:
        raise ValueError("Invalid threshold method. Choose 'binary', 'adaptive', or 'otsu'.")
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create a copy of the original image to draw contours
    contour_image = image.copy()
    
    # Draw contours on the image
    cv2.drawContours(
        contour_image, 
        contours, 
        -1,  # draw all contours
        (0, 255, 0),  # contour color (green)
        2  # contour thickness
    )
    
    return contour_image, contours

def main():
    try:
        image_path = 'horse.jpg'
        
        # Extract contours using different methods
        methods = ['binary', 'adaptive', 'otsu']
        
        for method in methods:
            # Extract contours
            result_image, contours = extract_contours(
                image_path, 
                threshold_method=method
            )
            
            # Save the result
            output_path = f'contours_{method}.jpg'
            cv2.imwrite(output_path, result_image)
            
            # Print number of contours found
            print(f"Method {method}: Found {len(contours)} contours")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()