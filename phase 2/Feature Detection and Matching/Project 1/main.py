"""
Implement corner detection using the Harris corner detector and visualize the results.
"""

import cv2
import matplotlib.pyplot as plt

def harris_corner_detection(image_path, k=0.04, threshold=0.01):
    """
    Detect corners in an image using the Harris corner detector.
    
    Parameters:
    - image_path: Path to the input image
    - k: Harris detector free parameter (typically 0.04-0.06)
    - threshold: Corner response threshold (0-1)
    
    Returns:
    - Original image with detected corners marked
    """
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute derivatives
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Compute components of the Harris matrix
    ixx = dx**2
    iyy = dy**2
    ixy = dx*dy
    
    # Compute cornerness response
    k_param = 0.04
    det = (ixx * iyy) - (ixy**2)
    trace = ixx + iyy
    r = det - k_param * (trace**2)
    
    # Normalize the response
    r_norm = cv2.normalize(r, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply non-maximum suppression and thresholding
    corner_img = img.copy()
    corner_coordinates = []
    
    for y in range(r.shape[0]):
        for x in range(r.shape[1]):
            # Check if pixel is a local maximum and above threshold
            if r_norm[y, x] > threshold:
                is_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if (dy != 0 or dx != 0):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < r.shape[0] and 0 <= nx < r.shape[1] and 
                                r_norm[y, x] < r_norm[ny, nx]):
                                is_max = False
                                break
                    if not is_max:
                        break
                
                # If local maximum, mark as corner
                if is_max:
                    cv2.circle(corner_img, (x, y), 5, (0, 255, 0), 2)
                    corner_coordinates.append((x, y))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Corner Response')
    plt.imshow(r_norm, cmap='jet')
    plt.colorbar()
    
    plt.subplot(133)
    plt.title('Detected Corners')
    plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corner_coordinates

corners = harris_corner_detection('horse.jpg')
