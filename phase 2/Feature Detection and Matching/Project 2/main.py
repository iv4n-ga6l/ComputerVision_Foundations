"""
Use SIFT or ORB to detect and match features between two similar images.
"""

import cv2

def detect_and_match_features(image1_path, image2_path):
    """
    Detect and match features between two images using ORB
    
    Args:
    image1_path (str): Path to the first image
    image2_path (str): Path to the second image
    
    Returns:
    numpy.ndarray: Image with matched features
    """
    # Read images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw top 10 matches
    result = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches[:10], 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return result

result = detect_and_match_features('image1.jpg', 'image2.jpg')
cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()