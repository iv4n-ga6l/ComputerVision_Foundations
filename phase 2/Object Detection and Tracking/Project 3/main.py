"""
Build a basic object detection system using template matching to find logos in a series of images.
"""

import cv2
import numpy as np
from pathlib import Path

class LogoDetector:
    def __init__(self, template_path, threshold=0.8):
        """
        Initialize the logo detector
        
        Args:
            template_path (str): Path to the template logo image
            threshold (float): Matching threshold (0-1), higher means more strict matching
        """
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not load template image from {template_path}")
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.threshold = threshold
        
    def find_logo(self, image_path):
        """
        Find the logo in a single image
        
        Args:
            image_path (str): Path to the image to search in
            
        Returns:
            list: List of tuples containing (x, y, width, height) for each match
        """
        # Read and convert image to grayscale
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get template dimensions
        w, h = self.template_gray.shape[::-1]
        
        # Perform template matching
        result = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where matching exceeds threshold
        locations = np.where(result >= self.threshold)
        matches = []
        
        # Convert locations to rectangles
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], w, h))
            
        # Apply non-maximum suppression to remove overlapping boxes
        if matches:
            matches = self._non_max_suppression(matches)
            
        return matches
    
    def process_directory(self, directory_path):
        """
        Process all images in a directory
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            dict: Dictionary with image paths as keys and lists of matches as values
        """
        results = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for file_path in Path(directory_path).iterdir():
            if file_path.suffix.lower() in image_extensions:
                try:
                    matches = self.find_logo(str(file_path))
                    results[str(file_path)] = matches
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        return results
    
    def visualize_results(self, image_path, matches):
        """
        Draw boxes around detected logos
        
        Args:
            image_path (str): Path to the original image
            matches (list): List of match coordinates
            
        Returns:
            numpy.ndarray: Image with drawn rectangles
        """
        image = cv2.imread(image_path)
        for (x, y, w, h) in matches:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image
    
    def _non_max_suppression(self, boxes, overlap_thresh=0.3):
        """
        Apply non-maximum suppression to remove overlapping boxes
        
        Args:
            boxes (list): List of (x, y, w, h) tuples
            overlap_thresh (float): Maximum allowed overlap ratio
            
        Returns:
            list: Filtered list of boxes
        """
        if not boxes:
            return []
        
        # Convert to numpy array
        boxes = np.array(boxes)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        # Compute area
        area = (x2 - x1) * (y2 - y1)
        
        # Sort by bottom-right y-coordinate
        idxs = np.argsort(y2)
        
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find the intersection
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Compute intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[idxs[:last]]
            
            # Delete overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick].tolist()

if __name__ == "__main__":
    detector = LogoDetector("logo_template.png")
    results = detector.process_directory("images")
    
    for image_path, matches in results.items():
        output_image = detector.visualize_results(image_path, matches)
        cv2.imshow("Matches", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()