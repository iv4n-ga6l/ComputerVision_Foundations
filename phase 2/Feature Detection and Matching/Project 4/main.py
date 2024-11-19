"""
Develop a Python program that matches logos or objects in a series of images using feature matching.
"""

import cv2
import numpy as np
import logging

class FeatureMatcher:
    def __init__(self, min_match_count=10):
        """
        Initialize the feature matcher with SIFT detector
        
        Args:
            min_match_count (int): Minimum number of good matches required
        """
        self.min_match_count = min_match_count
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger('FeatureMatcher')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def load_image(self, image_path):
        """
        Load and preprocess an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return img
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def detect_features(self, image):
        """
        Detect SIFT features in an image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match features between two sets of descriptors
        
        Args:
            desc1, desc2 (numpy.ndarray): Feature descriptors to match
            
        Returns:
            list: Good matches that pass the ratio test
        """
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
    
    def find_object(self, template_path, scene_path, output_path=None):
        """
        Find a template object in a scene image
        
        Args:
            template_path (str): Path to template image
            scene_path (str): Path to scene image
            output_path (str, optional): Path to save visualization
            
        Returns:
            tuple: (success boolean, homography matrix if found)
        """
        try:
            # Load images
            template_img = self.load_image(template_path)
            scene_img = self.load_image(scene_path)
            
            # Detect features
            kp1, desc1 = self.detect_features(template_img)
            kp2, desc2 = self.detect_features(scene_img)
            
            if desc1 is None or desc2 is None:
                self.logger.warning("No features detected in one or both images")
                return False, None
            
            # Match features
            good_matches = self.match_features(desc1, desc2)
            
            if len(good_matches) < self.min_match_count:
                self.logger.warning(f"Not enough good matches found: {len(good_matches)}/{self.min_match_count}")
                return False, None
            
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if output_path:
                # Draw matches visualization
                matchesMask = mask.ravel().tolist()
                h, w = template_img.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                
                # Draw bounding box
                scene_img_with_box = scene_img.copy()
                cv2.polylines(scene_img_with_box, [np.int32(dst)], True, (0, 255, 0), 3)
                
                # Draw matches
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=None,
                    matchesMask=matchesMask,
                    flags=2
                )
                img_matches = cv2.drawMatches(
                    template_img, kp1, scene_img_with_box, kp2, 
                    good_matches, None, **draw_params
                )
                
                cv2.imwrite(output_path, img_matches)
                self.logger.info(f"Saved visualization to {output_path}")
            
            return True, H
            
        except Exception as e:
            self.logger.error(f"Error in feature matching: {str(e)}")
            return False, None

def main():
    matcher = FeatureMatcher(min_match_count=10)
    
    template_path = "obj.png"
    scene_path = "scene.jpg"
    output_path = "matches_visualization.jpg"
    
    success, homography = matcher.find_object(template_path, scene_path, output_path)
    
    if success:
        print("Object found successfully!")
    else:
        print("Object not found or not enough matches.")

if __name__ == "__main__":
    main()