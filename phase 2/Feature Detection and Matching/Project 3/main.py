"""
Create a panorama stitching application using feature detection and homography estimation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class PanoramaStitcher:
    def __init__(self, detector='sift', matcher='flann'):
        self.detector = self._get_detector(detector)
        self.matcher = self._get_matcher(matcher)
    
    def _get_detector(self, detector_type):
        if detector_type == 'sift':
            return cv2.SIFT_create()
        elif detector_type == 'orb':
            return cv2.ORB_create()
        else:
            raise ValueError("Unsupported detector type")
    
    def _get_matcher(self, matcher_type):
        if matcher_type == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'bf':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError("Unsupported matcher type")
    
    def _detect_and_compute_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def _match_features(self, desc1, desc2):
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:int(len(matches) * 0.3)]
        return good_matches
    
    def stitch_images(self, images):
        if len(images) < 2:
            raise ValueError("At least two images required for stitching")
        
        # Detect features for all images
        all_keypoints = []
        all_descriptors = []
        for img in images:
            kp, desc = self._detect_and_compute_features(img)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
        
        # Stitch images progressively
        result = images[0]
        for i in range(1, len(images)):
            # Match features between consecutive images
            good_matches = self._match_features(
                all_descriptors[i-1], 
                all_descriptors[i]
            )
            
            # Extract matched keypoints
            src_pts = np.float32([
                all_keypoints[i-1][m.queryIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                all_keypoints[i][m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 1, 2)
            
            # Find homography
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # Warp current result and next image
            h1, w1 = result.shape[:2]
            h2, w2 = images[i].shape[:2]
            
            # Compute panorama size dynamically
            pts1 = np.float32([[0,0], [0,h1-1], [w1-1,h1-1], [w1-1,0]]).reshape(-1,1,2)
            pts2 = np.float32([[0,0], [0,h2-1], [w2-1,h2-1], [w2-1,0]]).reshape(-1,1,2)
            
            # Transform points
            pts1_transformed = cv2.perspectiveTransform(pts1, H)
            
            # Concatenate points
            pts = np.concatenate((pts1_transformed, pts2), axis=0)
            
            # Get panorama bounds
            [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
            
            # Translation matrix
            t = [-x_min, -y_min]
            Ht = np.array([[1,0,t[0]], [0,1,t[1]], [0,0,1]])
            
            # Warp images
            warped_result = cv2.warpPerspective(result, Ht.dot(np.eye(3)), (x_max-x_min, y_max-y_min))
            warped_img = cv2.warpPerspective(images[i], Ht.dot(H), (x_max-x_min, y_max-y_min))
            
            # Blend images
            mask = warped_result > 0
            warped_img[mask] = warped_result[mask]
            result = warped_img
        
        return result

    def visualize_matches(self, img1, img2):
        kp1, desc1 = self._detect_and_compute_features(img1)
        kp2, desc2 = self._detect_and_compute_features(img2)
        
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:30]
        
        match_img = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            good_matches, 
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_img

def main():
    stitcher = PanoramaStitcher(detector='sift', matcher='flann')
    
    images = [
        cv2.imread('image1.jpg'),
        cv2.imread('image2.jpg'),
        cv2.imread('image3.jpg')
    ]
    
    # Stitch images
    panorama = stitcher.stitch_images(images)
    
    # Visualize result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title('Panorama Stitching Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save panorama
    cv2.imwrite('panorama_result.jpg', panorama)

if __name__ == '__main__':
    main()