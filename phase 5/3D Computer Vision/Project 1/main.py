"""
Stereo Vision and Depth Estimation
=================================

A comprehensive implementation of stereo vision algorithms for depth estimation
from stereo image pairs, including camera calibration and disparity computation.

Features:
- Complete stereo vision pipeline
- Camera calibration and rectification
- Multiple disparity estimation algorithms
- Real-time stereo processing
- Depth map visualization and analysis

Author: Computer Vision Foundations Project
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from tqdm import tqdm
import json
from typing import Tuple, List, Optional
import seaborn as sns
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

class StereoCalibration:
    """Stereo camera calibration class"""
    
    def __init__(self, checkerboard_size=(9, 6), square_size=1.0):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration data
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane
        
        # Calibration results
        self.camera_matrix_left = None
        self.distortion_left = None
        self.camera_matrix_right = None
        self.distortion_right = None
        self.R = None  # Rotation matrix between cameras
        self.T = None  # Translation vector between cameras
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        
        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi_left = None
        self.roi_right = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
    
    def find_corners(self, left_images: List[str], right_images: List[str]) -> bool:
        """Find checkerboard corners in stereo image pairs"""
        print("Finding checkerboard corners...")
        
        for left_path, right_path in tqdm(zip(left_images, right_images)):
            # Read images
            img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            
            if img_left is None or img_right is None:
                print(f"Could not read images: {left_path}, {right_path}")
                continue
            
            # Find corners
            ret_left, corners_left = cv2.findChessboardCorners(
                img_left, self.checkerboard_size, None
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                img_right, self.checkerboard_size, None
            )
            
            if ret_left and ret_right:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
                
                # Store data
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
        
        print(f"Found {len(self.objpoints)} valid stereo pairs")
        return len(self.objpoints) > 0
    
    def calibrate_cameras(self, image_shape: Tuple[int, int]) -> bool:
        """Calibrate individual cameras and stereo system"""
        if len(self.objpoints) == 0:
            print("No calibration data available")
            return False
        
        print("Calibrating cameras...")
        
        # Calibrate individual cameras
        ret_left, self.camera_matrix_left, self.distortion_left, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_shape, None, None
        )
        
        ret_right, self.camera_matrix_right, self.distortion_right, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_shape, None, None
        )
        
        if not (ret_left and ret_right):
            print("Camera calibration failed")
            return False
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_stereo, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.camera_matrix_left, self.distortion_left,
            self.camera_matrix_right, self.distortion_right,
            image_shape, criteria=criteria, flags=flags
        )
        
        if ret_stereo:
            print("Stereo calibration successful")
            print(f"Reprojection error: {ret_stereo:.4f}")
            return True
        else:
            print("Stereo calibration failed")
            return False
    
    def compute_rectification(self, image_shape: Tuple[int, int]):
        """Compute rectification parameters"""
        print("Computing rectification parameters...")
        
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(
            self.camera_matrix_left, self.distortion_left,
            self.camera_matrix_right, self.distortion_right,
            image_shape, self.R, self.T, alpha=0
        )
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.distortion_left, self.R1, self.P1, image_shape, cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.distortion_right, self.R2, self.P2, image_shape, cv2.CV_16SC2
        )
    
    def rectify_images(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify stereo image pair"""
        rect_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def save_calibration(self, filename: str):
        """Save calibration parameters"""
        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left.tolist(),
            'distortion_left': self.distortion_left.tolist(),
            'camera_matrix_right': self.camera_matrix_right.tolist(),
            'distortion_right': self.distortion_right.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist(),
            'R1': self.R1.tolist(),
            'R2': self.R2.tolist(),
            'P1': self.P1.tolist(),
            'P2': self.P2.tolist(),
            'Q': self.Q.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration parameters"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix_left = np.array(data['camera_matrix_left'])
            self.distortion_left = np.array(data['distortion_left'])
            self.camera_matrix_right = np.array(data['camera_matrix_right'])
            self.distortion_right = np.array(data['distortion_right'])
            self.R = np.array(data['R'])
            self.T = np.array(data['T'])
            self.E = np.array(data['E'])
            self.F = np.array(data['F'])
            self.R1 = np.array(data['R1'])
            self.R2 = np.array(data['R2'])
            self.P1 = np.array(data['P1'])
            self.P2 = np.array(data['P2'])
            self.Q = np.array(data['Q'])
            
            print(f"Calibration loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False

class StereoMatcher:
    """Stereo matching algorithms for disparity estimation"""
    
    def __init__(self, method='sgbm'):
        self.method = method
        self.matcher = None
        self._create_matcher()
    
    def _create_matcher(self):
        """Create stereo matcher based on method"""
        if self.method == 'bm':
            self.matcher = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        elif self.method == 'sgbm':
            # Semi-Global Block Matching
            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*5,  # Must be divisible by 16
                blockSize=5,
                P1=8 * 3 * 5**2,  # 8*number_of_image_channels*blockSize^2
                P2=32 * 3 * 5**2,  # 32*number_of_image_channels*blockSize^2
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
    
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """Compute disparity map from rectified stereo pair"""
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Compute disparity
        disparity = self.matcher.compute(gray_left, gray_right)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Convert disparity to depth using Q matrix"""
        # Avoid division by zero
        disparity[disparity <= 0] = 0.1
        
        # Apply Q matrix transformation
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        depth = points_3d[:, :, 2]
        
        # Filter out invalid depths
        depth[depth <= 0] = 0
        depth[depth > 1000] = 0  # Remove very far points
        
        return depth

class StereoProcessor:
    """Complete stereo processing pipeline"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        self.calibration = StereoCalibration()
        self.matcher = StereoMatcher('sgbm')
        
        if calibration_file and os.path.exists(calibration_file):
            self.calibration.load_calibration(calibration_file)
    
    def calibrate_system(self, left_images: List[str], right_images: List[str], 
                        image_shape: Tuple[int, int], save_file: str = 'stereo_calibration.json'):
        """Calibrate stereo system"""
        # Find corners
        if not self.calibration.find_corners(left_images, right_images):
            return False
        
        # Calibrate cameras
        if not self.calibration.calibrate_cameras(image_shape):
            return False
        
        # Compute rectification
        self.calibration.compute_rectification(image_shape)
        
        # Save calibration
        self.calibration.save_calibration(save_file)
        
        return True
    
    def process_stereo_pair(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process stereo pair to get disparity and depth"""
        # Rectify images
        if self.calibration.map1_left is not None:
            rect_left, rect_right = self.calibration.rectify_images(img_left, img_right)
        else:
            rect_left, rect_right = img_left, img_right
        
        # Compute disparity
        disparity = self.matcher.compute_disparity(rect_left, rect_right)
        
        # Convert to depth
        if self.calibration.Q is not None:
            depth = self.matcher.disparity_to_depth(disparity, self.calibration.Q)
        else:
            depth = disparity.copy()
        
        return disparity, depth
    
    def visualize_results(self, img_left: np.ndarray, disparity: np.ndarray, 
                         depth: np.ndarray, save_path: Optional[str] = None):
        """Visualize stereo processing results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB) if len(img_left.shape) == 3 else img_left, cmap='gray')
        axes[0, 0].set_title('Left Image')
        axes[0, 0].axis('off')
        
        # Disparity map
        disp_vis = np.where(disparity > 0, disparity, np.nan)
        im1 = axes[0, 1].imshow(disp_vis, cmap='viridis')
        axes[0, 1].set_title('Disparity Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Depth map
        depth_vis = np.where(depth > 0, depth, np.nan)
        im2 = axes[1, 0].imshow(depth_vis, cmap='plasma')
        axes[1, 0].set_title('Depth Map')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Depth histogram
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            axes[1, 1].hist(valid_depth, bins=50, alpha=0.7)
            axes[1, 1].set_title('Depth Distribution')
            axes[1, 1].set_xlabel('Depth (units)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def calibrate_stereo_system(left_dir: str, right_dir: str, output_file: str = 'stereo_calibration.json'):
    """Calibrate stereo camera system"""
    # Get image files
    left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')) + 
                        glob.glob(os.path.join(left_dir, '*.png')))
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')) + 
                         glob.glob(os.path.join(right_dir, '*.png')))
    
    if len(left_images) != len(right_images):
        print(f"Mismatch in number of images: {len(left_images)} vs {len(right_images)}")
        return False
    
    if len(left_images) == 0:
        print("No images found")
        return False
    
    # Get image shape
    sample_img = cv2.imread(left_images[0])
    image_shape = (sample_img.shape[1], sample_img.shape[0])
    
    # Create processor and calibrate
    processor = StereoProcessor()
    success = processor.calibrate_system(left_images, right_images, image_shape, output_file)
    
    if success:
        print("Stereo calibration completed successfully")
    else:
        print("Stereo calibration failed")
    
    return success

def compute_depth_from_stereo(left_image_path: str, right_image_path: str, 
                             calibration_file: str = 'stereo_calibration.json',
                             output_dir: str = 'results'):
    """Compute depth from stereo image pair"""
    # Load images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        print("Could not load images")
        return
    
    # Create processor
    processor = StereoProcessor(calibration_file)
    
    # Process stereo pair
    disparity, depth = processor.process_stereo_pair(img_left, img_right)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    cv2.imwrite(os.path.join(output_dir, 'disparity.png'), 
                (disparity * 255 / disparity.max()).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'depth.png'), 
                (depth * 255 / depth.max()).astype(np.uint8))
    
    # Visualize results
    processor.visualize_results(img_left, disparity, depth, 
                               os.path.join(output_dir, 'stereo_results.png'))
    
    print(f"Results saved to {output_dir}/")

def evaluate_on_dataset(dataset_path: str, calibration_file: str = 'stereo_calibration.json'):
    """Evaluate stereo algorithm on dataset"""
    # This would typically load a standard dataset like KITTI or Middlebury
    # and compute evaluation metrics
    print(f"Evaluating on dataset: {dataset_path}")
    print("Dataset evaluation not implemented - placeholder for future development")

def realtime_stereo(left_camera: int = 0, right_camera: int = 1, 
                   calibration_file: str = 'stereo_calibration.json'):
    """Real-time stereo processing"""
    # Open cameras
    cap_left = cv2.VideoCapture(left_camera)
    cap_right = cv2.VideoCapture(right_camera)
    
    if not (cap_left.isOpened() and cap_right.isOpened()):
        print("Could not open cameras")
        return
    
    # Create processor
    processor = StereoProcessor(calibration_file)
    
    print("Starting real-time stereo processing. Press 'q' to quit.")
    
    while True:
        # Capture frames
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not (ret_left and ret_right):
            break
        
        # Process stereo pair
        disparity, depth = processor.process_stereo_pair(frame_left, frame_right)
        
        # Visualize
        disp_vis = cv2.applyColorMap((disparity * 255 / disparity.max()).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Display results
        cv2.imshow('Left', frame_left)
        cv2.imshow('Disparity', disp_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Stereo Vision and Depth Estimation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['calibrate', 'depth', 'evaluate', 'realtime'],
                       help='Mode to run')
    parser.add_argument('--left_images', type=str, help='Left images directory for calibration')
    parser.add_argument('--right_images', type=str, help='Right images directory for calibration')
    parser.add_argument('--left', type=str, help='Left image path')
    parser.add_argument('--right', type=str, help='Right image path')
    parser.add_argument('--calibration', type=str, default='stereo_calibration.json',
                       help='Calibration file path')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--dataset', type=str, help='Dataset path for evaluation')
    parser.add_argument('--left_camera', type=int, default=0, help='Left camera index')
    parser.add_argument('--right_camera', type=int, default=1, help='Right camera index')
    
    args = parser.parse_args()
    
    if args.mode == 'calibrate':
        if not args.left_images or not args.right_images:
            print("Please provide left and right image directories")
            return
        calibrate_stereo_system(args.left_images, args.right_images, args.calibration)
        
    elif args.mode == 'depth':
        if not args.left or not args.right:
            print("Please provide left and right image paths")
            return
        compute_depth_from_stereo(args.left, args.right, args.calibration, args.output)
        
    elif args.mode == 'evaluate':
        if not args.dataset:
            print("Please provide dataset path")
            return
        evaluate_on_dataset(args.dataset, args.calibration)
        
    elif args.mode == 'realtime':
        realtime_stereo(args.left_camera, args.right_camera, args.calibration)

if __name__ == "__main__":
    main()
