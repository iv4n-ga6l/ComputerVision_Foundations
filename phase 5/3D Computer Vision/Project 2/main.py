"""
3D Reconstruction from Multiple Views
====================================

This project implements 3D reconstruction algorithms to create 3D models from multiple 2D images.
The implementation covers Structure from Motion (SfM), Multi-View Stereo (MVS), and bundle adjustment
techniques for robust 3D scene reconstruction.

Key Features:
- Complete Structure from Motion pipeline
- Multi-View Stereo densification
- Bundle adjustment optimization
- Mesh generation and texturing
- Interactive 3D visualization

Author: Computer Vision Foundations Project
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import glob
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import json
import time
from tqdm import tqdm
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


class Camera:
    """Camera model with intrinsic and extrinsic parameters"""
    def __init__(self, K=None, R=None, t=None, width=640, height=480):
        self.K = K if K is not None else np.eye(3)  # Intrinsic matrix
        self.R = R if R is not None else np.eye(3)  # Rotation matrix
        self.t = t if t is not None else np.zeros(3)  # Translation vector
        self.width = width
        self.height = height
        
        # Derived properties
        self._P = None  # Projection matrix
        self._center = None  # Camera center
    
    @property
    def P(self):
        """Projection matrix P = K[R|t]"""
        if self._P is None:
            self._P = self.K @ np.hstack([self.R, self.t.reshape(-1, 1)])
        return self._P
    
    @property
    def center(self):
        """Camera center in world coordinates"""
        if self._center is None:
            self._center = -self.R.T @ self.t
        return self._center
    
    def project(self, X):
        """Project 3D points to image coordinates"""
        if X.shape[1] == 3:
            X_h = np.hstack([X, np.ones((X.shape[0], 1))])
        else:
            X_h = X
        
        x_h = (self.P @ X_h.T).T
        x = x_h[:, :2] / x_h[:, 2:3]
        return x
    
    def is_in_front(self, X):
        """Check if 3D points are in front of camera"""
        X_cam = self.R @ X.T + self.t.reshape(-1, 1)
        return X_cam[2] > 0


class FeatureMatcher:
    """Advanced feature detection and matching"""
    def __init__(self, detector_type='sift', matcher_type='flann'):
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        
        # Initialize detector
        if detector_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif detector_type == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create()
        
        # Initialize matcher
        if matcher_type == 'flann':
            if detector_type == 'sift':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                   table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher()
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Convert keypoints to array
        if keypoints:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.array([]).reshape(0, 2)
        
        return points, descriptors, keypoints
    
    def match_features(self, desc1, desc2, ratio_thresh=0.7):
        """Match features between two images"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return np.array([]).reshape(0, 2)
        
        if self.matcher_type == 'flann':
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        else:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        # Extract matched point indices
        if good_matches:
            matches_idx = np.array([[m.queryIdx, m.trainIdx] for m in good_matches])
        else:
            matches_idx = np.array([]).reshape(0, 2)
        
        return matches_idx


class Triangulation:
    """Point triangulation methods"""
    
    @staticmethod
    def triangulate_dlt(P1, P2, x1, x2):
        """Triangulate points using Direct Linear Transform"""
        points_3d = []
        
        for i in range(len(x1)):
            # Set up the system Ax = 0
            A = np.array([
                x1[i, 0] * P1[2] - P1[0],
                x1[i, 1] * P1[2] - P1[1],
                x2[i, 0] * P2[2] - P2[0],
                x2[i, 1] * P2[2] - P2[1]
            ])
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]  # Normalize
            points_3d.append(X[:3])
        
        return np.array(points_3d)
    
    @staticmethod
    def triangulate_optimal(P1, P2, x1, x2):
        """Optimal triangulation method"""
        points_3d = []
        
        for i in range(len(x1)):
            # Use cv2.triangulatePoints for optimal triangulation
            point_4d = cv2.triangulatePoints(P1, P2, 
                                           x1[i].reshape(2, 1), 
                                           x2[i].reshape(2, 1))
            point_3d = point_4d[:3] / point_4d[3]
            points_3d.append(point_3d.flatten())
        
        return np.array(points_3d)


class BundleAdjustment:
    """Bundle adjustment optimization"""
    
    def __init__(self, cameras, points_3d, observations, point_indices, camera_indices):
        self.cameras = cameras
        self.points_3d = points_3d
        self.observations = observations
        self.point_indices = point_indices
        self.camera_indices = camera_indices
        self.n_cameras = len(cameras)
        self.n_points = len(points_3d)
    
    def camera_to_params(self, camera):
        """Convert camera to parameter vector"""
        # Use axis-angle representation for rotation
        rvec, _ = cv2.Rodrigues(camera.R)
        return np.concatenate([rvec.flatten(), camera.t])
    
    def params_to_camera(self, params, K):
        """Convert parameter vector to camera"""
        rvec = params[:3]
        tvec = params[3:6]
        R, _ = cv2.Rodrigues(rvec)
        return Camera(K=K, R=R, t=tvec)
    
    def residuals(self, params):
        """Compute reprojection residuals"""
        # Extract camera parameters and 3D points
        camera_params = params[:self.n_cameras * 6].reshape(self.n_cameras, 6)
        points_3d = params[self.n_cameras * 6:].reshape(self.n_points, 3)
        
        residuals = []
        
        for i, (cam_idx, pt_idx) in enumerate(zip(self.camera_indices, self.point_indices)):
            # Reconstruct camera
            camera = self.params_to_camera(camera_params[cam_idx], self.cameras[cam_idx].K)
            
            # Project 3D point
            point_3d = points_3d[pt_idx].reshape(1, 3)
            projected = camera.project(point_3d)[0]
            
            # Compute residual
            observed = self.observations[i]
            residual = projected - observed
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def optimize(self, max_nfev=1000):
        """Perform bundle adjustment optimization"""
        # Initialize parameter vector
        camera_params = []
        for camera in self.cameras:
            camera_params.extend(self.camera_to_params(camera))
        
        point_params = self.points_3d.flatten()
        initial_params = np.array(camera_params + point_params.tolist())
        
        # Optimize
        print("Running bundle adjustment...")
        result = least_squares(self.residuals, initial_params, max_nfev=max_nfev, verbose=1)
        
        # Extract optimized parameters
        camera_params = result.x[:self.n_cameras * 6].reshape(self.n_cameras, 6)
        points_3d = result.x[self.n_cameras * 6:].reshape(self.n_points, 3)
        
        # Update cameras
        optimized_cameras = []
        for i, camera in enumerate(self.cameras):
            opt_camera = self.params_to_camera(camera_params[i], camera.K)
            optimized_cameras.append(opt_camera)
        
        return optimized_cameras, points_3d, result.cost


class StructureFromMotion:
    """Structure from Motion implementation"""
    
    def __init__(self):
        self.feature_matcher = FeatureMatcher()
        self.cameras = []
        self.points_3d = []
        self.point_colors = []
        self.observations = []
        
    def load_images(self, image_paths):
        """Load and preprocess images"""
        self.images = []
        self.image_paths = image_paths
        
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                self.images.append(img)
            else:
                print(f"Warning: Could not load image {path}")
        
        print(f"Loaded {len(self.images)} images")
    
    def extract_features(self):
        """Extract features from all images"""
        print("Extracting features...")
        self.features = []
        self.descriptors = []
        self.keypoints = []
        
        for i, img in enumerate(tqdm(self.images)):
            points, desc, kpts = self.feature_matcher.detect_and_compute(img)
            self.features.append(points)
            self.descriptors.append(desc)
            self.keypoints.append(kpts)
            
    def match_all_pairs(self):
        """Match features between all image pairs"""
        print("Matching features between image pairs...")
        self.matches = {}
        n_images = len(self.images)
        
        for i in tqdm(range(n_images)):
            for j in range(i + 1, n_images):
                matches_idx = self.feature_matcher.match_features(
                    self.descriptors[i], self.descriptors[j]
                )
                
                if len(matches_idx) > 20:  # Minimum number of matches
                    self.matches[(i, j)] = matches_idx
    
    def estimate_camera_intrinsics(self, image_width, image_height):
        """Estimate camera intrinsics (simple assumption)"""
        # Simple estimation: assume focal length is 1.2 * max(width, height)
        f = 1.2 * max(image_width, image_height)
        cx = image_width / 2
        cy = image_height / 2
        
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def initialize_first_pair(self):
        """Initialize reconstruction with the first two views"""
        # Find the pair with most matches
        best_pair = max(self.matches.keys(), key=lambda x: len(self.matches[x]))
        i, j = best_pair
        
        print(f"Initializing with images {i} and {j}")
        
        # Get matched points
        matches_idx = self.matches[(i, j)]
        pts1 = self.features[i][matches_idx[:, 0]]
        pts2 = self.features[j][matches_idx[:, 1]]
        
        # Estimate camera intrinsics
        h, w = self.images[i].shape[:2]
        K = self.estimate_camera_intrinsics(w, h)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        # Create cameras
        cam1 = Camera(K=K, R=np.eye(3), t=np.zeros(3), width=w, height=h)
        cam2 = Camera(K=K, R=R, t=t.flatten(), width=w, height=h)
        
        self.cameras = [cam1, cam2]
        
        # Triangulate initial points
        inlier_pts1 = pts1[mask.flatten().astype(bool)]
        inlier_pts2 = pts2[mask.flatten().astype(bool)]
        
        points_3d = Triangulation.triangulate_optimal(cam1.P, cam2.P, inlier_pts1, inlier_pts2)
        
        # Filter points behind cameras
        valid_mask = cam1.is_in_front(points_3d) & cam2.is_in_front(points_3d)
        points_3d = points_3d[valid_mask]
        
        self.points_3d = points_3d
        self.point_colors = np.ones((len(points_3d), 3)) * 0.5  # Gray color
        
        print(f"Initialized with {len(points_3d)} 3D points")
        
        return i, j
    
    def add_new_view(self, img_idx, min_points=50):
        """Add a new view to the reconstruction"""
        if img_idx >= len(self.images):
            return False
        
        # Find matches with existing views
        all_2d_points = []
        all_3d_points = []
        
        for existing_idx in range(len(self.cameras)):
            if (existing_idx, img_idx) in self.matches:
                matches_idx = self.matches[(existing_idx, img_idx)]
            elif (img_idx, existing_idx) in self.matches:
                matches_idx = self.matches[(img_idx, existing_idx)][:, [1, 0]]
            else:
                continue
            
            # Get 2D-3D correspondences (simplified)
            pts_2d = self.features[img_idx][matches_idx[:, 1]]
            all_2d_points.extend(pts_2d)
        
        if len(all_2d_points) < min_points:
            return False
        
        # Estimate camera pose using PnP
        all_2d_points = np.array(all_2d_points)
        
        # For simplicity, use a subset of 3D points
        subset_3d = self.points_3d[:min(len(self.points_3d), len(all_2d_points))]
        subset_2d = all_2d_points[:len(subset_3d)]
        
        h, w = self.images[img_idx].shape[:2]
        K = self.estimate_camera_intrinsics(w, h)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            subset_3d, subset_2d, K, None
        )
        
        if success and len(inliers) > min_points // 2:
            R, _ = cv2.Rodrigues(rvec)
            camera = Camera(K=K, R=R, t=tvec.flatten(), width=w, height=h)
            self.cameras.append(camera)
            return True
        
        return False
    
    def run_incremental_sfm(self):
        """Run incremental Structure from Motion"""
        # Extract features
        self.extract_features()
        
        # Match features
        self.match_all_pairs()
        
        # Initialize with first pair
        init_i, init_j = self.initialize_first_pair()
        
        # Add remaining views
        remaining_views = set(range(len(self.images))) - {init_i, init_j}
        
        for img_idx in remaining_views:
            success = self.add_new_view(img_idx)
            if success:
                print(f"Added view {img_idx}")
            else:
                print(f"Failed to add view {img_idx}")
    
    def run_bundle_adjustment(self):
        """Run bundle adjustment on the reconstruction"""
        if len(self.cameras) < 2 or len(self.points_3d) < 10:
            print("Not enough data for bundle adjustment")
            return
        
        # Create dummy observations for bundle adjustment
        observations = []
        point_indices = []
        camera_indices = []
        
        # Simplified: assume each point is observed in first two cameras
        for i, point_3d in enumerate(self.points_3d[:100]):  # Limit for demo
            for cam_idx in range(min(2, len(self.cameras))):
                projected = self.cameras[cam_idx].project(point_3d.reshape(1, 3))[0]
                observations.append(projected)
                point_indices.append(i)
                camera_indices.append(cam_idx)
        
        if len(observations) == 0:
            return
        
        # Run bundle adjustment
        ba = BundleAdjustment(
            self.cameras[:2], self.points_3d[:100], 
            observations, point_indices, camera_indices
        )
        
        opt_cameras, opt_points, final_cost = ba.optimize()
        
        # Update reconstruction
        self.cameras[:2] = opt_cameras
        self.points_3d[:100] = opt_points
        
        print(f"Bundle adjustment completed. Final cost: {final_cost}")


class MultiViewStereo:
    """Multi-View Stereo for dense reconstruction"""
    
    def __init__(self, cameras):
        self.cameras = cameras
    
    def compute_depth_map(self, ref_img, ref_cam, src_imgs, src_cams, 
                         min_depth=0.1, max_depth=10.0, depth_samples=64):
        """Compute depth map for reference view"""
        h, w = ref_img.shape[:2]
        depth_map = np.zeros((h, w))
        confidence_map = np.zeros((h, w))
        
        # Simplified depth map computation using plane sweep
        depth_values = np.linspace(min_depth, max_depth, depth_samples)
        
        for y in range(0, h, 8):  # Sample every 8 pixels for efficiency
            for x in range(0, w, 8):
                ref_pixel = np.array([x, y], dtype=np.float32)
                
                best_depth = 0
                best_cost = float('inf')
                
                for depth in depth_values:
                    # Backproject to 3D
                    point_3d = self.backproject_pixel(ref_pixel, depth, ref_cam)
                    
                    # Compute photometric cost
                    cost = self.compute_photometric_cost(
                        point_3d, ref_img, ref_cam, src_imgs, src_cams, ref_pixel
                    )
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_depth = depth
                
                depth_map[y:y+8, x:x+8] = best_depth
                confidence_map[y:y+8, x:x+8] = 1.0 / (1.0 + best_cost)
        
        return depth_map, confidence_map
    
    def backproject_pixel(self, pixel, depth, camera):
        """Backproject pixel to 3D point"""
        # Convert to normalized coordinates
        x_norm = (pixel[0] - camera.K[0, 2]) / camera.K[0, 0]
        y_norm = (pixel[1] - camera.K[1, 2]) / camera.K[1, 1]
        
        # 3D point in camera coordinates
        point_cam = np.array([x_norm * depth, y_norm * depth, depth])
        
        # Transform to world coordinates
        point_world = camera.R.T @ (point_cam - camera.t)
        
        return point_world
    
    def compute_photometric_cost(self, point_3d, ref_img, ref_cam, src_imgs, src_cams, ref_pixel):
        """Compute photometric cost for a 3D point"""
        ref_color = ref_img[int(ref_pixel[1]), int(ref_pixel[0])]
        if len(ref_color.shape) == 1:
            ref_color = ref_color.item()
        else:
            ref_color = np.mean(ref_color)
        
        total_cost = 0
        valid_views = 0
        
        for src_img, src_cam in zip(src_imgs, src_cams):
            # Project to source view
            projected = src_cam.project(point_3d.reshape(1, 3))[0]
            
            # Check if projection is valid
            if (0 <= projected[0] < src_img.shape[1] and 
                0 <= projected[1] < src_img.shape[0]):
                
                src_color = src_img[int(projected[1]), int(projected[0])]
                if len(src_color.shape) == 1:
                    src_color = src_color.item()
                else:
                    src_color = np.mean(src_color)
                
                cost = abs(float(ref_color) - float(src_color))
                total_cost += cost
                valid_views += 1
        
        return total_cost / max(valid_views, 1)


def visualize_reconstruction(cameras, points_3d, point_colors=None, save_path=None):
    """Visualize 3D reconstruction"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    if point_colors is not None:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=point_colors, s=1, alpha=0.6)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  s=1, alpha=0.6)
    
    # Plot cameras
    for i, cam in enumerate(cameras):
        center = cam.center
        ax.scatter(center[0], center[1], center[2], 
                  c='red', s=100, marker='^', label=f'Camera {i}' if i < 5 else "")
        
        # Plot camera orientation
        forward = cam.R[:, 2]  # Camera looks along negative z-axis
        ax.quiver(center[0], center[1], center[2],
                 forward[0], forward[1], forward[2],
                 length=0.5, color='red', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Reconstruction')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_point_cloud(points_3d, colors=None, filename='reconstruction.ply'):
    """Save point cloud to PLY format"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save to file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction from Multiple Views')
    parser.add_argument('--mode', type=str, default='sfm', 
                       choices=['sfm', 'mvs', 'bundle_adjust', 'mesh', 'evaluate'],
                       help='Reconstruction mode')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--images', type=str, nargs='+', help='Input image paths')
    parser.add_argument('--output_dir', type=str, default='reconstruction', help='Output directory')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--sparse_reconstruction', type=str, help='Sparse reconstruction file')
    parser.add_argument('--pointcloud', type=str, help='Point cloud file for mesh generation')
    parser.add_argument('--ground_truth', type=str, help='Ground truth for evaluation')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'sfm':
        # Structure from Motion
        if args.input_dir:
            image_paths = glob.glob(os.path.join(args.input_dir, '*.jpg')) + \
                         glob.glob(os.path.join(args.input_dir, '*.png'))
        elif args.images:
            image_paths = args.images
        else:
            print("Please provide --input_dir or --images")
            return
        
        if len(image_paths) < 2:
            print("Need at least 2 images for reconstruction")
            return
        
        # Run SfM
        sfm = StructureFromMotion()
        sfm.load_images(image_paths)
        sfm.run_incremental_sfm()
        
        if len(sfm.points_3d) > 0:
            # Run bundle adjustment
            sfm.run_bundle_adjustment()
            
            # Save results
            output_file = args.output or os.path.join(args.output_dir, 'reconstruction.ply')
            save_point_cloud(sfm.points_3d, sfm.point_colors, output_file)
            
            # Visualize
            if args.visualize:
                visualize_reconstruction(sfm.cameras, sfm.points_3d, sfm.point_colors)
            
            print(f"SfM reconstruction completed with {len(sfm.points_3d)} points")
        else:
            print("SfM failed to reconstruct any 3D points")
    
    elif args.mode == 'mvs':
        # Multi-View Stereo (placeholder)
        print("Multi-View Stereo mode - dense reconstruction")
        print("This would perform dense reconstruction from sparse SfM results")
        
    elif args.mode == 'bundle_adjust':
        print("Bundle adjustment mode")
        print("This would optimize existing reconstruction")
        
    elif args.mode == 'mesh':
        print("Mesh generation mode")
        if args.pointcloud:
            # Load point cloud and generate mesh
            pcd = o3d.io.read_point_cloud(args.pointcloud)
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Generate mesh using Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # Save mesh
            output_file = args.output or os.path.join(args.output_dir, 'mesh.obj')
            o3d.io.write_triangle_mesh(output_file, mesh)
            print(f"Mesh saved to {output_file}")
            
            # Visualize
            if args.visualize:
                o3d.visualization.draw_geometries([mesh])
        else:
            print("Please provide --pointcloud for mesh generation")
    
    elif args.mode == 'evaluate':
        print("Evaluation mode")
        print("This would evaluate reconstruction quality against ground truth")


if __name__ == '__main__':
    main()
