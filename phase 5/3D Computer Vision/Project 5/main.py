"""
SLAM Implementation
Author: Computer Vision Foundations Project
Description: Complete SLAM system with visual odometry, mapping, and loop closure
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import networkx as nx
from collections import defaultdict, deque
import time
import pickle
import os

class CameraModel:
    """Camera intrinsic and distortion parameters"""
    def __init__(self, fx, fy, cx, cy, k1=0, k2=0, p1=0, p2=0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        self.dist_coeffs = np.array([k1, k2, p1, p2])
        
    def project_points(self, points_3d):
        """Project 3D points to image plane"""
        if points_3d.shape[0] == 3:
            points_3d = points_3d.T
        
        # Perspective projection
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        
        return np.column_stack([u, v])
    
    def unproject_points(self, image_points, depth):
        """Unproject image points to 3D using depth"""
        if len(image_points.shape) == 1:
            image_points = image_points.reshape(1, -1)
        
        x = (image_points[:, 0] - self.cx) / self.fx
        y = (image_points[:, 1] - self.cy) / self.fy
        
        points_3d = np.column_stack([x * depth, y * depth, depth])
        return points_3d

class Landmark:
    """3D landmark with uncertainty and observations"""
    def __init__(self, position, covariance=None, descriptor=None):
        self.position = np.array(position)
        self.covariance = covariance if covariance is not None else np.eye(3) * 0.1
        self.descriptor = descriptor
        self.observations = []  # (keyframe_id, feature_idx, image_point)
        self.age = 0
        self.id = None
    
    def add_observation(self, keyframe_id, feature_idx, image_point):
        self.observations.append((keyframe_id, feature_idx, image_point))
        self.age += 1

class Keyframe:
    """Keyframe with pose, features, and descriptors"""
    def __init__(self, frame_id, image, pose, features, descriptors):
        self.frame_id = frame_id
        self.image = image
        self.pose = pose  # 4x4 transformation matrix
        self.features = features  # Nx2 array of keypoints
        self.descriptors = descriptors
        self.landmarks = []  # Associated landmark IDs
        self.covariance = np.eye(6) * 0.01  # Pose uncertainty
        
    def get_position(self):
        return self.pose[:3, 3]
    
    def get_rotation(self):
        return self.pose[:3, :3]

class FeatureTracker:
    """Feature detection and tracking"""
    def __init__(self, detector_type='ORB'):
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.array([]), np.array([])
        
        # Convert keypoints to array
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        return points, descriptors
    
    def match_features(self, desc1, desc2, ratio_threshold=0.8):
        """Match features between two frames"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return np.array([]), np.array([])
        
        matches = self.matcher.match(desc1, desc2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched indices
        if len(matches) > 0:
            matches_idx1 = np.array([m.queryIdx for m in matches])
            matches_idx2 = np.array([m.trainIdx for m in matches])
            return matches_idx1, matches_idx2
        else:
            return np.array([]), np.array([])

class PoseEstimator:
    """Camera pose estimation from correspondences"""
    def __init__(self, camera_model):
        self.camera = camera_model
    
    def estimate_pose_2d2d(self, points1, points2):
        """Estimate pose from 2D-2D correspondences"""
        if len(points1) < 8:
            return None, np.array([])
        
        # Estimate Essential matrix
        E, inliers = cv2.findEssentialMat(
            points1, points2, self.camera.K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None, np.array([])
        
        # Recover pose from Essential matrix
        _, R, t, mask = cv2.recoverPose(E, points1, points2, self.camera.K)
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        inlier_indices = np.where(mask.flatten() > 0)[0]
        
        return T, inlier_indices
    
    def estimate_pose_3d2d(self, points_3d, points_2d):
        """Estimate pose from 3D-2D correspondences using PnP"""
        if len(points_3d) < 4:
            return None, np.array([])
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float32),
            points_2d.astype(np.float32),
            self.camera.K,
            self.camera.dist_coeffs,
            confidence=0.99,
            reprojectionError=5.0
        )
        
        if not success:
            return None, np.array([])
        
        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        
        return T, inliers.flatten() if inliers is not None else np.array([])
    
    def triangulate_points(self, points1, points2, pose1, pose2):
        """Triangulate 3D points from two views"""
        # Projection matrices
        P1 = self.camera.K @ pose1[:3, :]
        P2 = self.camera.K @ pose2[:3, :]
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        
        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T

class LoopDetector:
    """Loop closure detection using bag-of-words"""
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
        self.vocabulary = None
        self.keyframe_descriptors = {}
        self.bow_vectors = {}
        
    def build_vocabulary(self, all_descriptors):
        """Build vocabulary from training descriptors"""
        if len(all_descriptors) == 0:
            return
        
        # Combine all descriptors
        combined_desc = np.vstack(all_descriptors)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            combined_desc.astype(np.float32),
            self.vocabulary_size,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        self.vocabulary = centers
    
    def compute_bow_vector(self, descriptors):
        """Compute bag-of-words vector for descriptors"""
        if self.vocabulary is None or descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocabulary_size)
        
        # Find nearest vocabulary words
        distances = cdist(descriptors, self.vocabulary)
        word_indices = np.argmin(distances, axis=1)
        
        # Create histogram
        bow_vector = np.bincount(word_indices, minlength=self.vocabulary_size)
        
        # Normalize
        if np.sum(bow_vector) > 0:
            bow_vector = bow_vector / np.sum(bow_vector)
        
        return bow_vector
    
    def add_keyframe(self, keyframe_id, descriptors):
        """Add keyframe to database"""
        self.keyframe_descriptors[keyframe_id] = descriptors
        bow_vector = self.compute_bow_vector(descriptors)
        self.bow_vectors[keyframe_id] = bow_vector
    
    def detect_loop_candidates(self, current_descriptors, min_score=0.1, min_gap=30):
        """Detect loop closure candidates"""
        current_bow = self.compute_bow_vector(current_descriptors)
        
        candidates = []
        for kf_id, bow_vec in self.bow_vectors.items():
            # Skip recent keyframes
            if len(self.bow_vectors) - kf_id < min_gap:
                continue
            
            # Compute similarity score
            score = np.dot(current_bow, bow_vec)
            
            if score > min_score:
                candidates.append((kf_id, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates

class BundleAdjustment:
    """Bundle adjustment optimization"""
    def __init__(self, camera_model):
        self.camera = camera_model
    
    def optimize_poses_and_points(self, keyframes, landmarks, observations):
        """Optimize camera poses and 3D points"""
        if len(keyframes) < 2 or len(landmarks) < 10:
            return
        
        # Prepare optimization variables
        pose_params = []
        point_params = []
        
        # Extract pose parameters (6DOF: rotation + translation)
        for kf in keyframes:
            rvec, _ = cv2.Rodrigues(kf.pose[:3, :3])
            tvec = kf.pose[:3, 3]
            pose_params.extend([rvec[0, 0], rvec[1, 0], rvec[2, 0], 
                              tvec[0], tvec[1], tvec[2]])
        
        # Extract 3D point parameters
        for landmark in landmarks:
            point_params.extend([landmark.position[0], 
                               landmark.position[1], 
                               landmark.position[2]])
        
        # Combine parameters
        x0 = np.array(pose_params + point_params)
        
        # Optimization
        result = least_squares(
            self._residual_function,
            x0,
            args=(keyframes, landmarks, observations),
            method='lm',
            max_nfev=100
        )
        
        # Update poses and points
        self._update_from_optimization(result.x, keyframes, landmarks)
    
    def _residual_function(self, params, keyframes, landmarks, observations):
        """Compute reprojection residuals"""
        n_poses = len(keyframes)
        pose_params = params[:n_poses * 6]
        point_params = params[n_poses * 6:]
        
        residuals = []
        
        for obs in observations:
            kf_idx, landmark_idx, image_point = obs
            
            # Get pose parameters
            pose_start = kf_idx * 6
            rvec = pose_params[pose_start:pose_start + 3]
            tvec = pose_params[pose_start + 3:pose_start + 6]
            
            # Get 3D point
            point_start = landmark_idx * 3
            point_3d = point_params[point_start:point_start + 3]
            
            # Project to image
            R, _ = cv2.Rodrigues(rvec)
            projected = self.camera.K @ (R @ point_3d + tvec)
            projected = projected[:2] / projected[2]
            
            # Compute residual
            residual = projected - image_point
            residuals.extend([residual[0], residual[1]])
        
        return np.array(residuals)
    
    def _update_from_optimization(self, params, keyframes, landmarks):
        """Update poses and landmarks from optimization result"""
        n_poses = len(keyframes)
        pose_params = params[:n_poses * 6]
        point_params = params[n_poses * 6:]
        
        # Update poses
        for i, kf in enumerate(keyframes):
            pose_start = i * 6
            rvec = pose_params[pose_start:pose_start + 3]
            tvec = pose_params[pose_start + 3:pose_start + 6]
            
            R, _ = cv2.Rodrigues(rvec)
            kf.pose[:3, :3] = R
            kf.pose[:3, 3] = tvec
        
        # Update landmarks
        for i, landmark in enumerate(landmarks):
            point_start = i * 3
            landmark.position = point_params[point_start:point_start + 3]

class VisualSLAM:
    """Main SLAM system"""
    def __init__(self, camera_model, config=None):
        self.camera = camera_model
        self.config = config or {}
        
        # Initialize components
        self.feature_tracker = FeatureTracker(
            self.config.get('detector_type', 'ORB')
        )
        self.pose_estimator = PoseEstimator(camera_model)
        self.loop_detector = LoopDetector(
            self.config.get('vocabulary_size', 1000)
        )
        self.bundle_adjuster = BundleAdjustment(camera_model)
        
        # SLAM state
        self.keyframes = {}
        self.landmarks = {}
        self.current_pose = np.eye(4)
        self.last_keyframe_id = -1
        self.frame_id = 0
        self.landmark_id = 0
        
        # Tracking state
        self.is_initialized = False
        self.tracking_quality = 0.0
        self.lost_tracking = False
        
        # Map management
        self.local_map_size = self.config.get('local_map_size', 20)
        self.keyframe_threshold = self.config.get('keyframe_threshold', 0.3)
        
        # Loop closure
        self.loop_closure_enabled = self.config.get('loop_closure', True)
        self.min_loop_gap = self.config.get('min_loop_gap', 30)
        
        # Visualization
        self.trajectory = []
        self.tracking_lost_frames = []
    
    def process_frame(self, image):
        """Process a single frame"""
        self.frame_id += 1
        
        # Extract features
        features, descriptors = self.feature_tracker.detect_and_compute(image)
        
        if not self.is_initialized:
            return self._initialize_slam(image, features, descriptors)
        
        # Track current frame
        pose, inliers = self._track_frame(features, descriptors)
        
        if pose is not None:
            self.current_pose = pose
            self.lost_tracking = False
            self.tracking_quality = len(inliers) / max(len(features), 1)
            
            # Check if keyframe is needed
            if self._should_add_keyframe(features, descriptors, inliers):
                self._add_keyframe(image, features, descriptors)
                
                # Perform local bundle adjustment
                self._local_bundle_adjustment()
                
                # Check for loop closure
                if self.loop_closure_enabled:
                    self._detect_and_process_loop_closure(descriptors)
        else:
            self.lost_tracking = True
            self.tracking_quality = 0.0
            self.tracking_lost_frames.append(self.frame_id)
        
        # Update trajectory
        self.trajectory.append(self.current_pose[:3, 3].copy())
        
        return {
            'pose': self.current_pose.copy(),
            'tracking_quality': self.tracking_quality,
            'lost_tracking': self.lost_tracking,
            'num_features': len(features),
            'num_landmarks': len(self.landmarks),
            'num_keyframes': len(self.keyframes)
        }
    
    def _initialize_slam(self, image, features, descriptors):
        """Initialize SLAM system with first two frames"""
        if len(features) < 100:
            return None
        
        # Store first frame
        if not hasattr(self, 'init_features'):
            self.init_features = features
            self.init_descriptors = descriptors
            self.init_image = image
            return None
        
        # Match with previous frame
        matches1, matches2 = self.feature_tracker.match_features(
            self.init_descriptors, descriptors
        )
        
        if len(matches1) < 50:
            return None
        
        # Estimate initial pose
        points1 = self.init_features[matches1]
        points2 = features[matches2]
        
        pose, inliers = self.pose_estimator.estimate_pose_2d2d(points1, points2)
        
        if pose is None or len(inliers) < 30:
            return None
        
        # Create initial keyframes
        init_pose = np.eye(4)
        kf1 = Keyframe(0, self.init_image, init_pose, 
                      self.init_features, self.init_descriptors)
        kf2 = Keyframe(1, image, pose, features, descriptors)
        
        self.keyframes[0] = kf1
        self.keyframes[1] = kf2
        self.last_keyframe_id = 1
        
        # Triangulate initial landmarks
        self._triangulate_initial_landmarks(kf1, kf2, matches1, matches2, inliers)
        
        self.current_pose = pose
        self.is_initialized = True
        
        # Initialize loop detector vocabulary
        self.loop_detector.add_keyframe(0, self.init_descriptors)
        self.loop_detector.add_keyframe(1, descriptors)
        
        return {
            'pose': self.current_pose.copy(),
            'tracking_quality': 1.0,
            'lost_tracking': False,
            'num_features': len(features),
            'num_landmarks': len(self.landmarks),
            'num_keyframes': len(self.keyframes)
        }
    
    def _track_frame(self, features, descriptors):
        """Track current frame against map"""
        if len(self.landmarks) == 0:
            return None, np.array([])
        
        # Get recent keyframes for tracking
        recent_kf_ids = sorted(self.keyframes.keys())[-5:]
        
        best_pose = None
        best_inliers = np.array([])
        
        for kf_id in recent_kf_ids:
            kf = self.keyframes[kf_id]
            
            # Match features with keyframe
            matches1, matches2 = self.feature_tracker.match_features(
                kf.descriptors, descriptors
            )
            
            if len(matches1) < 10:
                continue
            
            # Get 3D-2D correspondences
            points_3d = []
            points_2d = []
            
            for i, j in zip(matches1, matches2):
                if i < len(kf.landmarks) and kf.landmarks[i] is not None:
                    landmark_id = kf.landmarks[i]
                    if landmark_id in self.landmarks:
                        landmark = self.landmarks[landmark_id]
                        points_3d.append(landmark.position)
                        points_2d.append(features[j])
            
            if len(points_3d) < 6:
                continue
            
            # Estimate pose
            pose, inliers = self.pose_estimator.estimate_pose_3d2d(
                np.array(points_3d), np.array(points_2d)
            )
            
            if pose is not None and len(inliers) > len(best_inliers):
                best_pose = pose
                best_inliers = inliers
        
        return best_pose, best_inliers
    
    def _should_add_keyframe(self, features, descriptors, inliers):
        """Determine if a new keyframe should be added"""
        if len(self.keyframes) == 0:
            return True
        
        # Check tracking quality
        if self.tracking_quality < self.keyframe_threshold:
            return True
        
        # Check translation from last keyframe
        last_kf = self.keyframes[self.last_keyframe_id]
        translation = np.linalg.norm(
            self.current_pose[:3, 3] - last_kf.pose[:3, 3]
        )
        
        if translation > 0.1:  # 10cm threshold
            return True
        
        # Check rotation from last keyframe
        R_rel = last_kf.pose[:3, :3].T @ self.current_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        
        if angle > 0.1:  # ~6 degrees
            return True
        
        return False
    
    def _add_keyframe(self, image, features, descriptors):
        """Add new keyframe to map"""
        kf_id = len(self.keyframes)
        kf = Keyframe(kf_id, image, self.current_pose.copy(), 
                     features, descriptors)
        
        self.keyframes[kf_id] = kf
        self.last_keyframe_id = kf_id
        
        # Initialize landmark associations
        kf.landmarks = [None] * len(features)
        
        # Try to associate with existing landmarks
        self._associate_landmarks(kf)
        
        # Create new landmarks from unassociated features
        self._create_new_landmarks(kf)
        
        # Add to loop detector
        self.loop_detector.add_keyframe(kf_id, descriptors)
    
    def _associate_landmarks(self, keyframe):
        """Associate keyframe features with existing landmarks"""
        if len(self.landmarks) == 0:
            return
        
        # Project existing landmarks to keyframe
        landmark_positions = np.array([lm.position for lm in self.landmarks.values()])
        landmark_ids = list(self.landmarks.keys())
        
        if len(landmark_positions) == 0:
            return
        
        projected_points = self.camera.project_points(
            (keyframe.pose[:3, :3].T @ (landmark_positions - keyframe.pose[:3, 3]).T).T
        )
        
        # Find associations using nearest neighbor
        for i, feature in enumerate(keyframe.features):
            distances = np.linalg.norm(projected_points - feature, axis=1)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < 10.0:  # 10 pixel threshold
                landmark_id = landmark_ids[min_idx]
                keyframe.landmarks[i] = landmark_id
                self.landmarks[landmark_id].add_observation(
                    keyframe.frame_id, i, feature
                )
    
    def _create_new_landmarks(self, keyframe):
        """Create new landmarks from unassociated features"""
        if len(self.keyframes) < 2:
            return
        
        # Find previous keyframe for triangulation
        prev_kf_id = max(kf_id for kf_id in self.keyframes.keys() 
                        if kf_id < keyframe.frame_id)
        prev_kf = self.keyframes[prev_kf_id]
        
        # Match features between keyframes
        matches1, matches2 = self.feature_tracker.match_features(
            prev_kf.descriptors, keyframe.descriptors
        )
        
        for i, j in zip(matches1, matches2):
            # Skip if already associated
            if (i < len(prev_kf.landmarks) and prev_kf.landmarks[i] is not None) or \
               keyframe.landmarks[j] is not None:
                continue
            
            # Triangulate new landmark
            points1 = prev_kf.features[i:i+1]
            points2 = keyframe.features[j:j+1]
            
            points_3d = self.pose_estimator.triangulate_points(
                points1, points2, prev_kf.pose, keyframe.pose
            )
            
            if len(points_3d) > 0:
                # Create new landmark
                landmark = Landmark(points_3d[0])
                landmark.id = self.landmark_id
                landmark.add_observation(prev_kf.frame_id, i, points1[0])
                landmark.add_observation(keyframe.frame_id, j, points2[0])
                
                self.landmarks[self.landmark_id] = landmark
                prev_kf.landmarks[i] = self.landmark_id
                keyframe.landmarks[j] = self.landmark_id
                
                self.landmark_id += 1
    
    def _triangulate_initial_landmarks(self, kf1, kf2, matches1, matches2, inliers):
        """Create initial landmarks from first two keyframes"""
        inlier_matches1 = matches1[inliers]
        inlier_matches2 = matches2[inliers]
        
        points1 = kf1.features[inlier_matches1]
        points2 = kf2.features[inlier_matches2]
        
        points_3d = self.pose_estimator.triangulate_points(
            points1, points2, kf1.pose, kf2.pose
        )
        
        # Initialize landmarks
        kf1.landmarks = [None] * len(kf1.features)
        kf2.landmarks = [None] * len(kf2.features)
        
        for i, (idx1, idx2) in enumerate(zip(inlier_matches1, inlier_matches2)):
            landmark = Landmark(points_3d[i])
            landmark.id = self.landmark_id
            landmark.add_observation(kf1.frame_id, idx1, points1[i])
            landmark.add_observation(kf2.frame_id, idx2, points2[i])
            
            self.landmarks[self.landmark_id] = landmark
            kf1.landmarks[idx1] = self.landmark_id
            kf2.landmarks[idx2] = self.landmark_id
            
            self.landmark_id += 1
    
    def _local_bundle_adjustment(self):
        """Perform local bundle adjustment"""
        if len(self.keyframes) < 3:
            return
        
        # Select recent keyframes
        recent_kf_ids = sorted(self.keyframes.keys())[-self.local_map_size:]
        recent_keyframes = [self.keyframes[kf_id] for kf_id in recent_kf_ids]
        
        # Collect associated landmarks
        landmark_ids = set()
        for kf in recent_keyframes:
            for lm_id in kf.landmarks:
                if lm_id is not None:
                    landmark_ids.add(lm_id)
        
        if len(landmark_ids) == 0:
            return
        
        landmarks = [self.landmarks[lm_id] for lm_id in landmark_ids]
        
        # Collect observations
        observations = []
        for kf_idx, kf in enumerate(recent_keyframes):
            for feat_idx, lm_id in enumerate(kf.landmarks):
                if lm_id in landmark_ids:
                    lm_idx = list(landmark_ids).index(lm_id)
                    observations.append((kf_idx, lm_idx, kf.features[feat_idx]))
        
        # Optimize
        self.bundle_adjuster.optimize_poses_and_points(
            recent_keyframes, landmarks, observations
        )
    
    def _detect_and_process_loop_closure(self, descriptors):
        """Detect and process loop closures"""
        candidates = self.loop_detector.detect_loop_candidates(
            descriptors, min_gap=self.min_loop_gap
        )
        
        if len(candidates) == 0:
            return
        
        # Verify best candidate geometrically
        best_candidate, score = candidates[0]
        
        if self._verify_loop_closure(best_candidate, descriptors):
            self._perform_loop_closure(best_candidate)
    
    def _verify_loop_closure(self, candidate_id, current_descriptors):
        """Verify loop closure candidate geometrically"""
        candidate_kf = self.keyframes[candidate_id]
        
        # Match features
        matches1, matches2 = self.feature_tracker.match_features(
            candidate_kf.descriptors, current_descriptors
        )
        
        if len(matches1) < 20:
            return False
        
        # Try to estimate relative pose
        points1 = candidate_kf.features[matches1]
        points2 = self.keyframes[self.last_keyframe_id].features[matches2]
        
        pose, inliers = self.pose_estimator.estimate_pose_2d2d(points1, points2)
        
        return pose is not None and len(inliers) > 15
    
    def _perform_loop_closure(self, loop_keyframe_id):
        """Perform loop closure correction"""
        print(f"Loop closure detected with keyframe {loop_keyframe_id}")
        
        # Perform global bundle adjustment
        all_keyframes = list(self.keyframes.values())
        all_landmarks = list(self.landmarks.values())
        
        # Collect all observations
        observations = []
        for kf_idx, kf in enumerate(all_keyframes):
            for feat_idx, lm_id in enumerate(kf.landmarks):
                if lm_id is not None and lm_id in self.landmarks:
                    lm_idx = list(self.landmarks.keys()).index(lm_id)
                    observations.append((kf_idx, lm_idx, kf.features[feat_idx]))
        
        # Global optimization
        self.bundle_adjuster.optimize_poses_and_points(
            all_keyframes, all_landmarks, observations
        )
    
    def get_trajectory(self):
        """Get camera trajectory"""
        return np.array(self.trajectory)
    
    def get_map_points(self):
        """Get 3D map points"""
        if len(self.landmarks) == 0:
            return np.array([])
        
        return np.array([lm.position for lm in self.landmarks.values()])
    
    def visualize_trajectory(self):
        """Visualize camera trajectory"""
        if len(self.trajectory) == 0:
            print("No trajectory to visualize")
            return
        
        trajectory = np.array(self.trajectory)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, label='Camera Trajectory')
        
        # Plot keyframes
        keyframe_positions = np.array([kf.get_position() for kf in self.keyframes.values()])
        if len(keyframe_positions) > 0:
            ax.scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], 
                      keyframe_positions[:, 2], c='red', s=50, label='Keyframes')
        
        # Plot map points
        map_points = self.get_map_points()
        if len(map_points) > 0:
            ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], 
                      c='green', s=1, alpha=0.6, label='Map Points')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('SLAM Trajectory and Map')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_tracking_quality(self):
        """Visualize tracking quality over time"""
        if len(self.trajectory) == 0:
            return
        
        # Get tracking quality history
        frames = range(len(self.trajectory))
        quality = [0.5] * len(self.trajectory)  # Placeholder
        
        plt.figure(figsize=(12, 6))
        plt.plot(frames, quality, 'b-', linewidth=2)
        plt.axhline(y=self.keyframe_threshold, color='r', linestyle='--', 
                   label=f'Keyframe Threshold ({self.keyframe_threshold})')
        
        # Mark lost tracking frames
        if self.tracking_lost_frames:
            for frame in self.tracking_lost_frames:
                if frame < len(quality):
                    plt.axvline(x=frame, color='red', alpha=0.5)
        
        plt.xlabel('Frame Number')
        plt.ylabel('Tracking Quality')
        plt.title('SLAM Tracking Quality Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_map(self, filename):
        """Save SLAM map to file"""
        map_data = {
            'keyframes': self.keyframes,
            'landmarks': self.landmarks,
            'trajectory': self.trajectory,
            'camera_params': {
                'K': self.camera.K,
                'dist_coeffs': self.camera.dist_coeffs
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)
        
        print(f"Map saved to {filename}")
    
    def load_map(self, filename):
        """Load SLAM map from file"""
        with open(filename, 'rb') as f:
            map_data = pickle.load(f)
        
        self.keyframes = map_data['keyframes']
        self.landmarks = map_data['landmarks']
        self.trajectory = map_data['trajectory']
        
        print(f"Map loaded from {filename}")

def demo_slam():
    """Demonstrate SLAM system"""
    print("SLAM Implementation Demo")
    print("=" * 50)
    
    # Camera parameters (typical values)
    camera = CameraModel(fx=718.856, fy=718.856, cx=607.1928, cy=185.2157)
    
    # SLAM configuration
    config = {
        'detector_type': 'ORB',
        'vocabulary_size': 1000,
        'local_map_size': 20,
        'keyframe_threshold': 0.3,
        'loop_closure': True,
        'min_loop_gap': 30
    }
    
    # Initialize SLAM system
    slam = VisualSLAM(camera, config)
    
    # Simulate camera motion and features
    print("\nSimulating camera motion...")
    
    # Generate synthetic trajectory
    n_frames = 100
    trajectory_gt = []
    
    for i in range(n_frames):
        # Circular motion
        angle = i * 0.1
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        z = 0.1 * i
        
        # Camera pose
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])
        t = np.array([x, y, z])
        
        pose_gt = np.eye(4)
        pose_gt[:3, :3] = R
        pose_gt[:3, 3] = t
        trajectory_gt.append(pose_gt)
    
    # Generate synthetic images with features
    image_size = (640, 480)
    
    for i in range(min(20, n_frames)):  # Process subset for demo
        # Create synthetic image
        image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add some structure (corners, lines)
        for j in range(50):
            pt1 = (np.random.randint(0, image_size[0]), 
                   np.random.randint(0, image_size[1]))
            pt2 = (np.random.randint(0, image_size[0]), 
                   np.random.randint(0, image_size[1]))
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)
        
        # Process frame
        result = slam.process_frame(image)
        
        if result:
            print(f"Frame {i:2d}: Features={result['num_features']:3d}, "
                  f"Landmarks={result['num_landmarks']:3d}, "
                  f"Keyframes={result['num_keyframes']:2d}, "
                  f"Quality={result['tracking_quality']:.2f}")
    
    # Visualization
    print("\nSLAM Results:")
    print(f"Total keyframes: {len(slam.keyframes)}")
    print(f"Total landmarks: {len(slam.landmarks)}")
    print(f"Final trajectory length: {len(slam.trajectory)}")
    
    # Visualize results
    slam.visualize_trajectory()
    slam.visualize_tracking_quality()
    
    # Save map
    slam.save_map('demo_slam_map.pkl')
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    demo_slam()
