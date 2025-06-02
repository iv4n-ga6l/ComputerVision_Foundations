"""
Point Cloud Processing
=====================

A comprehensive implementation of point cloud processing algorithms for 3D data analysis,
including registration, filtering, segmentation, and surface reconstruction techniques.

Features:
- Multiple registration algorithms (ICP, feature-based)
- Robust filtering and outlier removal
- Plane and primitive detection
- Clustering and segmentation
- Surface reconstruction and mesh generation
- Visualization and analysis tools

 Project
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
import glob
import time
from tqdm import tqdm
import copy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')


class PointCloudRegistration:
    """Point cloud registration algorithms"""
    
    def __init__(self):
        self.max_iterations = 100
        self.tolerance = 1e-6
        
    def icp_point_to_point(self, source, target, initial_transform=None, max_iterations=100):
        """Iterative Closest Point registration (point-to-point)"""
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Convert to Open3D format if needed
        if isinstance(source, np.ndarray):
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source)
        else:
            source_pcd = source
            
        if isinstance(target, np.ndarray):
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target)
        else:
            target_pcd = target
        
        # ICP registration
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.02, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse
    
    def icp_point_to_plane(self, source, target, initial_transform=None, max_iterations=100):
        """ICP registration with point-to-plane distance"""
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Ensure point clouds have normals
        if not target.has_normals():
            target.estimate_normals()
        if not source.has_normals():
            source.estimate_normals()
        
        # ICP point-to-plane
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, 0.02, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        return reg_p2l.transformation, reg_p2l.fitness, reg_p2l.inlier_rmse
    
    def feature_based_registration(self, source, target, voxel_size=0.05):
        """Feature-based registration using FPFH descriptors"""
        # Downsample
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 2
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # Compute FPFH features
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        # RANSAC registration
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        return result.transformation, result.fitness, result.inlier_rmse
    
    def global_registration(self, source, target, voxel_size=0.05):
        """Global registration using Fast Global Registration"""
        # Downsample and compute features
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 2
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # Compute FPFH features
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        # Fast Global Registration
        distance_threshold = voxel_size * 0.5
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        
        return result.transformation, result.fitness, result.inlier_rmse


class PointCloudFiltering:
    """Point cloud filtering and preprocessing"""
    
    @staticmethod
    def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
        """Remove statistical outliers"""
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, 
                                                  std_ratio=std_ratio)
        return cl, ind
    
    @staticmethod
    def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
        """Remove radius outliers"""
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return cl, ind
    
    @staticmethod
    def voxel_down_sample(pcd, voxel_size=0.02):
        """Voxel grid downsampling"""
        return pcd.voxel_down_sample(voxel_size)
    
    @staticmethod
    def uniform_down_sample(pcd, every_k_points=5):
        """Uniform downsampling"""
        return pcd.uniform_down_sample(every_k_points)
    
    @staticmethod
    def bilateral_filter(pcd, kernel_size=5, sigma_s=0.1, sigma_r=0.1):
        """Bilateral filtering for point clouds"""
        # Custom implementation using KNN
        points = np.asarray(pcd.points)
        filtered_points = []
        
        # Build KDTree for efficient neighbor search
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        for i, point in enumerate(points):
            # Find neighbors
            [k, idx, _] = tree.search_knn_vector_3d(point, kernel_size)
            
            if k > 1:
                neighbors = points[idx[1:]]  # Exclude self
                
                # Compute weights
                spatial_weights = np.exp(-np.linalg.norm(neighbors - point, axis=1)**2 / (2 * sigma_s**2))
                intensity_weights = np.exp(-np.linalg.norm(neighbors - point, axis=1)**2 / (2 * sigma_r**2))
                weights = spatial_weights * intensity_weights
                
                # Weighted average
                if np.sum(weights) > 0:
                    filtered_point = np.average(neighbors, axis=0, weights=weights)
                else:
                    filtered_point = point
            else:
                filtered_point = point
            
            filtered_points.append(filtered_point)
        
        # Create filtered point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        
        if pcd.has_colors():
            filtered_pcd.colors = pcd.colors
        if pcd.has_normals():
            filtered_pcd.normals = pcd.normals
        
        return filtered_pcd


class PointCloudSegmentation:
    """Point cloud segmentation algorithms"""
    
    @staticmethod
    def plane_segmentation_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """RANSAC-based plane segmentation"""
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        
        return plane_model, inlier_cloud, outlier_cloud, inliers
    
    @staticmethod
    def euclidean_clustering(pcd, eps=0.02, min_points=10):
        """Euclidean clustering using DBSCAN"""
        points = np.asarray(pcd.points)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = clustering.labels_
        
        # Organize clusters
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            clusters.append(cluster_pcd)
        
        return clusters, labels
    
    @staticmethod
    def region_growing(pcd, min_pts_per_cluster=50, max_pts_per_cluster=1000000, 
                      neighbors=30, smoothness_threshold=30.0, curvature_threshold=1.0):
        """Region growing segmentation"""
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Convert angles to radians
        smoothness_threshold = np.deg2rad(smoothness_threshold)
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_points = len(points)
        
        # Build KDTree for neighbor search
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # Initialize
        processed = np.zeros(n_points, dtype=bool)
        clusters = []
        
        # Compute curvature (simplified)
        curvatures = np.zeros(n_points)
        for i in range(n_points):
            [k, idx, _] = tree.search_knn_vector_3d(points[i], neighbors)
            if k > 1:
                neighbor_normals = normals[idx]
                curvatures[i] = np.std(np.dot(neighbor_normals, normals[i]))
        
        # Region growing
        for seed_idx in range(n_points):
            if processed[seed_idx] or curvatures[seed_idx] > curvature_threshold:
                continue
            
            # Start new region
            region = [seed_idx]
            current_seeds = [seed_idx]
            processed[seed_idx] = True
            
            while current_seeds and len(region) < max_pts_per_cluster:
                new_seeds = []
                
                for seed in current_seeds:
                    [k, idx, _] = tree.search_knn_vector_3d(points[seed], neighbors)
                    
                    for neighbor_idx in idx[1:]:  # Skip self
                        if processed[neighbor_idx]:
                            continue
                        
                        # Check smoothness constraint
                        angle = np.arccos(np.clip(np.dot(normals[seed], normals[neighbor_idx]), -1, 1))
                        
                        if angle < smoothness_threshold:
                            region.append(neighbor_idx)
                            processed[neighbor_idx] = True
                            
                            # Add as seed if curvature is low
                            if curvatures[neighbor_idx] < curvature_threshold:
                                new_seeds.append(neighbor_idx)
                
                current_seeds = new_seeds
            
            # Add region if it's large enough
            if len(region) >= min_pts_per_cluster:
                cluster_pcd = pcd.select_by_index(region)
                clusters.append(cluster_pcd)
        
        return clusters
    
    @staticmethod
    def cylinder_segmentation_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """RANSAC-based cylinder segmentation"""
        # This is a simplified version - Open3D doesn't have built-in cylinder RANSAC
        # We'll detect cylinders by finding points that are equidistant from an axis
        
        points = np.asarray(pcd.points)
        n_points = len(points)
        
        best_inliers = []
        best_model = None
        
        for _ in range(num_iterations):
            # Random sample
            sample_indices = np.random.choice(n_points, ransac_n, replace=False)
            sample_points = points[sample_indices]
            
            # Fit axis (simplified - using PCA)
            centroid = np.mean(sample_points, axis=0)
            pca = PCA(n_components=3)
            pca.fit(sample_points - centroid)
            axis = pca.components_[0]  # Primary direction
            
            # Find points close to cylinder surface
            inliers = []
            for i, point in enumerate(points):
                # Distance from point to axis
                vec_to_point = point - centroid
                proj_length = np.dot(vec_to_point, axis)
                closest_on_axis = centroid + proj_length * axis
                dist_to_axis = np.linalg.norm(point - closest_on_axis)
                
                # Check if points form a cylinder (within distance threshold of some radius)
                if len(inliers) == 0:
                    expected_radius = dist_to_axis
                    inliers.append(i)
                elif abs(dist_to_axis - expected_radius) < distance_threshold:
                    inliers.append(i)
            
            # Update best model
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (centroid, axis, expected_radius if 'expected_radius' in locals() else 0)
        
        if best_inliers:
            inlier_cloud = pcd.select_by_index(best_inliers)
            outlier_cloud = pcd.select_by_index(best_inliers, invert=True)
            return best_model, inlier_cloud, outlier_cloud, best_inliers
        else:
            return None, None, pcd, []


class SurfaceReconstruction:
    """Surface reconstruction algorithms"""
    
    @staticmethod
    def poisson_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
        """Poisson surface reconstruction"""
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
        
        return mesh, densities
    
    @staticmethod
    def alpha_shape_reconstruction(pcd, alpha=0.1):
        """Alpha shape reconstruction"""
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        return mesh
    
    @staticmethod
    def ball_pivoting_reconstruction(pcd, radii=None):
        """Ball pivoting algorithm"""
        if radii is None:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [0.5 * avg_dist, avg_dist, 2 * avg_dist, 4 * avg_dist]
        
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        return mesh
    
    @staticmethod
    def delaunay_2d_reconstruction(pcd):
        """2D Delaunay triangulation (for height fields)"""
        points = np.asarray(pcd.points)
        
        # Project to 2D (XY plane)
        points_2d = points[:, :2]
        
        # Delaunay triangulation
        try:
            hull = ConvexHull(points_2d)
            triangles = hull.simplices
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            return mesh
        except:
            return None


class PointCloudAnalysis:
    """Point cloud analysis and feature extraction"""
    
    @staticmethod
    def compute_geometric_features(pcd, radius=0.05):
        """Compute geometric features for each point"""
        points = np.asarray(pcd.points)
        n_points = len(points)
        
        # Build KDTree
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        features = {
            'curvature': np.zeros(n_points),
            'density': np.zeros(n_points),
            'roughness': np.zeros(n_points),
            'planarity': np.zeros(n_points)
        }
        
        for i in range(n_points):
            # Find neighbors
            [k, idx, dists] = tree.search_radius_vector_3d(points[i], radius)
            
            if k > 3:
                neighbors = points[idx[1:]]  # Exclude self
                centroid = np.mean(neighbors, axis=0)
                
                # Density
                features['density'][i] = k / (4/3 * np.pi * radius**3)
                
                # Covariance matrix
                cov_matrix = np.cov(neighbors.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                
                # Curvature (smallest eigenvalue / sum of eigenvalues)
                if np.sum(eigenvals) > 0:
                    features['curvature'][i] = eigenvals[2] / np.sum(eigenvals)
                
                # Planarity
                if eigenvals[0] > 0:
                    features['planarity'][i] = (eigenvals[1] - eigenvals[2]) / eigenvals[0]
                
                # Roughness (standard deviation of distances to fitted plane)
                if len(neighbors) > 3:
                    # Fit plane using PCA
                    normal = eigenvecs[:, 2]  # Normal is eigenvector with smallest eigenvalue
                    distances_to_plane = np.abs(np.dot(neighbors - centroid, normal))
                    features['roughness'][i] = np.std(distances_to_plane)
        
        return features
    
    @staticmethod
    def compute_fpfh_features(pcd, radius=0.1):
        """Compute Fast Point Feature Histogram (FPFH) descriptors"""
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
        
        return np.array(fpfh.data).T
    
    @staticmethod
    def compute_shot_features(pcd, radius=0.1):
        """Compute SHOT (Signature of Histograms of Orientations) descriptors"""
        # Note: Open3D doesn't have SHOT, this is a placeholder
        # In practice, you would use PCL python bindings or implement SHOT
        print("SHOT features not implemented in this demo")
        return None
    
    @staticmethod
    def analyze_point_cloud_quality(pcd):
        """Analyze point cloud quality metrics"""
        points = np.asarray(pcd.points)
        
        # Basic statistics
        stats = {
            'num_points': len(points),
            'bounding_box': pcd.get_axis_aligned_bounding_box(),
            'centroid': np.mean(points, axis=0),
            'std_dev': np.std(points, axis=0)
        }
        
        # Density analysis
        distances = pcd.compute_nearest_neighbor_distance()
        stats['avg_nearest_neighbor_distance'] = np.mean(distances)
        stats['std_nearest_neighbor_distance'] = np.std(distances)
        
        # Uniformity (coefficient of variation of nearest neighbor distances)
        stats['uniformity'] = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        
        return stats


class PointCloudProcessor:
    """Main point cloud processing pipeline"""
    
    def __init__(self):
        self.registration = PointCloudRegistration()
        self.filtering = PointCloudFiltering()
        self.segmentation = PointCloudSegmentation()
        self.reconstruction = SurfaceReconstruction()
        self.analysis = PointCloudAnalysis()
    
    def load_point_cloud(self, filepath):
        """Load point cloud from file"""
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if len(pcd.points) == 0:
                raise ValueError(f"No points loaded from {filepath}")
            return pcd
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return None
    
    def save_point_cloud(self, pcd, filepath):
        """Save point cloud to file"""
        try:
            success = o3d.io.write_point_cloud(filepath, pcd)
            if success:
                print(f"Point cloud saved to {filepath}")
            else:
                print(f"Failed to save point cloud to {filepath}")
            return success
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            return False
    
    def create_sample_point_clouds(self, output_dir="data"):
        """Create sample point clouds for testing"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a cube point cloud
        cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        cube_pcd = cube.sample_points_uniformly(number_of_points=1000)
        cube_pcd.paint_uniform_color([1, 0, 0])  # Red
        self.save_point_cloud(cube_pcd, os.path.join(output_dir, "cube.ply"))
        
        # Create a sphere point cloud
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere_pcd = sphere.sample_points_uniformly(number_of_points=1000)
        sphere_pcd.paint_uniform_color([0, 1, 0])  # Green
        # Apply transformation
        transform = np.eye(4)
        transform[:3, 3] = [2, 0, 0]  # Translate
        sphere_pcd.transform(transform)
        self.save_point_cloud(sphere_pcd, os.path.join(output_dir, "sphere.ply"))
        
        # Create a plane with noise
        plane_points = []
        for x in np.linspace(-1, 1, 50):
            for y in np.linspace(-1, 1, 50):
                z = 0.1 * np.sin(2 * np.pi * x) + 0.05 * np.random.randn()  # Wavy plane with noise
                plane_points.append([x, y, z])
        
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(np.array(plane_points))
        plane_pcd.paint_uniform_color([0, 0, 1])  # Blue
        self.save_point_cloud(plane_pcd, os.path.join(output_dir, "noisy_plane.ply"))
        
        # Create a complex scene
        scene_pcd = cube_pcd + sphere_pcd + plane_pcd
        self.save_point_cloud(scene_pcd, os.path.join(output_dir, "scene.ply"))
        
        print(f"Sample point clouds created in {output_dir}/")
        return [
            os.path.join(output_dir, "cube.ply"),
            os.path.join(output_dir, "sphere.ply"),
            os.path.join(output_dir, "noisy_plane.ply"),
            os.path.join(output_dir, "scene.ply")
        ]
    
    def visualize_point_cloud(self, pcd, title="Point Cloud", point_size=1.0):
        """Visualize point cloud"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(pcd)
        
        # Set point size
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        
        vis.run()
        vis.destroy_window()
    
    def visualize_multiple_point_clouds(self, pcds, titles=None, colors=None):
        """Visualize multiple point clouds"""
        if colors is None:
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        
        # Color the point clouds
        for i, pcd in enumerate(pcds):
            if i < len(colors):
                pcd.paint_uniform_color(colors[i])
        
        # Visualize
        o3d.visualization.draw_geometries(pcds, window_name="Multiple Point Clouds")
    
    def process_registration_demo(self, source_path, target_path):
        """Demonstrate point cloud registration"""
        print("Loading point clouds...")
        source = self.load_point_cloud(source_path)
        target = self.load_point_cloud(target_path)
        
        if source is None or target is None:
            return
        
        print(f"Source: {len(source.points)} points")
        print(f"Target: {len(target.points)} points")
        
        # Apply random transformation to source
        transform = np.eye(4)
        transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.3])
        transform[:3, 3] = [0.5, 0.3, 0.1]
        source.transform(transform)
        
        print("\nTesting registration algorithms:")
        
        # Feature-based registration
        print("1. Feature-based registration...")
        start_time = time.time()
        transform_fgr, fitness_fgr, rmse_fgr = self.registration.feature_based_registration(
            source, target)
        time_fgr = time.time() - start_time
        print(f"   Fitness: {fitness_fgr:.4f}, RMSE: {rmse_fgr:.4f}, Time: {time_fgr:.2f}s")
        
        # Apply transformation
        source_registered = copy.deepcopy(source)
        source_registered.transform(transform_fgr)
        
        # ICP refinement
        print("2. ICP point-to-point refinement...")
        start_time = time.time()
        transform_icp, fitness_icp, rmse_icp = self.registration.icp_point_to_point(
            source_registered, target)
        time_icp = time.time() - start_time
        print(f"   Fitness: {fitness_icp:.4f}, RMSE: {rmse_icp:.4f}, Time: {time_icp:.2f}s")
        
        # Final result
        source_final = copy.deepcopy(source_registered)
        source_final.transform(transform_icp)
        
        # Visualize results
        source.paint_uniform_color([1, 0, 0])  # Red - original
        target.paint_uniform_color([0, 1, 0])  # Green - target
        source_final.paint_uniform_color([0, 0, 1])  # Blue - registered
        
        self.visualize_multiple_point_clouds([source, target, source_final], 
                                           ["Original", "Target", "Registered"])
    
    def process_filtering_demo(self, input_path):
        """Demonstrate point cloud filtering"""
        print("Loading point cloud...")
        pcd = self.load_point_cloud(input_path)
        
        if pcd is None:
            return
        
        print(f"Original: {len(pcd.points)} points")
        
        # Statistical outlier removal
        print("Applying statistical outlier removal...")
        pcd_stat, _ = self.filtering.statistical_outlier_removal(pcd)
        print(f"After statistical filtering: {len(pcd_stat.points)} points")
        
        # Voxel downsampling
        print("Applying voxel downsampling...")
        pcd_voxel = self.filtering.voxel_down_sample(pcd_stat, voxel_size=0.02)
        print(f"After voxel downsampling: {len(pcd_voxel.points)} points")
        
        # Bilateral filtering
        print("Applying bilateral filtering...")
        pcd_bilateral = self.filtering.bilateral_filter(pcd_voxel)
        print(f"After bilateral filtering: {len(pcd_bilateral.points)} points")
        
        # Visualize results
        pcd.paint_uniform_color([1, 0, 0])  # Red - original
        pcd_bilateral.paint_uniform_color([0, 1, 0])  # Green - filtered
        
        self.visualize_multiple_point_clouds([pcd, pcd_bilateral], 
                                           ["Original", "Filtered"])
    
    def process_segmentation_demo(self, input_path):
        """Demonstrate point cloud segmentation"""
        print("Loading point cloud...")
        pcd = self.load_point_cloud(input_path)
        
        if pcd is None:
            return
        
        print(f"Input: {len(pcd.points)} points")
        
        # Plane segmentation
        print("Plane segmentation using RANSAC...")
        plane_model, plane_pcd, remaining_pcd, _ = self.segmentation.plane_segmentation_ransac(pcd)
        print(f"Plane equation: {plane_model}")
        print(f"Plane points: {len(plane_pcd.points)}")
        print(f"Remaining points: {len(remaining_pcd.points)}")
        
        # Euclidean clustering on remaining points
        print("Euclidean clustering...")
        clusters, labels = self.segmentation.euclidean_clustering(remaining_pcd)
        print(f"Found {len(clusters)} clusters")
        
        # Visualize segmentation results
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        
        # Color plane
        plane_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
        
        # Color clusters
        vis_pcds = [plane_pcd]
        for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
            cluster.paint_uniform_color(colors[i % len(colors)])
            vis_pcds.append(cluster)
        
        self.visualize_multiple_point_clouds(vis_pcds)
    
    def process_reconstruction_demo(self, input_path):
        """Demonstrate surface reconstruction"""
        print("Loading point cloud...")
        pcd = self.load_point_cloud(input_path)
        
        if pcd is None:
            return
        
        print(f"Input: {len(pcd.points)} points")
        
        # Estimate normals
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        print("Surface reconstruction methods:")
        
        # Poisson reconstruction
        print("1. Poisson reconstruction...")
        mesh_poisson, densities = self.reconstruction.poisson_reconstruction(pcd)
        print(f"   Generated mesh with {len(mesh_poisson.vertices)} vertices, {len(mesh_poisson.triangles)} triangles")
        
        # Alpha shape
        print("2. Alpha shape reconstruction...")
        mesh_alpha = self.reconstruction.alpha_shape_reconstruction(pcd, alpha=0.1)
        print(f"   Generated mesh with {len(mesh_alpha.vertices)} vertices, {len(mesh_alpha.triangles)} triangles")
        
        # Ball pivoting
        print("3. Ball pivoting reconstruction...")
        mesh_ball = self.reconstruction.ball_pivoting_reconstruction(pcd)
        print(f"   Generated mesh with {len(mesh_ball.vertices)} vertices, {len(mesh_ball.triangles)} triangles")
        
        # Visualize original and reconstructed
        pcd.paint_uniform_color([1, 0, 0])  # Red points
        mesh_poisson.paint_uniform_color([0, 1, 0])  # Green mesh
        
        o3d.visualization.draw_geometries([pcd, mesh_poisson], window_name="Point Cloud + Poisson Mesh")


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Processing')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['register', 'filter', 'segment', 'reconstruct', 'analyze', 'demo', 'create_samples'],
                       help='Processing mode')
    parser.add_argument('--input', type=str, help='Input point cloud file')
    parser.add_argument('--source', type=str, help='Source point cloud for registration')
    parser.add_argument('--target', type=str, help='Target point cloud for registration')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='Voxel size for downsampling')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = PointCloudProcessor()
    
    if args.mode == 'create_samples':
        print("Creating sample point clouds...")
        sample_files = processor.create_sample_point_clouds(args.output_dir)
        print("Sample point clouds created:")
        for file in sample_files:
            print(f"  - {file}")
    
    elif args.mode == 'demo':
        print("Running comprehensive point cloud processing demo...")
        
        # Create sample data if input not provided
        if not args.input:
            print("Creating sample data...")
            sample_files = processor.create_sample_point_clouds("demo_data")
            scene_file = "demo_data/scene.ply"
            cube_file = "demo_data/cube.ply"
            sphere_file = "demo_data/sphere.ply"
        else:
            scene_file = args.input
            cube_file = args.input
            sphere_file = args.input
        
        # Run demos
        print("\n" + "="*50)
        print("REGISTRATION DEMO")
        print("="*50)
        processor.process_registration_demo(cube_file, sphere_file)
        
        print("\n" + "="*50)
        print("FILTERING DEMO")
        print("="*50)
        processor.process_filtering_demo(scene_file)
        
        print("\n" + "="*50)
        print("SEGMENTATION DEMO")
        print("="*50)
        processor.process_segmentation_demo(scene_file)
        
        print("\n" + "="*50)
        print("RECONSTRUCTION DEMO")
        print("="*50)
        processor.process_reconstruction_demo(cube_file)
    
    elif args.mode == 'register':
        if not args.source or not args.target:
            print("Error: --source and --target required for registration")
            return
        
        processor.process_registration_demo(args.source, args.target)
    
    elif args.mode == 'filter':
        if not args.input:
            print("Error: --input required for filtering")
            return
        
        processor.process_filtering_demo(args.input)
    
    elif args.mode == 'segment':
        if not args.input:
            print("Error: --input required for segmentation")
            return
        
        processor.process_segmentation_demo(args.input)
    
    elif args.mode == 'reconstruct':
        if not args.input:
            print("Error: --input required for reconstruction")
            return
        
        processor.process_reconstruction_demo(args.input)
    
    elif args.mode == 'analyze':
        if not args.input:
            print("Error: --input required for analysis")
            return
        
        print("Loading point cloud...")
        pcd = processor.load_point_cloud(args.input)
        
        if pcd is not None:
            # Quality analysis
            stats = processor.analysis.analyze_point_cloud_quality(pcd)
            print("\nPoint Cloud Quality Analysis:")
            print(f"Number of points: {stats['num_points']}")
            print(f"Centroid: {stats['centroid']}")
            print(f"Standard deviation: {stats['std_dev']}")
            print(f"Average nearest neighbor distance: {stats['avg_nearest_neighbor_distance']:.6f}")
            print(f"Uniformity: {stats['uniformity']:.4f}")
            
            # Geometric features
            features = processor.analysis.compute_geometric_features(pcd)
            print(f"\nGeometric Features:")
            print(f"Average curvature: {np.mean(features['curvature']):.6f}")
            print(f"Average density: {np.mean(features['density']):.6f}")
            print(f"Average planarity: {np.mean(features['planarity']):.6f}")
            print(f"Average roughness: {np.mean(features['roughness']):.6f}")
    
    print("\nPoint Cloud Processing completed!")
    print("\nKey Features Implemented:")
    print("- Multiple registration algorithms (ICP, feature-based, global)")
    print("- Robust filtering and outlier removal")
    print("- Plane and primitive detection using RANSAC")
    print("- Clustering and segmentation algorithms")
    print("- Surface reconstruction methods")
    print("- Geometric feature extraction")
    print("- Comprehensive visualization tools")
    print("- Quality assessment metrics")


if __name__ == '__main__':
    main()
