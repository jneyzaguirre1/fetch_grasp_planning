#!/usr/bin/env python

# This code is based on the robot manipulation class from Russ Tedrake at MITs 

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


class GraspPlanner:

    def __init__(self, rng, gripper_box, gripper_offset, verbose=False):
        """
        :param rng: a np.random.default_rng()
        :param gripper_box: np.array of size (3,) that represents the volume between the gripper's fingers.
        :param gripper_offset: np.array of size (3,) that represents the fixed translation between the ee 
                               and the center between the fingers.
        :param verbose: bool prints results.
        """
        self.rng = rng
        self.verbose = verbose
        self.crop_min = gripper_offset - gripper_box / 2.
        self.crop_max = gripper_offset + gripper_box / 2.
        self.finger_offset = gripper_offset + np.array([gripper_box[0]/2., 0, 0])
        self.g_offset = gripper_offset

        print("FINGERS: ", self.crop_min, self.crop_max, self.finger_offset)

    def sample_grasps(self, pcd_w, n=1):
        """
        Samples different grasp candidates and returns the best "n" candidates
        :param pcd_w: world frame open3d point cloud of the obj to grasp (consider N points)
        :param n: int representing the number of samples to return
        :return: list of "n" gripper poses wrt the world frame
        """
        assert n > 0, f"n must be a positive integer, got {n}"

        costs = []
        X_wgs = []
        for i in range(100):
            cost, X_wg = self.generate_antipodal_grasp_candidate(pcd_w)
            if np.isfinite(cost):
                costs.append(cost)
                X_wgs.append(X_wg)

        indices = np.asarray(costs).argsort()[:n]
        if self.verbose:
            for rank, index in enumerate(indices):
                print(f"{rank}th best")
                print(f"Cost: {costs[index]}")
                print(X_wgs[index])
        return [X_wgs[index] for index in indices]
    
    def generate_antipodal_grasp_candidate(self, pcd_w):
        """
        Computes an antipodal grasp candidate
        :param pcd_w: world frame open3d point cloud of the obj to grasp (consider N points)
        :return: cost: float
                 X_wg: np.arrays of size (4, 4) Homogeneus transformation representing the gripper pose wrt the world frame
        """
        pts_w = np.asarray(pcd_w.points).T
        ns_w = np.asarray(pcd_w.normals).T

        idx = self.rng.integers(0, pts_w.shape[1] - 1)
        
        # Sample point and normal.
        p_w = pts_w[:, idx]
        n_w = ns_w[:, idx]

        if self.verbose: print("\nNormal (world frame): ", n_w)

        # check normal norm == 1.0
        assert np.isclose(
            np.linalg.norm(n_w), 1.0
        ), f"Normal has magnitude: {np.linalg.norm(n_w)}"

        gripper_x = n_w  # gripper x axis aligns with normal
        # make orthonormal y axis, aligned with world down
        y = np.array([0.0, 0.0, -1.0])
        if np.abs(np.dot(y, gripper_x)) > 0.9:
            # normal was pointing almost straight down.  reject this sample.
            if self.verbose: print("Rejected normal")
            return np.inf, None
    
        gripper_y = y - np.dot(y, gripper_x) * gripper_x     # Gram-schmidth process
        gripper_z = np.cross(gripper_x, gripper_y)
        R_wg = np.vstack((gripper_x, gripper_y, gripper_z)).T
        p_gs_g = np.array([0.054 - 0.01, 0.10625, 0.]) # TODO: Set parameters according to gripper
                                            # TODO: change to ray casting
        p_gs_g = np.array([0.06 - 0.01, 0.0, 0.0])
        p_gs_g = self.finger_offset - np.array([0.01, 0., 0.])

        # Try orientations from the center out
        min_roll = -np.pi / 3.0
        max_roll = np.pi / 3.0
        alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
        for theta in min_roll + (max_roll - min_roll) * alpha:
            # Rotate the object in the hand by a random rotation around the normal (X axis of the gripper).
            R_wg2 = R_wg @ self.rodrigues(np.array([1, 0, 0]), theta)

            # Move the gripper so that one of the fingers is touching the selected point.
            p_SG_W = -R_wg2 @ p_gs_g
            p_wg = p_w + p_SG_W

            X_wg = self.rigid_transform(R_wg2, p_wg)
            heat_map = np.ones(pts_w.shape[1])
            cost, X_wg = self.grasp_cost(pcd_w, X_wg, heat_map, align_center=True)
            if np.isfinite(cost):
                return cost, X_wg

    
    def grasp_cost(self, pcd_w, X_wg, heat_map, align_center=True):
        """
        Computes the grasp cost for an especific gripper pose, obj pointcloud, and heatmap
        :param pcd_w: world frame open3d point cloud of the obj to grasp (consider N points)
        :param X_wg: np.arrays of size (4, 4) Homogeneus transformation representing the gripper pose wrt the world frame
        :param heat_map: np.arrays of size (N,) Heat map representing the uncertaintly of the point belonging to the obj.
        :param align_center: bool whether to align the gripper or not to wrt the min max of the obj points
        :param verbose: bool
        :return: cost: float
        """
        pts_w = np.asarray(pcd_w.points).T
        ns_w = np.asarray(pcd_w.normals).T

        # Transform points and normals into gripper frame
        X_gw = self.rigid_transformation_inv(X_wg)
        pts_g = self.transform_pts(X_gw, pts_w)
        ns_g = X_gw[:3, :3] @ ns_w

        # Crop to a region inside of the finger box.
        idx = self.get_bbx_idx(pts_g, self.crop_min, self.crop_max)
        pts_g_crop = pts_g[:, idx]
        ns_g_crop = ns_g[:, idx]
        hm_cropped = heat_map[idx]
        HM = np.diag(hm_cropped)

        if align_center and np.sum(idx) > 0:
            # align the gripper to be at the center between the min max of the pc_bbx
            pts_g_x = pts_g_crop[0, :]              # TODO: check gripper frame to see if it matches
            pts_g_x_center = (pts_g_x.min() + pts_g_x.max()) / 2.0
            X_wg = self.set_translation(X_wg, self.transform_pts(X_wg, np.array([[pts_g_x_center, 0, 0]]).T)[:, 0])
            X_gw = self.rigid_transformation_inv(X_wg)

        # TODO: Check collisions --> return inf in case of collision

        # Cost function
        try:
            weight = 20.0
            normal_cost = -ns_g_crop[0, :].T @ HM @ ns_g_crop[0, :]
            gripper_cost = weight * X_gw[2, 1]          # TODO: check gripper frame to see if it matches
            cost = normal_cost + gripper_cost
        except:
            breakpoint()

        if self.verbose:
            print(f"cost: {cost}")
            print(f"normal terms: {normal_cost}")
            print(f"gripper terms: {gripper_cost}")

        return cost, X_wg
    
    def compute_centroid(self, pcd):
        """
        Computes the centroid of the point cloud
        :param pcd: np.array of size (3, N) containing the points
        :return: np.array of size (3,)
        """
        return np.mean(pcd[:3, :], axis=1)
  
    def get_bbx_idx(self, pc, min_limits, max_limits):
        """
        Computes an array of bools indicating if the point is inside the bbx
        :param pc: np.array of size (3, N) containing the points
        :param min_limits: np.arrays of size (3,)
        :param max_limits: np.arrays of size (3,)
        :return: np.array of size (N,)
        """
        idx_min_bool = np.all(np.where(pc[:3, :].T > min_limits, True, False), axis=1)
        idx_max_bool = np.all(np.where(pc[:3, :].T < max_limits, True, False), axis=1)
        idx_bool = np.logical_and(idx_min_bool, idx_max_bool)
        return idx_bool
    
    def set_translation(self, X, p):
        X[:3, 3] = p
        return X
    
    def transform_pts(self, X, pts):
        """
        Transform a point cloud applying a rotation and a translation
        :param X: np.array of size (4,4) representing an homogeneus transformation.
        :param pts: np.arrays of size (3, N)
        :return: np.array of size (3, N) resulting in applying the transformation (t,R) on the points.
        """
        pts_cloud = np.block([[pts.copy()], [np.ones(pts.shape[1])]])
        #R = X[:3, :3]
        #t = X[:3, 3]
        pts_transformed = X @ pts_cloud
        return pts_transformed[:3, :]
    
    def rodrigues(self, w, theta):
        """
        Creates a rotational matrix based on a rotation axis and theta around that axis
        :param w: np.array of size (3,) representing the axis of rotation
        :param theta: float representing the angle in rads to rotate around the axis
        :return: np.array of size (3, 3)
        """
        w_matrix = self.get_skew_symmetric(w)
        R = np.eye(3) + np.sin(theta) * w_matrix + (1 - np.cos(theta)) * w_matrix @ w_matrix
        return R
    
    def get_skew_symmetric(self, w):
        w_matrix = np.zeros((3, 3))
        w_matrix[0, 1] = -w[2]
        w_matrix[0, 2] =  w[1]
        w_matrix[1, 0] =  w[2]
        w_matrix[1, 2] = -w[0]
        w_matrix[2, 0] = -w[1]
        w_matrix[2, 1] =  w[0]
        return w_matrix
    
    def rigid_transform(self, R, t):
        """
        Combines the rotation R and translation t to form a rigid body trnasformation.
        :param R: np.array of size (3, 3) representing the Rotation
        :param t: np.array of size (3,) representing the translation
        :return: np.array of size (4, 4)
        """
        t_ = t.reshape(3, 1)
        return np.block([[R, t_], [np.zeros((1, 3)), 1.]])
    
    def rigid_transformation_inv(self, X):
        R = np.linalg.inv(X[:3, :3])
        t = (-R @ X[:3, 3]).reshape(3, 1)
        return np.block([[R, t], [np.zeros((1, 3)), 1.]])
    

if __name__ == "__main__":
    import copy
    from mesh_generator import MeshGenerator

    # Create a pcd
    mesh_gen = MeshGenerator()
    mesh_gen.low_limit = mesh_gen.high_limit = 0.05               # make the size of the object constant
    name, params, mesh = mesh_gen.create_mesh(0)                                     # 0: box
    #mesh = mesh.translate(-mesh.get_center(), relative=True)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=150000)
    pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Create grasp planner
    rng = np.random.default_rng(seed=0)
    g_box = np.array([0.12, 0.0125, 0.0125])
    g_offset = np.array([0., 0., 0.])
    grasp_planner = GraspPlanner(rng, g_box, g_offset, verbose=False)
    pts_w = np.asarray(pcd.points).T
    pcd.colors = o3d.utility.Vector3dVector(np.zeros(pts_w.T.shape, dtype=np.float64))
    print("pcd shape: ", pts_w.shape)

    # Plan the grasp
    X_wg = grasp_planner.sample_grasps(pcd, n=5)[0]

    # Compute grasp region of the pcd
    # Transform points and normals into gripper frame
    X_gw = grasp_planner.rigid_transformation_inv(X_wg)
    pts_g = grasp_planner.transform_pts(X_gw, pts_w)
    # Crop to a region inside of the finger box.
    idx = grasp_planner.get_bbx_idx(pts_g, grasp_planner.crop_min, grasp_planner.crop_max)
    pts_w_crop = pts_w[:, idx]

    # Show panned grasp in red
    selected = o3d.geometry.PointCloud()
    print("shape of selected: ", pts_w_crop.shape)
    colors = np.zeros(pts_w_crop.T.shape, dtype=np.float64)
    colors[:, 0] = 1.   # set to red
    selected.colors = o3d.utility.Vector3dVector(colors)
    selected.points = o3d.utility.Vector3dVector(pts_w_crop.T)

    # Add finger region
    finger_box = o3d.geometry.TriangleMesh.create_box(width=g_box[0], 
                                                      height=g_box[1], 
                                                      depth=g_box[2], 
                                                      create_uv_map=False, 
                                                      map_texture_to_each_face=False)
    finger_box.transform(X_wg)
    finger_box.translate(g_box/2., relative=True)

    # Add coordinate frames
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=np.array([0.0, 0.0, 0.0]))
    gripper_frame = copy.deepcopy(origin_frame).transform(X_wg)

    o3d.visualization.draw_geometries([pcd, selected, origin_frame, ]) # gripper_frame, finger_box