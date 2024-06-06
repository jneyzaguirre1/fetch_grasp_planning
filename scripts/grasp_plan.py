#!/usr/bin/env python

# This code is based on the robot manipulation class from Russ Tedrake at MITs 

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d



class GraspPlanner:

    def __init__(self, rng):
        """
        :param rng: a np.random.default_rng()
        """
        self.rng = rng
    
    def generate_antipodal_grasp_candidate(self, pcd_w):
        """
        Computes an antipodal grasp candidate
        :param pcd_w: world frame open3d point cloud of the obj to grasp (consider N points)
        :return: cost: float
                 X_wg: np.arrays of size (4, 4) Homogeneus transformation representing the gripper pose wrt the world frame
        """
        pts_w = np.asarray(pcd_w.points).T
        ns_w = np.asarray(pcd_w.normals).T

        idx = self.rng.integers(0, pts_w.shape[0] - 1)
        
        # Sample point and normal.
        p_w = pts_w[idx]
        n_w = ns_w[idx]

        assert np.isclose(
            np.linalg.norm(n_w), 1.0
        ), f"Normal has magnitude: {np.linalg.norm(n_w)}"

        Gx = n_w  # gripper x axis aligns with normal
        # make orthonormal y axis, aligned with world down
        y = np.array([0.0, 0.0, -1.0])
        if np.abs(np.dot(y, Gx)) < 1e-6:
            # normal was pointing straight down.  reject this sample.
            return np.inf, None
        
        Gy = y - np.dot(y, Gx) * Gx     # Gram-schmidth process
        Gz = np.cross(Gx, Gy)
        R_WG = np.vstack((Gx, Gy, Gz)).T
        p_GS_G = [0.054 - 0.01, 0.10625, 0] # TODO: Set parameters according to gripper

        # Try orientations from the center out
        min_roll = -np.pi / 3.0
        max_roll = np.pi / 3.0
        alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
        for theta in min_roll + (max_roll - min_roll) * alpha:
            # Rotate the object in the hand by a random rotation (around the normal).
            R_WG2 = R_WG.multiply(self.rodrigues(theta))

            # Use G for gripper frame.
            p_SG_W = -R_WG2.multiply(p_GS_G)
            p_WG = p_WS + p_SG_W

            X_G = RigidTransform(R_WG2, p_WG)
            plant.SetFreeBodyPose(plant_context, wsg, X_G)
            cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True)
            X_G = plant.GetFreeBodyPose(plant_context, wsg)
            if np.isfinite(cost):
                return cost, X_G

    
    def grasp_cost(self, pcd_w, X_wg, heat_map, align_center=True, verbose=False):
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
        HM = np.diag(heat_map)

        # Transform points and normals into gripper frame
        X_gw = X_wg.inverse()
        pts_g = self.transform_pts(X_gw, pts_w)
        ns_g = X_gw[:3, :3] @ ns_w

        # Crop to a region inside of the finger box.
        crop_min = [-0.05, 0.1, -0.00625]           # TODO: define!!
        crop_max = [0.05, 0.1125, 0.00625]          # TODO: define!!
        idx = self.get_bbx_idx(pts_g, crop_min, crop_max)
        pts_g_crop = pts_g[:, idx]
        ns_g_crop = ns_g[:, idx]

        if align_center and np.sum(idx) > 0:
            # align the gripper to be at the center between the min max of the pc_bbx
            pts_g_x = pts_g_crop[:, 0]              # TODO: check gripper frame to see if it matches
            pts_g_x_center = (pts_g_x.min() + pts_g_x.max()) / 2.0
            X_wg = self.set_translation(X_wg, self.transform_pts(pts_g_x_center, X_wg))
            X_gw = X_wg.inverse()

        # TODO: Check collisions --> return inf in case of collision

        # Cost function
        weight = 20.0
        normal_cost = -ns_g_crop[0, :].T @ HM @ ns_g_crop[0, :]
        gripper_cost = weight * X_gw[2, 1]          # TODO: check gripper frame to see if it matches
        cost = normal_cost + gripper_cost

        if verbose:
            print(f"cost: {cost}")
            print(f"normal terms: {normal_cost}")
            print(f"gripper terms: {gripper_cost}")

        return cost
    
    def compute_centroid(self, pcd):
        """
        Computes the centroid of the point cloud
        :param pcd: np.array of size (3, N) containing the points
        :return: np.array of size (3,)
        """
        return np.mean(pcd[:3, :], axis=1)
  
    def get_bbx_idx(pc, min_limits, max_limits):
        """
        Computes an array of bools indicating if the point is inside the bbx
        :param pc: np.array of size (3, N) containing the points
        :param min_limits: np.arrays of size (3,)
        :param max_limits: np.arrays of size (3,)
        :return: np.array of size (N,)
        """
        idx_min_bool = np.all(np.where(pc[:3, :] > min_limits, True, False), axis=1)
        idx_max_bool = np.all(np.where(pc[:3, :] < max_limits, True, False), axis=1)
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
        pts_cloud = pts.copy()
        R = X[:3, :3]
        t = X[:3, 3]
        pts_transformed = (R @ pts_cloud + t)

        return pts_transformed
    
    def rodrigues(self, w, theta):
        """
        Creates a rotational matrices based on a rotation axises
        """
        w_matrix = self.get_skew_symmetric(w)
        R = np.eye(3) + np.sin(theta) * w_matrix + (1 - np.cos(theta)) * w_matrix @ w_matrix
        return R
    
    def get_skew_symmetric(self, w):
        w_matrix = np.zeros((3, 3))
        w_matrix[:, 0, 1] = -w[2, :]
        w_matrix[:, 0, 2] =  w[1, :]
        w_matrix[:, 1, 0] =  w[2, :]
        w_matrix[:, 1, 2] = -w[0, :]
        w_matrix[:, 2, 0] = -w[1, :]
        w_matrix[:, 2, 1] =  w[0, :]
        return w_matrix

