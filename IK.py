#!/usr/bin/env python3

import pinocchio as pin
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import os

# -----------------------
# Configuration
# -----------------------
# TODO: User should configure these paths and names according to their own robot setup
URDF_PATH = "/path/to/your/robot.urdf"  # Change this to your URDF file path

JOINT_NAMES = [
    "joint_1",  # Change these to your actual joint names
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7"
]

EE_LINK_NAME = "end_effector_link"  # Change this to your end-effector link name

# End-effector offset configuration (relative to EE_LINK_NAME)
EE_OFFSET_POS = np.array([0.0, 0.0, 0.0])  # Position offset [x, y, z]
EE_OFFSET_ROT = np.eye(3)  # Rotation offset (3x3 rotation matrix)


class PinocchioIK:
    """
    Inverse Kinematics solver using Pinocchio library.
    Supports both LOCAL and WORLD frame reference for error computation.
    """
    
    def __init__(self, urdf_path, joint_names, ee_link_name, ee_offset_pos, ee_offset_rot):
        """
        Initialize the IK solver.
        
        Args:
            urdf_path: Path to the robot URDF file
            joint_names: List of active joint names to control
            ee_link_name: Name of the end-effector link
            ee_offset_pos: Position offset of the end-effector tool
            ee_offset_rot: Rotation offset of the end-effector tool
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        
        # Build full model from URDF
        self.full_model = pin.buildModelFromUrdf(urdf_path)
        self.full_data = self.full_model.createData()
        self.ee_link_name = ee_link_name
        self.joint_names = joint_names
        
        # Identify joints to lock (not in joint_names)
        self.joints_to_lock_ids = []
        full_joint_names = list(self.full_model.names)
        
        for name in full_joint_names:
            if name != "universe" and name not in joint_names:
                idx = self.full_model.getJointId(name)
                self.joints_to_lock_ids.append(idx)
        
        # End-effector offset transformation
        self.T_site_offset = pin.SE3(ee_offset_rot, ee_offset_pos)
        
        # Initialize with neutral configuration
        self.update_robot_state(np.zeros(self.full_model.nq))

    def update_robot_state(self, full_q):
        """
        Update the reduced model based on current full configuration.
        Locks unused joints and creates reduced model for active joints.
        
        Args:
            full_q: Full configuration vector of the robot
        """
        if len(full_q) != self.full_model.nq:
            print(f"Warning: full_q size mismatch. Ignoring update.")
            return

        # Build reduced model with locked joints
        self.model = pin.buildReducedModel(self.full_model, self.joints_to_lock_ids, full_q)
        self.data = self.model.createData()
        
        if not self.model.existBodyName(self.ee_link_name):
            raise ValueError(f"Link {self.ee_link_name} not found in model.")
        
        self.ee_frame_id = self.model.getBodyId(self.ee_link_name)
        self.q_min = self.model.lowerPositionLimit
        self.q_max = self.model.upperPositionLimit

    def forward_kinematics(self, q):
        """
        Compute forward kinematics.
        
        Args:
            q: Joint configuration vector
            
        Returns:
            T_site_global: 4x4 homogeneous transformation matrix
            p_site: Position vector [x, y, z]
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_link_global = self.data.oMf[self.ee_frame_id]
        T_site_global = T_link_global * self.T_site_offset
        return T_site_global.homogeneous, T_site_global.translation

    def solve(self, target_pos, target_rot_mat, q0=None, q_ref=None,
              lambda_damp=1e-4, tol=1e-5, max_iter=100,
              frame_ref="LOCAL", pos_weight=1.0, rot_weight=1.0,
              step_scale=0.5, adaptive_damping=True, damping_ratio=5.0):
        """
        Solve inverse kinematics using Damped Least Squares (DLS) method.
        
        Args:
            target_pos: Target position [x, y, z]
            target_rot_mat: Target rotation (3x3 matrix)
            q0: Initial guess for joint configuration
            q_ref: Reference posture for null-space optimization
            lambda_damp: Damping factor for DLS
            tol: Convergence tolerance
            max_iter: Maximum iterations
            frame_ref: "LOCAL" or "WORLD" frame for error computation
            pos_weight: Weight for position error
            rot_weight: Weight for rotation error
            step_scale: Step size scaling factor
            adaptive_damping: Enable adaptive damping adjustment
            damping_ratio: Ratio for adaptive damping
            
        Returns:
            q: Solution joint configuration
            info: Dictionary with success status, iterations, and residual
        """
        # Initialize configuration
        if q0 is None:
            q = pin.neutral(self.model)
        else:
            q = q0.copy()

        if q_ref is None:
            q_ref = q.copy()

        # Target pose as SE3
        oMdes = pin.SE3(target_rot_mat, target_pos)
        info = {'success': False, 'iter': max_iter, 'residual': 0.0}
        
        # Weight vector for error components
        row_weights = np.concatenate((
            np.full(3, pos_weight, dtype=np.float64),
            np.full(3, rot_weight, dtype=np.float64)
        ))

        # Iterative optimization loop
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Current end-effector pose
            oM_current = self.data.oMf[self.ee_frame_id] * self.T_site_offset
            
            # ========================================================
            # Branch 1: LOCAL frame reference
            # ========================================================
            if frame_ref == "LOCAL":
                # Compute error in current frame
                # iMd = Current^-1 * Desired
                iMd = oM_current.actInv(oMdes)
                err_vector = pin.log6(iMd).vector
                
                # Compute Jacobian in local frame
                J_link = pin.computeFrameJacobian(
                    self.model, self.data, q, self.ee_frame_id, 
                    pin.ReferenceFrame.LOCAL
                )
                # Apply offset correction
                J = self.T_site_offset.actInv(pin.SE3.Identity()).toActionMatrix() @ J_link

            # ========================================================
            # Branch 2: WORLD frame reference
            # ========================================================
            elif frame_ref == "WORLD":
                # Compute error in local frame then rotate to world
                iMd = oM_current.actInv(oMdes)
                err_local = pin.log6(iMd).vector
                # Transform error to world frame
                err_vector = oM_current.action @ err_local
                
                # Compute Jacobian in world frame
                J_link_local = pin.computeFrameJacobian(
                    self.model, self.data, q, self.ee_frame_id,
                    pin.ReferenceFrame.LOCAL
                )
                J_local = self.T_site_offset.actInv(
                    pin.SE3.Identity()
                ).toActionMatrix() @ J_link_local
                
                # Project to world frame
                J = oM_current.action @ J_local
                
            else:
                raise ValueError("frame_ref must be 'LOCAL' or 'WORLD'")

            # --- DLS (Damped Least Squares) solver ---
            err_weighted = row_weights * err_vector
            residual = np.linalg.norm(err_weighted)
            
            # Check convergence
            if residual < tol:
                info['success'] = True
                info['iter'] = i
                info['residual'] = residual
                return q, info

            # Compute weighted Jacobian
            J_weighted = row_weights[:, None] * J
            JJt = J_weighted @ J_weighted.T

            # Adaptive damping
            if adaptive_damping:
                lambda_eff = lambda_damp * (1.0 + damping_ratio * min(residual, 1.0))
            else:
                lambda_eff = lambda_damp

            damp_mat = lambda_eff**2 * np.eye(6)

            # Task velocity (weighted error feedforward)
            v = err_weighted
            
            # Solve for joint velocity
            dq_task = J_weighted.T @ np.linalg.solve(JJt + damp_mat, v)

            # Null-space posture control
            dq_posture = 1.0 * (q_ref - q)
            twist_posture = J_weighted @ dq_posture
            dq_correction = J_weighted.T @ np.linalg.solve(JJt + damp_mat, twist_posture)
            dq_null = dq_posture - dq_correction
            
            # Combine task and null-space motions
            dq = dq_task + dq_null
            
            # Update configuration with integration and clamping
            q = pin.integrate(self.model, q, dq * step_scale)
            q = np.clip(q, self.q_min, self.q_max)

        # Store final residual if not converged
        info['residual'] = residual
        return q, info
    
    def get_full_dynamics(self, full_q, full_dq):
        """
        Compute full dynamics matrices (M, C, g).
        
        Args:
            full_q: Full joint positions
            full_dq: Full joint velocities
            
        Returns:
            M_full: Mass matrix
            C_full: Coriolis and centrifugal terms
            g_full: Gravity terms
        """
        # Mass matrix
        pin.crba(self.full_model, self.full_data, full_q)
        M_full = self.full_data.M
        M_full = np.triu(M_full) + np.triu(M_full, 1).T
        
        # Nonlinear effects (Coriolis + gravity)
        nle_full = pin.rnea(
            self.full_model, self.full_data, full_q, full_dq, 
            np.zeros(self.full_model.nv)
        )
        
        # Gravity terms
        g_full = pin.rnea(
            self.full_model, self.full_data, full_q, 
            np.zeros(self.full_model.nv), np.zeros(self.full_model.nv)
        )
        
        return M_full, nle_full - g_full, g_full


# Initialize IK solver
try:
    ik_solver = PinocchioIK(URDF_PATH, JOINT_NAMES, EE_LINK_NAME, EE_OFFSET_POS, EE_OFFSET_ROT)
    print("IK solver initialized successfully")
except Exception as e:
    print(f"IK solver initialization failed: {e}")
    import traceback
    traceback.print_exc()
    ik_solver = None


def forward_kinematics(q):
    """
    Wrapper for forward kinematics.
    
    Args:
        q: Joint configuration
        
    Returns:
        T_end: Homogeneous transformation matrix
        empty list (for compatibility)
        p_site: End-effector position
    """
    T_end, p_site = ik_solver.forward_kinematics(q)
    return T_end, [], p_site


def ik_solve(target_pos, target_rot_mat, q0=None, q_prev=None,
             lambda_damp=1e-4, k_null=0.2, step_scale=0.5,
             tol=1e-4, max_iter=20, verbose=False, frame_ref="LOCAL",
             pos_weight=1.0, rot_weight=1.0, adaptive_damping=True,
             damping_ratio=5.0):
    """
    Legacy interface wrapper for IK solve.
    
    Args:
        target_pos: Target position [x, y, z]
        target_rot_mat: Target rotation matrix (3x3)
        q0: Initial configuration guess
        q_prev: Previous configuration (for null-space, not used currently)
        lambda_damp: Damping factor
        k_null: Null-space gain (legacy, not used)
        step_scale: Step size scaling
        tol: Convergence tolerance
        max_iter: Maximum iterations
        verbose: Print debug info (legacy)
        frame_ref: "LOCAL" or "WORLD"
        pos_weight: Position error weight
        rot_weight: Rotation error weight
        adaptive_damping: Enable adaptive damping
        damping_ratio: Adaptive damping ratio
        
    Returns:
        q: Solution configuration
        info: Solver information dictionary
    """
    return ik_solver.solve(
        target_pos, target_rot_mat, q0,
        lambda_damp=lambda_damp, tol=tol, max_iter=max_iter,
        frame_ref=frame_ref, pos_weight=pos_weight,
        rot_weight=rot_weight, step_scale=step_scale,
        adaptive_damping=adaptive_damping,
        damping_ratio=damping_ratio
    )


# -----------------------
# Demo / Main
# -----------------------
if __name__ == "__main__":
    if ik_solver is None:
        print("Error: Solver not initialized.")
        exit(1)
        
    # Test configuration
    N = len(JOINT_NAMES)
    q0 = np.zeros(N)
    
    # 1. Compute initial pose
    T0, _, p0 = forward_kinematics(q0)
    R0 = T0[:3, :3]
    print(f"Initial Position: {p0}")
    
    # 2. Set target
    target_pos = p0 + np.array([0, 0, 0.1])  # Move 10cm up in Z
    # Small rotation (0 degrees here, modify as needed)
    target_rot = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix() @ R0
    
    print("-" * 30)
    print("Starting Pinocchio IK...")
    
    start_time = time.time()
    q_sol, info = ik_solve(
        target_pos, target_rot, q0=q0,
        lambda_damp=1e-5, tol=1e-3, max_iter=20,
        frame_ref="WORLD", rot_weight=2.0,
        step_scale=0.5, adaptive_damping=True,
        damping_ratio=6.0, verbose=True
    )
    end_time = time.time()
    
    print(f'Time cost: {end_time - start_time:.6f}s')
    print("IK Info:", info)
    
    # Verify result
    T_final, _, p_final = forward_kinematics(q_sol)
    
    pos_err = np.linalg.norm(p_final - target_pos)
    
    # Compute rotation error
    R_err_mat = target_rot @ T_final[:3, :3].T
    rot_err_vec = R.from_matrix(R_err_mat).as_rotvec()
    rot_err = np.linalg.norm(rot_err_vec)
    
    print(f"Solution q: {q_sol}")
    print(f"Position Error: {pos_err:.6e}")
    print(f"Rotation Error: {rot_err:.6e}")
    print(f'Current Position: {p_final}')