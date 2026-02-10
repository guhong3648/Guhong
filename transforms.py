import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from pytorch3d.transforms import euler_angles_to_matrix
    from pytorch3d.transforms import matrix_to_euler_angles
    from pytorch3d.transforms import matrix_to_quaternion
    from pytorch3d.transforms import quaternion_to_matrix
except ImportError:
    euler_angles_to_matrix = None
    matrix_to_euler_angles = None

# Plot func for 6d pos vec
def plot_emt(emt_data):
    plt.figure(figsize=(14, 4))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(emt_data[:, i])
    plt.show()
    
'''
- ZYX Cenvention = [yaw, pitch, roll]
- After Initialization = [lateral, depth, elevational, yaw, pitch, roll]
- Form =  [param (6D Vector), tforms (4x4 Matrix)]
- y_vec:  [B, S, 6] parameter representation
- a_vec:  [B, S, 3] parameter representation of the Euler angle
- Y_mat:  [B, S, 4, 4] matrix representation
- A_mat:  [B, S, 3, 3] matrix representation of the Euler angle
- y_rel:  relative pos vec
- y_abs:  absolute pos vec
- Y_rel:  relative pos mat
- Y_abs:  absolute pos mat

- y_quat: [B, S, 7] [translation, Quaternion]
- q_vec:  [B, S, 4] Quaternion
'''

class Transform_bundle():
    def __init__(self, deg=True, convention='ZYX', first_angle=False, p3d=True):
        self.deg = deg
        self.convention = convention
        self.first_angle = first_angle
        self.p3d = p3d
        self.alpha = torch.pi/180 if self.deg else 1
    
    def get_tensor(self, y):
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y)
        
        if (y.shape[-1]==6) & (len(y.shape)!=3): y = y.unsqueeze(0)
        if (y.shape[-1]!=6) & (len(y.shape)!=4): y = y.unsqueeze(0)
        
        return y.to(torch.float64).clone()
    
    def get_convert(self, y):
        beta = self.alpha if self.deg else 1/self.alpha
        if self.first_angle: 
            y[..., :3] = y[..., :3]*beta
        else:
            y[..., 3:] = y[..., 3:]*beta
        
        return y
    
    def get_split(self, y):
        if self.first_angle:
            return y[..., 3:], y[..., :3]
        else:
            return y[..., :3], y[..., 3:]
        
    def get_marge(self, s_vec, a_vec):
        a_vec = a_vec/self.alpha
        if self.first_angle:
            return torch.concat([a_vec, s_vec], axis=-1)
        else:
            return torch.concat([s_vec, a_vec], axis=-1)
    
    def get_angle_mat(self, a_vec):
        if self.p3d:
            A_mat = euler_angles_to_matrix(a_vec, self.convention)
        else:
            A_mat = torch.from_numpy(np.stack([torch.from_numpy(R.from_euler(self.convention, a_vec[i].numpy()).as_matrix()) for i in range(len(a_vec))]))
        
        return A_mat.to(torch.float32)
    
    def get_angle_vec(self, A_mat):
        if self.p3d:
            a_vec = matrix_to_euler_angles(A_mat, self.convention)
        else:
            a_vec = torch.from_numpy(np.stack([torch.from_numpy(R.from_matrix(A_mat[i].numpy()).as_euler(self.convention)) for i in range(len(A_mat))]))
        
        return a_vec.to(torch.float32)
    
    def get_from_type(self, pose):
        if pose.shape[-1]==6:
            return 'vec'
        elif pose.shape[-1]==4:
            return 'mat'
        else: raise ValueError(f'Invalid Shape: pose={pose.shape}')
    
    def transform_a2m(self, y_vec):
        y_vec = self.get_tensor(y_vec)
        y_vec = self.get_convert(y_vec)
        s_vec, a_vec = self.get_split(y_vec)
        A_mat = self.get_angle_mat(a_vec)
        
        return A_mat
        
    def transform_v2m(self, y_vec):
        y_vec = self.get_tensor(y_vec)
        y_vec = self.get_convert(y_vec)
        s_vec, a_vec = self.get_split(y_vec)
        A_mat = self.get_angle_mat(a_vec)
        
        Y_mat = torch.eye(4, dtype=A_mat.dtype, device=A_mat.device).repeat(A_mat.shape[0], A_mat.shape[1], 1, 1)
        Y_mat[..., :3, :3] = A_mat
        Y_mat[..., :3,  3] = s_vec
        
        return Y_mat
        
    def transform_m2v(self, Y_mat):
        A_mat = Y_mat[..., :3, :3]
        s_vec = Y_mat[..., :3,  3]
        a_vec = self.get_angle_vec(A_mat)
        y_vec = self.get_marge(s_vec, a_vec)
        
        return y_vec
    
    def transform_acum(self, pose_rel, source=None, target=None):
        pose_rel = self.get_tensor(pose_rel)
        if source is None: source = self.get_from_type(pose_rel)
        if target is None: target = self.get_from_type(pose_rel)
        
        if source=='vec': pose_rel = self.transform_v2m(pose_rel)
        pose_abs = torch.eye(4, dtype=pose_rel.dtype, device=pose_rel.device).repeat(pose_rel.shape[0], pose_rel.shape[1]+1, 1, 1)
        for i in range(pose_rel.shape[1]):
            pose_abs[:, i+1] = pose_abs[:, i]@pose_rel[:, i]
        
        if target=='vec': 
            pose_abs = self.transform_m2v(pose_abs)
            s_vec, a_vec = self.get_split(pose_abs)
            a_vec = a_vec*self.alpha
            pose_abs = self.get_marge(s_vec, a_vec)
        
        return pose_abs
    
    def transform_diff(self, pose_abs, source=None, target=None):
        pose_abs = self.get_tensor(pose_abs)
        if source is None: source = self.get_from_type(pose_abs)
        if target is None: target = self.get_from_type(pose_abs)
        
        if source=='vec': pose_abs = self.transform_v2m(pose_abs)
        pose_rel = torch.eye(4, dtype=pose_abs.dtype, device=pose_abs.device).repeat(pose_abs.shape[0], pose_abs.shape[1]-1, 1, 1)
        for i in range(pose_rel.shape[1]):
            pose_rel[:, i] = torch.linalg.inv(pose_abs[:, i])@pose_abs[:, i+1]
        
        if target=='vec': 
            pose_rel = self.transform_m2v(pose_rel)
            s_vec, a_vec = self.get_split(pose_rel)
            a_vec = a_vec*self.alpha
            pose_rel = self.get_marge(s_vec, a_vec)
        
        return pose_rel
    
    def transform_m2q(self, Y_mat):
        s_vec = Y_mat[..., :3,  3]
        A_mat = Y_mat[..., :3, :3]

        if self.p3d:
            q_vec = matrix_to_quaternion(A_mat.to(torch.float32))  # (w,x,y,z)

        else:
            A_np = A_mat.detach().cpu().numpy()
            A_flat = A_np.reshape(-1, 3, 3)
            q_xyzw = R.from_matrix(A_flat).as_quat()  # (x,y,z,w)
            q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
            q_vec = torch.from_numpy(q_wxyz).to(A_mat.device).to(torch.float32)
            q_vec = q_vec.reshape(*A_mat.shape[:2], 4)
        
        y_quat = torch.concat([s_vec, q_vec], axis=-1)
        
        return y_quat

    def transform_q2m(self, y_quat):
        s_vec = y_quat[..., :3]
        q_vec = y_quat[..., 3:]
        if self.p3d:
            A_mat = quaternion_to_matrix(q_vec.to(torch.float32))
            
        else:
            q_np = q_vec.detach().cpu().numpy()
            q_flat = q_np.reshape(-1, 4)
            q_xyzw = np.concatenate([q_flat[:, 1:], q_flat[:, 0:1]], axis=1)
            A_np = R.from_quat(q_xyzw).as_matrix()
            A_mat = torch.from_numpy(A_np).to(q_vec.device).to(torch.float32)
            A_mat = A_mat.reshape(*q_vec.shape[:-1], 3, 3)

        Y_mat = torch.eye(4, device=q_vec.device, dtype=A_mat.dtype)
        Y_mat = Y_mat.expand(*A_mat.shape[:-2], 4, 4).clone()
        Y_mat[..., :3, :3] = A_mat
        Y_mat[..., :3,  3] = s_vec
        
        return Y_mat
