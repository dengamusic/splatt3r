
from typing import List, Tuple
import numpy as np
import torch

def rotmat_to_angle_deg(R: np.ndarray) -> float:
    """Return the angle (degrees) of a rotation matrix via trace -> acos."""
    tr = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def umeyama_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Rigid+scale alignment (Sim(3)) from Umeyama (1991). Returns (s, R, t)."""
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    A_c = A - mu_A
    B_c = B - mu_B
    C = B_c.T @ A_c / A.shape[0]
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_A = (A_c ** 2).sum() / A.shape[0]
    s = np.trace(np.diag(D) @ S) / var_A
    t = mu_B - s * (R @ mu_A)
    return float(s), R, t

def pose_inv(M: torch.Tensor) -> torch.Tensor:
    """Invert 4x4 pose matrix (SE(3))."""
    R = M[:3, :3]
    t = M[:3, 3]
    Minv = torch.eye(4, device=M.device, dtype=M.dtype)
    Minv[:3, :3] = R.T
    Minv[:3, 3] = -R.T @ t
    return Minv

def get_dust3r_order(scene) -> List[int]:
    """Recover the image order from the pose graph edges (dust3r scene)."""
    order = []
    for i, j in scene.edges:
        if i not in order:
            order.append(i)
        if j not in order:
            order.append(j)
    return order
