
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

def concat_tensors(ds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    acc = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            acc[k].append(v.detach() if torch.is_tensor(v) else v)
    return {k: torch.cat(vs, 0) for k, vs in acc.items()}

def save_as_ply(cloud: Dict[str, torch.Tensor], path: str) -> None:
    xyz = cloud["means"].reshape(-1, 3).cpu().numpy()
    rot = cloud["rotations"].reshape(-1, 4).cpu().numpy()
    scale = np.exp(cloud["log_scales"].reshape(-1, 3).cpu().numpy())
    sh0 = cloud["sh"][..., :3].reshape(-1, 3).cpu().numpy()
    opa = torch.sigmoid(cloud["logit_opacities"]).reshape(-1, 1).cpu().numpy()

    zeros = np.zeros_like(xyz, dtype=np.float32)
    attrs = np.concatenate([xyz, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    ]
    dt = np.dtype([(n, "f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dt)
    verts[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(path)

def dedup_xyz(means: torch.Tensor, r: float) -> torch.Tensor:
    dev = means.device
    xyz = means.detach().cpu().numpy()
    dup = cKDTree(xyz).query_pairs(r)
    keep = np.ones(len(xyz), dtype=bool)
    for i, j in dup:
        keep[j] = False
    return torch.from_numpy(keep).to(dev)

def cloud_from_pred(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    m = pred["means"].reshape(-1, 3) if "means" in pred else pred["means_in_other_view"].reshape(-1, 3)
    c = {"means": m}
    for k in ("rotations", "sh", "log_scales", "logit_opacities"):
        if k in pred:
            v = pred[k]
            if k == "sh":
                v = v.reshape(-1, 3)
            elif k == "log_scales":
                v = v.reshape(-1, 3)
            elif k == "rotations":
                v = v.reshape(-1, 4)
            elif k == "logit_opacities":
                v = v.reshape(-1, 1)
            c[k] = v
    return c

def transform_cloud_inplace(cloud: Dict[str, torch.Tensor], R: torch.Tensor, t: torch.Tensor) -> None:
    cloud["means"] = (cloud["means"] @ R.to(cloud["means"].device).T) + t.to(cloud["means"].device)
