#!/usr/bin/env python3
"""
Merge Splatt3R Gaussian NPZ files, deduplicate overlapping centroids and
(optionally) export a PLY that is compatible with the original
`utils.export.save_as_ply()` from the 3D‑Gaussian‑Splatting code‑base.

Usage
-----
python merge_gaussians.py outputs/pair_*/gaussians.npz \
       --out  outputs/merged/merged.npz \
       --ply  outputs/merged/merged.ply \
       --radius 0.003

Requirements: `pip install torch numpy scipy plyfile`.
"""

import argparse, glob, os, sys, time, textwrap
from typing import Dict, List

import numpy as np
import torch
import einops
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.extend(["src"])  # allow local imports if needed

# ---------------------------------------------------------------- helpers

def load_npz(path: str) -> Dict[str, torch.Tensor]:
    """Load an .npz file into a dict of torch tensors."""
    return {k: torch.from_numpy(v) for k, v in np.load(path).items()}


def save_npz(path: str, d: Dict[str, torch.Tensor]):
    """Save a dict of torch tensors to a compressed .npz."""
    np.savez_compressed(path, **{k: v.cpu().numpy() for k, v in d.items()})


def flat(t: torch.Tensor) -> torch.Tensor:
    if t.ndim >= 5:                                  # (B,H,W,3,L)  or  (B,H,W,3,3)
        return t.reshape(-1, *t.shape[-2:])          # -> (N, 3,L)  or  (N,3,3)
    return t.reshape(-1, t.shape[-1])

# ---------------------------------------------------------- deduplication

def dedup(xyz: torch.Tensor, radius: float) -> torch.Tensor:
    """Return a boolean mask that keeps only one Gaussian within *radius*."""
    xyz_np = xyz.contiguous().cpu().numpy()

    # Make sure KD‑tree sees an (N,3) array
    if xyz_np.ndim > 2:
        xyz_np = xyz_np.reshape(-1, xyz_np.shape[-1])
    elif xyz_np.ndim == 1:
        xyz_np = xyz_np.reshape(1, -1)

    tree = cKDTree(xyz_np)
    duplicates = tree.query_pairs(radius)
    mask = np.ones(len(xyz_np), dtype=bool)
    for i, j in duplicates:
        mask[j] = False
    return torch.from_numpy(mask)


# ----------------------------------------------------------- PLY export


def _cov_to_rot_scale(cov: torch.Tensor):
    """Decompose 3×3 covariance matrices into rotation quaternions + scale."""
    # U S Vᵀ with cov = U diag(S) Vᵀ
    U, S, Vt = torch.linalg.svd(cov)
    R = torch.bmm(U, Vt)
    scale = torch.sqrt(S)  # σ along principal axes
    quat = Rotation.from_matrix(R.cpu()).as_quat()  # (N, 4)  xyz‑w
    return torch.from_numpy(quat).to(cov), scale


def _identity_rot_scale(n: int):
    """Return identity quaternion (0,0,0,1) and unit scale (1,1,1) for *n* rows."""
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    scale = np.ones((n, 3), dtype=np.float32)
    return quat, scale



def save_as_ply(pred, save_path):
    """
    Save a single network prediction (3-D Gaussians) as a point cloud in PLY.

    Parameters
    ----------
    pred : dict
        Must contain the keys
            'means'        : (B, H, W, 3)
            'covariances'  : (B, H, W, 3, 3)
            'sh'           : (B, H, W, 3, … )   # only SH-band-0 (RGB) is used
            'opacities'    : (B, H, W, 1)
        (B is the batch dimension; the function assumes B = 1.)
    save_path : str
        Destination *.ply* filename.
    """

    # ------------- helpers --------------------------------------------------
    def construct_attributes() -> list[str]:
        """Column names in the same order as the attribute vector we build."""
        attrs = ["x", "y", "z",
                 "nx", "ny", "nz",          # normals (here left at 0)
                 "f_dc_0", "f_dc_1", "f_dc_2",
                 "opacity",
                 "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3"]
        return attrs

    def covariance_to_quaternion_and_scale(covariance):
        """
        covariance : (N, 3, 3) tensor
        Returns
        -------
        quaternion : (N, 4)  float32
        scale      : (N, 3)  float32   # sqrt(eigvals)
        """
        # SVD → rotation & eigenvalues
        U, S, V = torch.linalg.svd(covariance)
        scale = torch.sqrt(S).cpu().numpy().astype(np.float32)          # (N,3)

        rotation_matrix = torch.bmm(U, V.transpose(-2, -1)).cpu().numpy()
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()    # (N,4)

        return quaternion.astype(np.float32), scale
    # -----------------------------------------------------------------------

    # Flatten spatial grid to (N,·) while stripping the batch dim (= 0).
    means = einops.rearrange(pred["means"][0],        "h w xyz -> (h w) xyz").cpu().numpy()
    covs  = einops.rearrange(pred["covariances"][0],  "h w i j -> (h w) i j")
    sh0   = einops.rearrange(pred["sh"][0][..., 0],   "h w xyz -> (h w) xyz").cpu().numpy()
    alphas = einops.rearrange(pred["opacities"][0],   "h w c   -> (h w) c").cpu().numpy()

    # Quaternion + log-scales from covariance
    quats, scales = covariance_to_quaternion_and_scale(covs)

    # Normals are not predicted here → set to zero
    zeros = np.zeros_like(means, dtype=np.float32)

    # Assemble attribute matrix (N,17) in the agreed order
    attributes = np.concatenate(
        [means,          # x y z
         zeros,          # nx ny nz  (all 0)
         sh0,            # f_dc_0…2
         alphas,         # opacity
         np.log(scales), # scale_0…2  (log for PixelSplat convention)
         quats],         # rot_0…3
        axis=-1).astype(np.float32)

    # PLY header expects a dtype with field names matching the attribute list
    dtype_descr = [(name, "f4") for name in construct_attributes()]
    vertex_array = np.empty(attributes.shape[0], dtype=dtype_descr)
    vertex_array[:] = list(map(tuple, attributes))

    ply = PlyData([PlyElement.describe(vertex_array, "vertex")], text=False)
    ply.write(save_path)

# -------------------------------------------------------------- merging

def merge(files: List[str], radius: float):
    clouds = [load_npz(f) for f in files]

    # 1) flatten, then concat every tensor ----------------------------------
    merged = {k: torch.cat([flat(c[k]) for c in clouds], dim=0) for k in clouds[0]}

    # 1a) ensure xyz/means are (N,3) so all downstream steps agree ----------
    merged["means"] = merged["means"].reshape(-1, merged["means"].shape[-1])

    # 2) deduplicate centroids ---------------------------------------------
    mask = dedup(merged["means"], radius)
    base_len = mask.numel()

    # 3) apply mask to per‑Gaussian tensors --------------------------------
    for k, v in merged.items():
        if v.shape[0] == base_len:
            merged[k] = v[mask]
    
    for k, v in merged.items():
        if v.ndim >= 2 and v.shape[0] == mask.sum():
            merged[k] = v.unsqueeze(0).unsqueeze(0)

    def count_gauss(t: torch.Tensor) -> int:
        return t.reshape(-1, t.shape[-1]).shape[0]

    n_in = sum(count_gauss(c["means"]) for c in clouds)
    n_out = int(mask.sum())
    return merged, n_in, n_out


# -------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    ap.add_argument("npzs", nargs="+", help="Input NPZ globs, e.g. outputs/pair_*/gaussians.npz")
    ap.add_argument("--out", required=True, help="Output merged .npz path")
    ap.add_argument("--ply", help="Optional PLY export path")
    ap.add_argument("--radius", type=float, default=0.02, help="Deduplication radius in world units (default 2 cm)")
    args = ap.parse_args()

    files = sorted(sum([glob.glob(p) for p in args.npzs], []))
    if not files:
        sys.exit("No npz matched")

    print(f"\nMerging {len(files)} clouds  (radius={args.radius*100:.1f} cm)")
    t0 = time.time()
    merged, n_in, n_out = merge(files, args.radius)

    print(f"  input   : {n_in:,}")
    print(f"  kept    : {n_out:,}")
    pruned = n_in - n_out
    pct = (pruned / n_in * 100) if n_in else 0.0
    print(f"  pruned  : {pruned:,}  ({pct:.1f} %)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_npz(args.out, merged)
    print("  →", args.out)

    if args.ply:
        os.makedirs(os.path.dirname(args.ply), exist_ok=True)
        save_as_ply(merged, args.ply)
        print("  →", args.ply)

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
