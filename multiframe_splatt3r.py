#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  multiframe_splatt3r.py — demonstrator for 3 images  ==  3 pairs
# ---------------------------------------------------------------------------
#  Usage:
#  python multiframe_splatt3r.py img0.jpg img1.jpg img2.jpg \
#         --outdir outputs/three --save-ply
#
#  Needs: torch numpy einops scipy plyfile huggingface_hub
# ---------------------------------------------------------------------------

import argparse, itertools, os, sys
from collections import defaultdict
import torch, numpy as np, einops
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree

# ---- local paths to Splatt3r ------------------------------------------------
sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])
from dust3r.utils.image import load_images
import main                                   # MASt3R entry
from pytorch3d.transforms import (
    quaternion_to_matrix, matrix_to_quaternion
)

# ---------------------------------------------------------------------------


def build_rotation(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6-D rotation (x-form in last dim) to a 3 × 3 matrix.

    Args
    ----
    rot6d : (..., 6) tensor

    Returns
    -------
    (..., 3, 3) tensor  —  proper rotation, det = +1
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)  

def transform_cloud_(cloud, R, t):
    """In-place rigid transform of a Splatt3r Gaussian dict."""
    key = "means_in_other_view" if "means_in_other_view" in cloud else "means"
    cloud[key] = cloud[key] @ R.T + t

    R_old = quaternion_to_matrix(cloud["rotations"])
    R_new = torch.einsum("ij,...jk->...ik", R, R_old)
    cloud["rotations"] = matrix_to_quaternion(R_new)


def dedup_xyz(means, radius):
    xyz = means.cpu().numpy()
    tree = cKDTree(xyz)
    dup = tree.query_pairs(radius)
    keep = np.ones(len(xyz), dtype=bool)
    for i, j in dup:
        keep[j] = False
    return torch.from_numpy(keep).to(means.device)


def concat(dicts):
    out = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out[k].append(v.detach())
    return {k: torch.cat(vs, dim=0) for k, vs in out.items()}


def save_as_ply(cloud, path):
    from plyfile import PlyData, PlyElement
    attrs = ["x","y","z","nx","ny","nz",
             "f_dc_0","f_dc_1","f_dc_2","opacity",
             "scale_0","scale_1","scale_2",
             "rot_0","rot_1","rot_2","rot_3"]
    N = len(cloud["means"])
    zeros = torch.zeros_like(cloud["means"])
    rgb   = cloud["sh"][:, :3] if "sh" in cloud else zeros
    opac  = torch.sigmoid(cloud["logit_opacities"]) if "logit_opacities" in cloud \
            else torch.ones((N,1), device=cloud["means"].device)
    scales= torch.exp(cloud["log_scales"]) if "log_scales" in cloud \
            else torch.ones((N,3), device=cloud["means"].device)
    data  = torch.cat([cloud["means"], zeros, rgb, opac,
                       torch.log(scales), cloud["rotations"]], dim=1).cpu().numpy()
    dt    = np.dtype([(n,"f4") for n in attrs])
    PlyData([PlyElement.describe(np.array(list(map(tuple, data)), dtype=dt),"vertex")],
            text=False).write(path)

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs=3, help="exactly three images")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--radius", type=float, default=0.02, help="dedup radius")
    ap.add_argument('--model-dir', default=None)
    ap.add_argument("--save-ply", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Splatt3r model ----------------------------------------------------
    ckpt = hf_hub_download(
        "brandonsmart/splatt3r_v1.0", "epoch=19-step=1200.ckpt",
        local_dir=args.model_dir, local_dir_use_symlinks=False, resume_download=True
    )
    model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device).eval().to(device)


    # Pose dictionary: frame-index → (R_to0, t_to0)
    poses_to_0 = {0: (torch.eye(3, device=device),
                      torch.zeros(3, device=device))}

    merged_clouds = []

    # --- helper -------------------------------------------------------------
    def run_pair(i, j):
        ims = load_images([args.images[i], args.images[j]], size=512, verbose=False)
        for im in ims:
            im["img"]           = im["img"].to(device)
            im["original_img"]  = im["original_img"].to(device)
            im["true_shape"]    = torch.as_tensor(im["true_shape"])
        return model(ims[0], ims[1])   # returns (pred_i, pred_j)

    # -----------------------------------------------------------------------
    print("\nSTEP 1: pairs (0,1) and (0,2) — bootstrap others")
    for j in (1, 2):
        p0, pj = run_pair(0, j)

        # get R_ j→0 , t_j→0 from Splatt3r
        R_j0 = build_rotation(pj["cam_unnorm_rots"])[0]
        t_j0 = pj["cam_trans"][0]

        poses_to_0[j] = (R_j0, t_j0)

        print(f"\nPair (0,{j}):")
        print("  R_j0 =\n", R_j0.cpu().numpy())
        print("  det(R) =", torch.det(R_j0).item())
        print("  t_j0 =", t_j0.cpu().tolist())

        transform_cloud_(p0, *poses_to_0[0])    # identity
        transform_cloud_(pj, R_j0, t_j0)
        merged_clouds += [p0, pj]

    print("\nSTEP 2: pair (1,2) — uses chained poses")
    p1, p2 = run_pair(1, 2)

    # Splatt3r pose R_2→1, t_2→1
    R21 = build_rotation(p2["cam_unnorm_rots"])[0]
    t21 = p2["cam_trans"][0]
    print("\nPair (1,2):")
    print("  R_21 =\n", R21.cpu().numpy())
    print("  det(R) =", torch.det(R21).item())
    print("  t_21 =", t21.cpu().tolist())

    # frame-1 → frame-0 we already know
    R10, t10 = poses_to_0[1]

    # chain: R20 = R10 · R21 ,  t20 = R10·t21 + t10
    R20 = R10 @ R21
    t20 = R10 @ t21 + t10
    poses_to_0[2] = (R20, t20)    # (was identical, just checking)

    print("\nChained pose 2→0 (via 1):")
    print("  R_20 =\n", R20.cpu().numpy())
    print("  det(R) =", torch.det(R20).item())
    print("  t_20 =", t20.cpu().tolist())

    transform_cloud_(p1, R10, t10)
    transform_cloud_(p2, R20, t20)
    merged_clouds += [p1, p2]

    # -----------------------------------------------------------------------
    print("\nSTEP 3: merge & deduplicate")
    merged = concat(merged_clouds)
    mask = dedup_xyz(merged["means"], args.radius)
    for k, v in merged.items():
        if v.shape[0] == len(mask):
            merged[k] = v[mask]

    npz_path = os.path.join(args.outdir, "merged.npz")
    np.savez_compressed(npz_path, **{k: v.cpu().numpy() for k, v in merged.items()})
    print(f"\n✓ wrote {npz_path}  (kept {mask.sum().item()} / {len(mask)})")

    if args.save_ply:
        ply_path = os.path.join(args.outdir, "merged.ply")
        save_as_ply(merged, ply_path)
        print("✓ wrote", ply_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
