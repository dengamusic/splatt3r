#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  Splatt3R – two-view inference that anchors the cloud in Replica WORLD space
#  (y-up, right-handed, forward = −Z) using traj_w_cgl.txt poses.
# -----------------------------------------------------------------------------
import argparse, os, re, sys, torch, numpy as np
from huggingface_hub import hf_hub_download

# ── repo imports ────────────────────────────────────────────────────────────
sys.path.extend(['src/mast3r_src', 'src/mast3r_src/dust3r', 'src/pixelsplat_src'])
from dust3r.utils.image import load_images
import main
import utils.export as export               # expects 'means' / 'rotations' …

# quaternion helpers ---------------------------------------------------------
from pytorch3d.transforms import (
    quaternion_to_matrix, matrix_to_quaternion
)

# ── helpers ────────────────────────────────────────────────────────────────
def frame_index(img_path: str) -> int:
    """rgb/000123.jpg -> 123"""
    m = re.search(r'(\d+)\D*$', os.path.splitext(os.path.basename(img_path))[0])
    if not m:
        raise ValueError(f"Cannot parse frame index from {img_path}")
    return int(m.group(1))

def load_c2w(traj_file: str, idx: int) -> torch.Tensor:
    """camera→world (4×4) from Replica traj_w_cgl.txt"""
    mats = np.loadtxt(traj_file, dtype=np.float32).reshape(-1, 4, 4)
    if idx >= len(mats):
        raise IndexError(f"Index {idx} out of bounds for {traj_file}")
    return torch.from_numpy(np.linalg.inv(mats[idx]))  # (4,4)

def lift_in_place(cloud: dict, R: torch.Tensor, t: torch.Tensor):
    """
    Lift MASt3R Gaussian dict from cam-1 frame ➜ world frame.

    Expects keys:
      cloud['means']      (..., 3)
      cloud['rotations']  (..., 4)   (x,y,z,w) unit quaternion
    """
    means_key = 'means'
    if 'means_in_other_view' in cloud:
        means_key = 'means_in_other_view'
    # ── 1. centroids  --------------------------------------------------------
    # (...,3) @ (3,3)ᵀ  + t  -> (...,3)
    cloud[means_key] = cloud[means_key] @ R.T + t

    # ── 2. orientations -----------------------------------------------------
    R_cam = quaternion_to_matrix(cloud['rotations'])       # (...,3,3)
    # left-multiply by world rotation; einsum broadcasts cleanly
    R_world = torch.einsum('ij,...jk->...ik', R, R_cam)    # (...,3,3)
    cloud['rotations'] = matrix_to_quaternion(R_world)
    
def combine_predictions(pred1: dict, pred2: dict, *, to_cpu=True) -> dict:
    default_alt = {
        "means": "means_in_other_view",
        "pts3d": "pts3d_in_other_view",
    }
    merged = {}
    for k, v1 in pred1.items():
        if k in default_alt and default_alt[k] in pred2:
            v2 = pred2[default_alt[k]]
        else:
            v2 = pred2[k]  # assume same key exists

        combined = torch.cat((v1, v2), dim=0)
        if to_cpu:
            combined = combined.detach().cpu()
        merged[k] = combined

    return merged

# ── CLI ────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('img1'); p.add_argument('img2')
p.add_argument('--traj-file', required=True, help='Replica traj_w_cgl.txt')
p.add_argument('--outdir', default='results')
p.add_argument('--model-dir', default=None)
p.add_argument('--save-npz', action='store_true')
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── checkpoint ────────────────────────────────────────────────────────────
ckpt = hf_hub_download(
    "brandonsmart/splatt3r_v1.0", "epoch=19-step=1200.ckpt",
    local_dir=args.model_dir, local_dir_use_symlinks=False, resume_download=True
)
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device).eval().to(device)

# ── load images ───────────────────────────────────────────────────────────
imgs = load_images([args.img1, args.img2], size=512, verbose=False)
for im in imgs:
    im['img']          = im['img'].to(device)
    im['original_img'] = im['original_img'].to(device)
    im['true_shape']   = torch.as_tensor(im['true_shape'])

pred1, pred2 = model(imgs[0], imgs[1])      # keys must be 'means', 'rotations', …

# ── world-frame lift (using camera-1 pose) ─────────────────────────────────
idx1 = frame_index(args.img1)
c2w1 = load_c2w(args.traj_file, idx1).to(device)
R1, t1 = c2w1[:3, :3], c2w1[:3, 3]

lift_in_place(pred1, R1, t1)
lift_in_place(pred2, R1, t1)   # still expressed in cam-1 frame

# ── write outputs ─────────────────────────────────────────────────────────
os.makedirs(args.outdir, exist_ok=True)
ply_path = os.path.join(args.outdir, 'gaussians.ply')
export.save_as_ply(pred1, pred2, ply_path)
print(f"✓ wrote {ply_path}")


npz_path = os.path.join(args.outdir, 'gaussians.npz')
combined = combine_predictions(pred1, pred2)
np.savez_compressed(npz_path, **{k: v.numpy() for k, v in combined.items()})
print(f"✓ wrote {npz_path}")
