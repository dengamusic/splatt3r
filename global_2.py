#!/usr/bin/env python3
import argparse, os, sys, itertools, glob, json, yaml
from collections import defaultdict

import numpy as np, torch
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from plyfile      import PlyData, PlyElement

# local libs
sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])
from dust3r.utils.image  import load_images
from dust3r.utils.device import collate_with_cat
import main                                                  # Splatt3r model
import re
# ─────────────────────────────── CLI ─────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+",
                help="five RGB files in the same order as the extrinsics")
ap.add_argument("--extrinsics", required=True,
                help="txt with 4×4 row-wise matrices (one per image)")
ap.add_argument("--w2c", action="store_true",
                help="set if the extrinsics file is world→camera")
ap.add_argument("--outdir", required=True)

ap.add_argument("--radius", type=float, default=0.003,
                help="dedup radius (metres, replica units)")
ap.add_argument("--model-dir")
ap.add_argument("--gaussian-subsample", type=int,
                help="randomly keep at most this many per-view splats")
ap.add_argument("--save-ply", action="store_true")
args = ap.parse_args()

# ───────────────────────────── setup ─────────────────────────────────────────
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir, exist_ok=True)

ckpt  = hf_hub_download("brandonsmart/splatt3r_v1.0",
                        "epoch=19-step=1200.ckpt",
                        local_dir=args.model_dir, local_dir_use_symlinks=False)
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, dev).eval().to(dev)

# ──────────────────── load intrinsics & helper scaler ────────────────────────
W0, H0 = 1200, 680
#fx0, fy0, cx0, cy0 = map(float,
 #   (cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
fx0 = fy0 = 600.0
cx0 = 600.0
cy0 = 340.0

IM_SIZE = 512                               # Splatt3r resize target (square)
sx, sy = IM_SIZE / W0, IM_SIZE / H0         # pure resize, no crop
fx, fy = fx0 * sx, fy0 * sy
cx, cy = cx0 * sx, cy0 * sy

print(f"[intrinsics] resized {W0}×{H0} → {IM_SIZE}×{IM_SIZE}; "
      f"fx={fx:.1f}, fy={fy:.1f}")

all_poses = np.loadtxt(args.extrinsics).reshape(-1, 4, 4)  # assume full trajectory

# Extract image IDs like 1400 from paths like rgb_1400.png
pattern = re.compile(r'rgb_(\d+)\.png')
image_ids = [int(pattern.search(p).group(1)) for p in args.images]

c2w = np.stack([all_poses[i] for i in image_ids], axis=0)
if not args.w2c:
    c2w = np.linalg.inv(c2w)  # default case: camera→world

assert len(c2w) == len(args.images), "Pose/image count mismatch after filtering."
# ──────────────────── load & resize RGBs for Splatt3r ────────────────────────
imgs = load_images(args.images, size=IM_SIZE, verbose=False)
for idx, im in enumerate(imgs):
    im["img"], im["original_img"] = im["img"].to(dev), im["original_img"].to(dev)
    im["true_shape"], im["idx"] = torch.as_tensor(im["true_shape"]), idx

# ──────────────────── util helpers ───────────────────────────────────────────
def concat(list_of_dicts):
    acc = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            acc[k].append(v.detach())
    return {k: torch.cat(vs, 0) for k, vs in acc.items()}

def save_as_ply(cloud, path):
    xyz   = cloud["means"].reshape(-1,3).cpu().numpy()
    rot   = cloud["rotations"].reshape(-1,4).cpu().numpy()
    scale = np.exp(cloud["log_scales"].reshape(-1,3).cpu().numpy())
    sh0   = cloud["sh"][..., :3].reshape(-1,3).cpu().numpy()
    opa   = torch.sigmoid(cloud["logit_opacities"]).reshape(-1,1).cpu().numpy()
    zeros = np.zeros_like(xyz, dtype=np.float32)
    attrs = np.concatenate([xyz, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)
    names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
             "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
    dt    = np.dtype([(n,"f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dt); verts[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(verts,"vertex")], text=False).write(path)

@torch.no_grad()
def run_pair(i,j):
    v_i, v_j = imgs[i], imgs[j]
    p_i, p_j = model(v_i, v_j)

    for p in (p_i, p_j):
        if "log_scales" not in p:       p["log_scales"]       = p["scales"].log()
        if "logit_opacities" not in p:
            a = p["opacities"].clamp(1e-4, 1-1e-4)
            p["logit_opacities"] = (a / (1-a)).log()

    d_i = {"pts3d": p_i["pts3d"], "conf": p_i["conf"]}
    d_j = {"pts3d": p_j["pts3d_in_other_view"], "conf": p_j["conf"]}
    for k in ("means","rotations","sh","log_scales","logit_opacities"):
        if k in p_j:
            key = "means_in_other_view" if k=="means" else k
            d_j[key] = p_j[k]
    return v_i, v_j, d_i, d_j, p_i, p_j

# ──────────────────── run all Splatt3r pairs ─────────────────────────
pairs = list(itertools.combinations(range(len(imgs)), 2))
pairs += [(j, i) for i, j in pairs]
v1, v2, p1, p2, gauss = [], [], [], [], defaultdict(list)

for i, j in pairs:
    a, b, c, d, full_i, full_j = run_pair(i, j)
    gauss[i].append(full_i)

# ──────────────────── build per-view clouds in replica world ─────────
def cloud_from(pred):
    m = pred["means"] if "means" in pred else pred["means_in_other_view"]
    c = {"means": m.reshape(-1,3)}
    for k in ("rotations","sh","log_scales","logit_opacities"):
        if k in pred: c[k] = pred[k].reshape(c["means"].shape[0], -1)
    return c

clouds = []
for idx in range(len(imgs)):
    cl = concat([cloud_from(p) for p in gauss[idx]])

    if args.gaussian_subsample and cl["means"].shape[0] > args.gaussian_subsample:
        sel = torch.randperm(cl["means"].shape[0], device=dev)[:args.gaussian_subsample]
        for k in list(cl): cl[k] = cl[k][sel]

    # transform means with replica camera-to-world
    M = torch.tensor(c2w[idx], dtype=torch.float32, device=dev)
    R, t = M[:3,:3], M[:3,3]
    cl["means"] = (cl["means"] @ R.T) + t

    npz = os.path.join(args.outdir, f"cloud_{idx}.npz")
    np.savez_compressed(npz, **{k: v.cpu().numpy() for k,v in cl.items()})
    clouds.append(cl)
    print(f"✓ saved {npz}  ({cl['means'].shape[0]} pts)")

# ──────────────────── simple merge (no ICP) ──────────────────────────
merged = defaultdict(list)
for cl in clouds:
    for k,v in cl.items(): merged[k].append(v)
merged = {k: torch.cat(vs,0) for k,vs in merged.items()}

# deduplicate identical/very close splats
xyz  = merged["means"].cpu().numpy()
dup  = cKDTree(xyz).query_pairs(args.radius)
keep = np.ones(len(xyz), dtype=bool)
for i, j in dup:           # keep the first, discard the duplicate
    keep[j] = False
keep = torch.from_numpy(keep).to(dev)

for k in list(merged):
    merged[k] = merged[k][keep]

out = os.path.join(args.outdir, "merged.npz")
print("saving keys:", {k: v.shape for k,v in merged.items()})
np.savez_compressed(out, **{k: v.cpu().numpy() for k,v in merged.items()})
print("✓ wrote merged.npz  (unique pts:", keep.sum().item(), ")")

# ──────────────────── dump poses for fine-tune ───────────────────────
poses_txt = os.path.join(args.outdir, "poses_replica.txt")
with open(poses_txt, "w") as f:
    for M in c2w:
        np.savetxt(f, M, fmt="%.8f"); f.write("\n")
print("✓ wrote", poses_txt, "(camera→world)")


if args.save_ply:
    save_as_ply(merged, out.replace(".npz", ".ply"))
    print("wrote PLY")
print("Done")
# ──────────────────── ICP refinement (commented out) ─────────────────
"""
import open3d as o3d
print("## ICP block disabled – remove triple-quotes to enable.")
#  … full ICP loop from the previous script here …
"""
