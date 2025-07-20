#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  multiframe_splatt3r.py — 3 images  ⇒  global alignment (DUSt3R cloud_opt)
# ---------------------------------------------------------------------------
#  Usage:
#  python multiframe_splatt3r.py img0.jpg img1.jpg img2.jpg \
#         --outdir outputs/three --save-ply
#
#  Needs: torch numpy einops scipy plyfile huggingface_hub
# ---------------------------------------------------------------------------

import argparse, os, sys, copy, itertools
from collections import defaultdict

import torch, numpy as np, einops
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
# ---- local paths to Splatt3r / DUSt3R code ----------------------------------
sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])

from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat, to_cpu
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import main  # Splatt3R's MASt3RGaussians LightningModule


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------ DEBUG
def debug_tree(obj, name="", indent=0):
    pad = " " * indent
    if name:
        print(f"{pad}{name}:")
    if torch.is_tensor(obj):
        print(f"{pad}  tensor  shape={tuple(obj.shape)}  dtype={obj.dtype}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            debug_tree(v, k, indent+2)
    elif isinstance(obj, list):
        print(f"{pad}  list len={len(obj)}")
        for i, v in enumerate(obj[:5]):                 # show first few
            debug_tree(v, f"[{i}]", indent+4)
        if len(obj) > 5:
            print(f"{pad}    … ({len(obj)-5} more)")
    else:
        print(f"{pad}  {type(obj).__name__}")
# ------------------------------------------------------------------------

def dedup_xyz(means, radius):
    """Return boolean mask marking a single survivor per cluster < radius."""
    xyz = means.cpu().numpy()
    tree = cKDTree(xyz)
    dup = tree.query_pairs(radius)
    keep = np.ones(len(xyz), dtype=bool)
    for i, j in dup:
        keep[j] = False
    return torch.from_numpy(keep).to(means.device)


def concat(dicts):
    """Cat list of *flat* cloud dicts along dim=0."""
    out = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out[k].append(v.detach())
    return {k: torch.cat(vs, dim=0) for k, vs in out.items()}


def save_as_ply(pred, save_path):
    means = pred["means"].reshape(-1, 3).cpu().numpy()
    rot   = pred["rotations"].reshape(-1, 4).cpu().numpy()
    scale = np.exp(pred["log_scales"].reshape(-1, 3).cpu().numpy())
    sh0   = pred["sh"][..., :3].reshape(-1, 3).cpu().numpy()
    opac  = torch.sigmoid(pred["logit_opacities"]).reshape(-1, 1).cpu().numpy()

    zeros = np.zeros_like(means, dtype=np.float32)

    attrs = np.concatenate(
        [means,                 # xyz
         zeros,                 # normals
         sh0,                   # f_dc_*
         opac,                  # opacity
         np.log(scale),         # log scales
         rot],                  # quaternion
        axis=-1).astype(np.float32)

    names = ["x","y","z",
             "nx","ny","nz",
             "f_dc_0","f_dc_1","f_dc_2",
             "opacity",
             "scale_0","scale_1","scale_2",
             "rot_0","rot_1","rot_2","rot_3"]

    dtype = np.dtype([(n, "f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dtype)
    verts[:] = list(map(tuple, attrs))

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(save_path)

# ---------------------------------------------------------------------------
#  DEBUG helper – paste near the top of the file, after imports
# ---------------------------------------------------------------------------
def dump_dict(d, title="", indent=0):
    pad = " " * indent
    if title:
        print(f"{pad}{title}")
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"{pad}  {k:20s}  tensor  shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, list):
            # lists can be images (N dicts) or per-pair tensors
            print(f"{pad}  {k:20s}  list    len={len(v)}")
            if len(v) and torch.is_tensor(v[0]):
                shapes = [tuple(x.shape) for x in v]
                print(f"{pad}      element shapes: {shapes[:3]}{' …' if len(shapes)>3 else ''}")
            elif len(v) and isinstance(v[0], dict):
                for i, itm in enumerate(v):
                    print(f"{pad}        [{i}]")
                    dump_dict(itm, indent=indent+10)
        else:
            print(f"{pad}  {k:20s}  {type(v).__name__}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs=3, help="exactly three images")
ap.add_argument("--outdir", required=True)
ap.add_argument("--radius", type=float, default=0.003, help="dedup radius in world units")
ap.add_argument('--model-dir', default=None)
ap.add_argument("--save-ply", action="store_true")
ap.add_argument("--no-sym", action="store_true",
                help="disable adding reverse-direction pairs (default: include them)")
ap.add_argument("--gaussian-subsample", type=int, default=None,
                help="keep at most N highest-confidence Gaussians per view before merging")
ap.add_argument("--align-iters", type=int, default=1000,
                help="global alignment optimization steps")
ap.add_argument("--align-lr", type=float, default=0.01,
                help="optimizer LR for global alignment")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir, exist_ok=True)


# ---------------------------------------------------------------------------
# Load Splatt3R model
# ---------------------------------------------------------------------------

ckpt = hf_hub_download(
    "brandonsmart/splatt3r_v1.0", "epoch=19-step=1200.ckpt",
    local_dir=args.model_dir, local_dir_use_symlinks=False, resume_download=True
)
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device).eval().to(device)


# ---------------------------------------------------------------------------
# Load all 3 images once
# ---------------------------------------------------------------------------

IM_SIZE = 512  # fixed; adjust if you trained differently
imgs = load_images(args.images, size=IM_SIZE, verbose=False)

# patch up device + required fields
for k, im in enumerate(imgs):
    im["img"]          = im["img"].to(device)
    im["original_img"] = im["original_img"].to(device)
    im["true_shape"]   = torch.as_tensor(im["true_shape"])
    im["idx"]          = k  # DUSt3R uses this


# ---------------------------------------------------------------------------
# Helper: wrap a model pair pass into DUSt3R-style records
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_pair(i, j):
    """Run MASt3R Splatt3R on (i,j) → DUSt3R-compatible view/pred dicts."""
    view_i = {k: v for k, v in imgs[i].items()}  # shallow copy is fine
    view_j = {k: v for k, v in imgs[j].items()}

    pred_i, pred_j = model(view_i, view_j)  # Splatt3R forward

    for p in (pred_i, pred_j):
        # log_scales  = log of σ
        if "log_scales" not in p and "scales" in p:
            p["log_scales"] = p["scales"].log()                        # same shape

        # logit_opacities = log( α / (1-α) );  clamp α to avoid inf / nan
        if "logit_opacities" not in p and "opacities" in p:
            eps = 1e-4
            alpha = p["opacities"].clamp(min=eps, max=1-eps)
            p["logit_opacities"] = (alpha / (1 - alpha)).log()   
    # --- adapt Splatt3R preds to DUSt3R optimizer expectations -------------
    # DUSt3R wants 'pts3d' per-view (camera frame). Splatt3R returns pts3d for i
    # and pts3d_in_other_view for j (already in i-frame). We'll just rename it;
    # the optimizer's per-edge adaptor will absorb the frame inconsistency.
    # If you want to "unwind" properly, insert a rigid-fit here.
    pred_i_for_dust3r = {
        "pts3d": pred_i["pts3d"],     # (B,H,W,3) camera-i
        "conf":  pred_i["conf"],      # (B,H,W)
    }

    pred_j_for_dust3r = {
        # what the optimizer explicitly expects:
        "pts3d_in_other_view": pred_j["pts3d_in_other_view"],  # (H,W,3) in i-frame
        "conf":               pred_j["conf"],
        # Optional duplicate so later code that looks for 'pts3d' still works:
        "pts3d":              pred_j["pts3d_in_other_view"],
    }

    # keep Gaussian extras if you want them later
    for extra_key in ("means","rotations","sh","log_scales","logit_opacities","covariances"):
        if extra_key in pred_j:
            if extra_key == "means":
                # these means are in i-frame already
                pred_j_for_dust3r["means_in_other_view"] = pred_j[extra_key]
            else:
                pred_j_for_dust3r[extra_key] = pred_j[extra_key]
    
    return view_i, view_j, pred_i_for_dust3r, pred_j_for_dust3r, pred_i, pred_j


# ---------------------------------------------------------------------------
# Build all pairs
# ---------------------------------------------------------------------------

pair_indices = list(itertools.combinations(range(3), 2))  # (0,1) (0,2) (1,2)
if not args.no_sym:
    pair_indices += [(j,i) for (i,j) in pair_indices]

view1_list, view2_list = [], []
pred1_list, pred2_list = [], []
# keep originals clouds by image index so we can transform/merge later
gauss_by_img = defaultdict(list)

for (i, j) in pair_indices:
    v1, v2, p1_d, p2_d, p1_full, p2_full = run_pair(i, j)
    view1_list.append(v1)
    view2_list.append(v2)
    pred1_list.append(p1_d)
    pred2_list.append(p2_d)

    # store Gaussians in the *reference (i) frame* for later merging
    gauss_by_img[i].append(p1_full)
    # j side is in i frame; store paired with i so we know to transform after alignment
    # We'll defer until we have poses; see merge step.


# -------------------------------------------------------
#  Collapse lists-of-dicts → dict-of-lists (what optimizer wants)
# -------------------------------------------------------
view1_batch = collate_with_cat(view1_list, lists=True)
view2_batch = collate_with_cat(view2_list, lists=True)
pred1_batch = collate_with_cat(pred1_list, lists=True)
pred2_batch = collate_with_cat(pred2_list, lists=True)

dust3r_output = {
    "view1": view1_batch,
    "view2": view2_batch,
    "pred1": pred1_batch,
    "pred2": pred2_batch,
}

dump_dict(dust3r_output, title="\n=== INPUT TO global_aligner ===")

# ---------------------------------------------------------------------------
# Build the global aligner & optimize
# ---------------------------------------------------------------------------

# For >2 images we want PointCloudOptimizer
scene = global_aligner(dust3r_output, device=device,
                       mode=GlobalAlignerMode.PointCloudOptimizer,
                       verbose=True)

# You can pass schedule='linear' or 'cosine'; use linear here.
loss = scene.compute_global_alignment(init='mst',
                                      niter=args.align_iters,
                                      schedule='cosine',
                                      lr=args.align_lr)

# Extract cam→world poses (tensor [N,4,4])
cams2world = scene.get_im_poses().detach()  # camera → world
focals     = scene.get_focals().detach()
depths     = scene.get_depthmaps()  # list length N
pts3d_list = scene.get_pts3d()      # list length N (H,W,3) in *world* frame


# ---------------------------------------------------------------------------
# Re-express everything relative to image 0 (optional; this matches your sketch)
# ---------------------------------------------------------------------------

P0 = cams2world[0]              # cam0→world
R0 = P0[:3,:3]; t0 = P0[:3,3]
R0_inv = R0.T
t0_inv = -(R0_inv @ t0)

poses_to_0 = {}
for k in range(len(cams2world)):
    P = cams2world[k]
    Rk = P[:3,:3]; tk = P[:3,3]
    Rk0 = R0_inv @ Rk
    tk0 = R0_inv @ (tk - t0)
    poses_to_0[k] = (Rk0, tk0)


# ---------------------------------------------------------------------------
# Collect Gaussians per view, transform into frame-0, merge, dedup
# ---------------------------------------------------------------------------

def _cloud_from_pred(pred):
    """Flatten a Splatt3R pred dict (B,H,W,...) → flat Nx?... we assume B=1."""
    # Accept either 'means' (view frame) or 'means_in_other_view' (already in ref frame)
    if "means" in pred:      means = pred["means"].reshape(-1, 3)
    elif "means_in_other_view" in pred: means = pred["means_in_other_view"].reshape(-1, 3)
    else: raise KeyError("no Gaussian means in pred")

    cloud = {"means": means}
    if "rotations" in pred:       cloud["rotations"] = pred["rotations"].reshape(-1, 4)
    if "sh" in pred:                       # (1,H,W,3,1)  ->  (N,3)
        B, H, W, C, _ = pred["sh"].shape
        cloud["sh"] = pred["sh"].reshape(-1, C)

    if "log_scales" in pred:               # (1,H,W,3)    ->  (N,3)
        cloud["log_scales"] = pred["log_scales"].reshape(-1, 3)

    if "logit_opacities" in pred:          # (1,H,W,1)    ->  (N,1)
        cloud["logit_opacities"] = pred["logit_opacities"].reshape(-1, 1)

    return cloud


clouds0 = []
for img_idx in range(3):
    # concat all pred_i results we stored for that image index
    preds_for_img = gauss_by_img.get(img_idx, [])
    if not preds_for_img:
        continue
    cloud_img = concat([_cloud_from_pred(p) for p in preds_for_img])

    # subsample by confidence if requested
    if args.gaussian_subsample is not None and "sh" in cloud_img:
        # Splatt3R conf is per-pixel; we lost that; fallback random
        if cloud_img["means"].shape[0] > args.gaussian_subsample:
            sel = torch.randperm(cloud_img["means"].shape[0], device=cloud_img["means"].device)[:args.gaussian_subsample]
            for k in list(cloud_img.keys()):
                cloud_img[k] = cloud_img[k][sel]

    # transform into frame-0
    Rk0, tk0 = poses_to_0[img_idx]
    cloud_img["means"] = (cloud_img["means"] @ Rk0.T) + tk0
    clouds0.append(cloud_img)

# merge
if len(clouds0) == 0:
    raise RuntimeError("No Gaussians collected!")
merged = concat(clouds0)

dump_dict(merged, title="\n=== MERGED CLOUD (before dedup) ===")
# dedup
# ---------------------------------------------------------------------------
# De-duplicate centroids but preserve ALL fields
# ---------------------------------------------------------------------------
mask = dedup_xyz(merged["means"], args.radius)        # bool, length = N_means
base_len = mask.shape[0]

for k, v in merged.items():
    if torch.is_tensor(v) and v.shape[0] == base_len:
        merged[k] = v[mask]          # 1-to-1 attributes masked
# all other tensors left untouched

# ---------------------------------------------------------------------------
# 3.  Add (batch=1, height=1) dims so shapes match the reference exporter
# ---------------------------------------------------------------------------
for k, v in merged.items():
    if torch.is_tensor(v) and v.ndim == 2:            # (N, ·)
        merged[k] = v.unsqueeze(0).unsqueeze(0)       # → (1, 1, N, ·)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

# Camera poses (relative to 0) save as txt
cam_txt = os.path.join(args.outdir, "poses_to_0.txt")
with open(cam_txt, "w") as f:
    for k in range(len(cams2world)):
        Rk0, tk0 = poses_to_0[k]
        M = torch.eye(4, device=Rk0.device)
        M[:3,:3] = Rk0
        M[:3, 3] = tk0
        np.savetxt(f, M.cpu().numpy(), fmt="%.8f")
        f.write("\n")
print(f"✓ wrote {cam_txt}")

# NPZ cloud
npz_path = os.path.join(args.outdir, "merged.npz")
np.savez_compressed(npz_path, **{k: v.cpu().numpy() for k, v in merged.items()})
print(f"✓ wrote {npz_path}  (kept {mask.sum().item()} / {len(mask)})")

# Optional PLY
if args.save_ply:
    ply_path = os.path.join(args.outdir, "merged.ply")
    save_as_ply(merged, ply_path)
    print(f"✓ wrote {ply_path}")

print("\nDone.\n")
