#!/usr/bin/env python3
# █████████████████████████████████████████████████
#  multiframe_splatt3r_global.py
#     • Inference with Splatt3r
#     • Global alignment with DUSt3R
#     • Saves per-view NPZ clouds in frame-0
# -------------------------------------------------
#  Usage example:
#    python multiframe_splatt3r_global.py img0.jpg img1.jpg img2.jpg \
#           --outdir outputs_room0_0_25_50 --align-lr 0.03 --save-ply
# -------------------------------------------------

import argparse, os, sys, itertools
from collections import defaultdict
import torch, numpy as np
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

#  Add local Splatt3r / DUSt3R to path
sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])

from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import main  # Splatt3R's LightningModule
# ------------------------------------------------- utilities
def concat(list_of_dicts):
    acc = defaultdict(list)
    for d in list_of_dicts:
        for k,v in d.items():  acc[k].append(v.detach())
    return {k: torch.cat(vs,0) for k,vs in acc.items()}

def save_as_ply(cloud, path):
    """Quick PLY dump (for inspection)."""
    xyz   = cloud["means"].reshape(-1,3).cpu().numpy()
    rot   = cloud["rotations"].reshape(-1,4).cpu().numpy()
    scale = np.exp(cloud["log_scales"].reshape(-1,3).cpu().numpy())
    sh0   = cloud["sh"][..., :3].reshape(-1,3).cpu().numpy()
    opa   = torch.sigmoid(cloud["logit_opacities"]).reshape(-1,1).cpu().numpy()

    zeros = np.zeros_like(xyz, dtype=np.float32)
    attrs = np.concatenate([xyz, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)
    names = ["x","y","z","nx","ny","nz",
             "f_dc_0","f_dc_1","f_dc_2",
             "opacity","scale_0","scale_1","scale_2",
             "rot_0","rot_1","rot_2","rot_3"]
    dt = np.dtype([(n,"f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dt); verts[:] = list(map(tuple,attrs))
    PlyData([PlyElement.describe(verts,"vertex")], text=False).write(path)
# ------------------------------------------------- CLI
ap = argparse.ArgumentParser()
ap.add_argument("images", nargs=3, help="exactly three images")
ap.add_argument("--outdir", required=True)
ap.add_argument("--radius", type=float, default=0.003)
ap.add_argument("--model-dir", default=None)
ap.add_argument("--align-iters", type=int, default=1000)
ap.add_argument("--align-lr", type=float, default=0.01)
ap.add_argument("--gaussian-subsample", type=int)
ap.add_argument("--save-ply", action="store_true")
ap.add_argument("--no-sym", action="store_true", help="omit reverse-direction pairs")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir, exist_ok=True)
# ------------------------------------------------- load model
ckpt = hf_hub_download("brandonsmart/splatt3r_v1.0",
                       "epoch=19-step=1200.ckpt",
                       local_dir=args.model_dir, local_dir_use_symlinks=False)
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device).eval().to(device)
# ------------------------------------------------- load images
IM_SIZE = 512
imgs = load_images(args.images, size=IM_SIZE, verbose=False)
for idx,im in enumerate(imgs):
    im["img"], im["original_img"] = im["img"].to(device), im["original_img"].to(device)
    im["true_shape"], im["idx"]   = torch.as_tensor(im["true_shape"]), idx
# ------------------------------------------------- run Splatt3r on all pairs
@torch.no_grad()
def run_pair(i,j):
    v_i, v_j = imgs[i], imgs[j]
    p_i, p_j = model(v_i, v_j)
    for p in (p_i,p_j):
        if "log_scales" not in p and "scales" in p:
            p["log_scales"]=p["scales"].log()
        if "logit_opacities" not in p and "opacities" in p:
            alpha=p["opacities"].clamp(1e-4,1-1e-4); p["logit_opacities"]=(alpha/(1-alpha)).log()
    d_i = {"pts3d":p_i["pts3d"], "conf":p_i["conf"]}
    d_j = {"pts3d_in_other_view":p_j["pts3d_in_other_view"],
           "conf":p_j["conf"], "pts3d":p_j["pts3d_in_other_view"]}
    for k in ("means","rotations","sh","log_scales","logit_opacities"):
        if k in p_j:
            if k=="means":
                d_j["means_in_other_view"]=p_j[k]
            else:
                d_j[k]=p_j[k]
    return v_i,v_j,d_i,d_j,p_i,p_j

pairs = list(itertools.combinations(range(3),2))
if not args.no_sym: pairs += [(j,i) for (i,j) in pairs]

v1,v2,p1,p2,gauss_by_img = [],[],[],[], defaultdict(list)
for i,j in pairs:
    a,b,c,d,full_i,_ = run_pair(i,j)
    v1.append(a); v2.append(b); p1.append(c); p2.append(d)
    gauss_by_img[i].append(full_i)

bat = lambda x: collate_with_cat(x, lists=True)
dust3r_out = {"view1":bat(v1),"view2":bat(v2),"pred1":bat(p1),"pred2":bat(p2)}
# ------------------------------------------------- global alignment
scene = global_aligner(dust3r_out, device=device,
                       mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init="mst", niter=args.align_iters,
                               schedule="cosine", lr=args.align_lr)
cams2world = scene.get_im_poses().detach()
P0 = cams2world[0]; R0i, t0 = P0[:3,:3].T, P0[:3,3]

def to_frame0(P): R,t=P[:3,:3],P[:3,3]; return R0i@R, R0i@(t-t0)
poses_to_0 = {k:to_frame0(P) for k,P in enumerate(cams2world)}
# ------------------------------------------------- build & save per-view clouds
def cloud_from(pred):
    m = pred["means"].reshape(-1,3) if "means" in pred else pred["means_in_other_view"].reshape(-1,3)
    c={"means":m}
    for k in ("rotations","sh","log_scales","logit_opacities"): 
        if k in pred: c[k]=pred[k].reshape(-1,pred[k].shape[-1] if k=="sh" else 3 if k=="log_scales" else 4 if k=="rotations" else 1)
    return c

all_clouds=[]
for idx in range(3):
    cl = concat([cloud_from(p) for p in gauss_by_img[idx]])
    if args.gaussian_subsample and cl["means"].shape[0]>args.gaussian_subsample:
        sel = torch.randperm(cl["means"].shape[0], device=device)[:args.gaussian_subsample]
        for k in list(cl): cl[k]=cl[k][sel]
    R,t = poses_to_0[idx]; R,t = R.to(device), t.to(device)
    cl["means"] = (cl["means"] @ R.T) + t
    npz = os.path.join(args.outdir, f"cloud_{idx}.npz")
    np.savez_compressed(npz, **{k:v.cpu().numpy() for k,v in cl.items()})
    if args.save_ply: save_as_ply(cl, npz.replace(".npz",".ply"))
    all_clouds.append(cl)
    print(f"✓ saved {npz}  ({cl['means'].shape[0]} pts)")

# also dump poses
with open(os.path.join(args.outdir,"poses_to_0.txt"),"w") as f:
    for k,(R,t) in poses_to_0.items():
        M=torch.eye(4); M[:3,:3],M[:3,3]=R,t
        np.savetxt(f,M.numpy(),fmt="%.8f"); f.write("\n")
print("✓ global alignment complete.")
