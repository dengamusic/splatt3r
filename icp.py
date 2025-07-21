#!/usr/bin/env python3
# -------------------------------------------------
#  multiframe_splatt3r_icp.py
#     • loads per-view NPZ clouds
#     • Open3D ICP refine → merge → de-dup → save
# -------------------------------------------------
import argparse, os, sys, glob
from collections import defaultdict
import numpy as np, torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except ImportError:
    sys.exit("Install open3d: pip install open3d")

# -------- full save_as_ply (same as above) --------
def save_as_ply(pred, path):
    """Full attribute dump identical to the original big script."""
    means = pred["means"].reshape(-1,3).cpu().numpy()
    rot   = pred["rotations"].reshape(-1,4).cpu().numpy()
    scale = np.exp(pred["log_scales"].reshape(-1,3).cpu().numpy())
    sh0   = pred["sh"][..., :3].reshape(-1,3).cpu().numpy()
    opac  = torch.sigmoid(pred["logit_opacities"]).reshape(-1,1).cpu().numpy()

    zeros = np.zeros_like(means, dtype=np.float32)
    attrs = np.concatenate([means, zeros, sh0, opac,
                            np.log(scale), rot], axis=-1).astype(np.float32)

    names = ["x","y","z",
             "nx","ny","nz",
             "f_dc_0","f_dc_1","f_dc_2",
             "opacity",
             "scale_0","scale_1","scale_2",
             "rot_0","rot_1","rot_2","rot_3"]
    verts = np.empty(attrs.shape[0],
                     dtype=np.dtype([(n,"f4") for n in names]))
    verts[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(verts,"vertex")],text=False).write(path)
# --------------------------------------------------
def load_cloud(p,dev): return {k:torch.as_tensor(v,device=dev) for k,v in np.load(p).items()}
def concat(ds):
    out=defaultdict(list)
    for d in ds:
        for k,v in d.items(): out[k].append(v)
    return {k:torch.cat(vs,0) for k,vs in out.items()}

def dedup_xyz(means, radius):
    """Return a boolean mask that keeps one centre per <radius cluster."""
    dev = means.device                     # remember where the tensor lives
    xyz = means.cpu().numpy()              # → NumPy for KD-tree
    tree = cKDTree(xyz)
    dup  = tree.query_pairs(radius)

    keep = np.ones(len(xyz), dtype=bool)
    for i, j in dup:
        keep[j] = False

    return torch.from_numpy(keep).to(dev)  # send mask back to the original device

def icp(src, dst, th, max_iter=1500):
    src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.astype(np.float64)))
    dst_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst.astype(np.float64)))

    result = o3d.pipelines.registration.registration_icp(
        src_pc, dst_pc,
        th, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return result

# ---------------- CLI -----------------
ap=argparse.ArgumentParser()
ap.add_argument("clouds",nargs="+")
ap.add_argument("--outdir",required=True)
ap.add_argument("--icp-thresh",type=float,default=0.01)
ap.add_argument("--radius",type=float,default=0.003)
ap.add_argument("--save-ply",action="store_true")
args=ap.parse_args()

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir,exist_ok=True)
paths=sorted(sum([glob.glob(p) for p in args.clouds],[]))
if not paths: sys.exit("No clouds found.")
clouds=[load_cloud(p,dev) for p in paths]
print(f"Loaded {len(clouds)} clouds.")
voxel_coarse = 0.02          # 2 cm grid
th_coarse    = 0.08          # 8 cm inlier radius
th_fine      = args.icp_thresh

base = clouds[0]
base_np = base["means"].cpu().numpy()

# pre-compute down-sampled + normals for the base cloud once
base_pc_full   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
base_pc_full.estimate_normals(                           # ← add this
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2,
                                         max_nn=30))
base_pc_coarse = base_pc_full.voxel_down_sample(voxel_coarse)
base_pc_coarse.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2, max_nn=30))

for i, cl in enumerate(clouds[1:], 1):
    src_full = cl["means"].cpu().numpy()
    src_pc_full   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_full))
    src_pc_full.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2,
                                             max_nn=30))
    src_pc_coarse = src_pc_full.voxel_down_sample(voxel_coarse)
    src_pc_coarse.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2, max_nn=30))

    # -------- coarse, point-to-plane -----------------------------------
    icp1 = o3d.pipelines.registration.registration_icp(
        src_pc_coarse, base_pc_coarse,
        th_coarse, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    # -------- refine on full res ---------------------------------------
    icp2 = o3d.pipelines.registration.registration_icp(
        src_pc_full, base_pc_full,
        th_fine, icp1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    T = icp2.transformation
    R, t = T[:3, :3], T[:3, 3]
    ang_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"[ICP {i}] coarse fitness={icp1.fitness:.3f}, "
          f"fine fitness={icp2.fitness:.3f}, "
          f"Δθ={ang_deg:6.2f}°, |Δt|={np.linalg.norm(t):.4f}")

    # apply to tensor cloud
    R_t = torch.from_numpy(R.copy()).to(device=dev, dtype=cl["means"].dtype)
    t_t = torch.from_numpy(t.copy()).to(device=dev, dtype=cl["means"].dtype)
    cl["means"] = (cl["means"] @ R_t.T) + t_t

    # merge into base (still keep growing set)
    base   = concat([base, cl])
    base_np = base["means"].cpu().numpy()
    base_pc_full   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))

    base_pc_full.estimate_normals(                      #  ←  add this line
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_coarse*2, max_nn=30))

    base_pc_coarse = base_pc_full.voxel_down_sample(voxel_coarse)
    base_pc_coarse.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2, max_nn=30))

    print(f"   ↳ merged {i}  (size {base_np.shape[0]})")

# ------------------  after ICP merging  ------------------
mask      = dedup_xyz(base["means"], args.radius)     # bool (N_before,)
N_before  = mask.shape[0]
keep_idx  = torch.nonzero(mask, as_tuple=False).squeeze(1)

# ---------- 1️⃣  REPAIR sh so it has shape (N_before, 3)  ----------
if "sh" in base:
    sh = base["sh"]
    # many checkpoints store SH as (N*3,1); collapse it to (N,3)
    if sh.ndim == 2 and sh.shape[1] == 1 and sh.shape[0] % 3 == 0:
        n_triplets = sh.shape[0] // 3
        if n_triplets == N_before:                 # confirmed one-to-one
            base["sh"] = sh.view(n_triplets, 3)
# ------------------------------------------------------------------

def mask_axes(t: torch.Tensor) -> torch.Tensor:
    """Slice t along every axis that equals N_before."""
    axes = [ax for ax, s in enumerate(t.shape) if s == N_before]
    if not axes:                                   # nothing to trim
        return t
    sl = [slice(None)] * t.ndim
    for ax in axes:
        sl[ax] = keep_idx
    return t[tuple(sl)]

for k in list(base.keys()):
    if torch.is_tensor(base[k]):
        base[k] = mask_axes(base[k])

out_npz=os.path.join(args.outdir,"merged_icp.npz")
np.savez_compressed(out_npz,**{k:v.cpu().numpy() for k,v in base.items()})
print(f"✓ wrote {out_npz}  ({mask.sum().item()} unique pts)")

# ---- quick sanity check -------------------------------------------------
print("\n[shape audit]")
for k, v in base.items():
    if torch.is_tensor(v):
        flag = "🟥" if N_before in v.shape else "🟩"
        print(f"{k:18s} {str(tuple(v.shape)):>18}  {flag}")
print("[end audit]\n")
# ------------------------------------------------------------------------

if args.save_ply:
    save_as_ply(base,out_npz.replace('.npz','.ply'))
    print("✓ wrote PLY")

print("Done – ICP-refined cloud complete.")
