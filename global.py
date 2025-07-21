#!/usr/bin/env python3
import argparse, os, sys, itertools
from collections import defaultdict
import numpy as np, torch
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])

from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import main

try:
    import open3d as o3d
except ImportError:
    sys.exit("Install open3d: pip install open3d")


def concat(ds):
    acc = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            acc[k].append(v.detach() if torch.is_tensor(v) else v)
    return {k: torch.cat(vs, 0) for k, vs in acc.items()}


def save_as_ply(cl, p):
    xyz = cl["means"].reshape(-1, 3).cpu().numpy()
    rot = cl["rotations"].reshape(-1, 4).cpu().numpy()
    scale = np.exp(cl["log_scales"].reshape(-1, 3).cpu().numpy())
    sh0 = cl["sh"][..., :3].reshape(-1, 3).cpu().numpy()
    opa = torch.sigmoid(cl["logit_opacities"]).reshape(-1, 1).cpu().numpy()
    zeros = np.zeros_like(xyz, dtype=np.float32)
    attrs = np.concatenate([xyz, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dt = np.dtype([(n, "f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dt)
    verts[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(p)


def dedup_xyz(means, r):
    dev = means.device
    xyz = means.cpu().numpy()
    dup = cKDTree(xyz).query_pairs(r)
    keep = np.ones(len(xyz), dtype=bool)
    for i, j in dup:
        keep[j] = False
    return torch.from_numpy(keep).to(dev)


ap = argparse.ArgumentParser()
ap.add_argument("images", nargs=5)
ap.add_argument("--outdir", required=True)
ap.add_argument("--radius", type=float, default=0.003)
ap.add_argument("--model-dir", default=None)
ap.add_argument("--align-iters", type=int, default=1000)
ap.add_argument("--align-lr", type=float, default=0.01)
ap.add_argument("--gaussian-subsample", type=int)
ap.add_argument("--icp-thresh", type=float, default=0.01)
ap.add_argument("--save-ply", action="store_true")
ap.add_argument("--no-sym", action="store_true")
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir, exist_ok=True)

ckpt = hf_hub_download("brandonsmart/splatt3r_v1.0", "epoch=19-step=1200.ckpt",
                       local_dir=args.model_dir, local_dir_use_symlinks=False)
model = main.MAST3RGaussians.load_from_checkpoint(ckpt, dev).eval().to(dev)

imgs = load_images(args.images, size=512, verbose=False)
for idx, im in enumerate(imgs):
    im["img"], im["original_img"] = im["img"].to(dev), im["original_img"].to(dev)
    im["true_shape"], im["idx"] = torch.as_tensor(im["true_shape"]), idx


@torch.no_grad()
def run_pair(i, j):
    v_i, v_j = imgs[i], imgs[j]
    p_i, p_j = model(v_i, v_j)
    for p in (p_i, p_j):
        if "log_scales" not in p and "scales" in p:
            p["log_scales"] = p["scales"].log()
        if "logit_opacities" not in p and "opacities" in p:
            a = p["opacities"].clamp(1e-4, 1 - 1e-4)
            p["logit_opacities"] = (a / (1 - a)).log()
    d_i = {"pts3d": p_i["pts3d"], "conf": p_i["conf"]}
    d_j = {"pts3d_in_other_view": p_j["pts3d_in_other_view"], "conf": p_j["conf"],
           "pts3d": p_j["pts3d_in_other_view"]}
    for k in ("means", "rotations", "sh", "log_scales", "logit_opacities"):
        if k in p_j:
            d_j["means_in_other_view" if k == "means" else k] = p_j[k]
    return v_i, v_j, d_i, d_j, p_i, p_j


pairs = list(itertools.combinations(range(5), 2))
if not args.no_sym:
    pairs += [(j, i) for i, j in pairs]

v1, v2, p1, p2, gauss = [], [], [], [], defaultdict(list)
for i, j in pairs:
    a, b, c, d, full_i, full_j = run_pair(i, j)
    v1.append(a); v2.append(b); p1.append(c); p2.append(d)
    gauss[i].append(full_i)

bat = lambda x: collate_with_cat(x, lists=True)
d3 = {"view1": bat(v1), "view2": bat(v2), "pred1": bat(p1), "pred2": bat(p2)}
scene = global_aligner(d3, device=dev, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init="mst", niter=args.align_iters,
                               schedule="cosine", lr=args.align_lr)
P0 = scene.get_im_poses().detach()
R0i, t0 = P0[0][:3, :3].T, P0[0][:3, 3]
poses = {k: (R0i @ P[:3, :3], R0i @ (P[:3, 3] - t0)) for k, P in enumerate(P0)}


def cloud_from(pred):
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


clouds = []
for idx in range(5):
    cl = concat([cloud_from(p) for p in gauss[idx]])
    if args.gaussian_subsample and cl["means"].shape[0] > args.gaussian_subsample:
        sel = torch.randperm(cl["means"].shape[0], device=dev)[:args.gaussian_subsample]
        for k in list(cl):
            cl[k] = cl[k][sel]
    R, t = poses[idx]
    R, t = R.to(dev), t.to(dev)
    cl["means"] = (cl["means"] @ R.T) + t
    npz = os.path.join(args.outdir, f"cloud_{idx}.npz")
    np.savez_compressed(npz, **{k: v.cpu().numpy() for k, v in cl.items()})
    if args.save_ply:
        save_as_ply(cl, npz.replace(".npz", ".ply"))
    clouds.append(cl)
    print(f"saved {npz} ({cl['means'].shape[0]} pts)")

with open(os.path.join(args.outdir, "poses_to_0.txt"), "w") as f:
    for R, t in poses.values():
        M = torch.eye(4); M[:3, :3], M[:3, 3] = R, t
        np.savetxt(f, M.numpy(), fmt="%.8f"); f.write("\n")

vox = 0.02
th_c = 0.08
th_f = args.icp_thresh

base = clouds[0]
base_np = base["means"].cpu().numpy()
b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
b_coarse = b_full.voxel_down_sample(vox)
b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

for i, cl in enumerate(clouds[1:], 1):
    s_np = cl["means"].cpu().numpy()
    s_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s_np))
    s_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
    s_coarse = s_full.voxel_down_sample(vox)
    s_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
    icp1 = o3d.pipelines.registration.registration_icp(
        s_coarse, b_coarse, th_c, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    icp2 = o3d.pipelines.registration.registration_icp(
        s_full, b_full, th_f, icp1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T = icp2.transformation
    R, t = T[:3, :3], T[:3, 3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"[ICP {i}] coarse={icp1.fitness:.3f} fine={icp2.fitness:.3f} dθ={ang:.2f} dt={np.linalg.norm(t):.4f}")
    R_t = torch.from_numpy(R.copy()).to(device=dev, dtype=cl["means"].dtype)
    t_t = torch.from_numpy(t.copy()).to(device=dev, dtype=cl["means"].dtype)
    cl["means"] = (cl["means"] @ R_t.T) + t_t
    base = concat([base, cl])
    base_np = base["means"].cpu().numpy()
    b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
    b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
    b_coarse = b_full.voxel_down_sample(vox)
    b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

m = dedup_xyz(base["means"], args.radius)
N = m.shape[0]
idx = torch.nonzero(m, as_tuple=False).squeeze(1)
if "sh" in base:
    sh = base["sh"]
    if sh.ndim == 2 and sh.shape[1] == 1 and sh.shape[0] % 3 == 0 and sh.shape[0] // 3 == N:
        base["sh"] = sh.view(-1, 3)


def mask_axes(t):
    ax = [a for a, s in enumerate(t.shape) if s == N]
    if not ax:
        return t
    sl = [slice(None)] * t.ndim
    for a in ax:
        sl[a] = idx
    return t[tuple(sl)]


for k in list(base.keys()):
    if torch.is_tensor(base[k]):
        base[k] = mask_axes(base[k])

out = os.path.join(args.outdir, "merged_icp.npz")
np.savez_compressed(out, **{k: v.cpu().numpy() for k, v in base.items()})
print(f"wrote {out} ({m.sum().item()} unique pts)")

if args.save_ply:
    save_as_ply(base, out.replace(".npz", ".ply"))
    print("wrote PLY")
print("Done")
