#!/usr/bin/env python3
import argparse, os, sys, itertools, shutil
from collections import defaultdict
import numpy as np, torch
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import json
import csv

sys.path.extend(["src/mast3r_src", "src/mast3r_src/dust3r", "src/pixelsplat_src"])

from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import main
from render_pose_sanity import render_pose_set

try:
    import open3d as o3d
except ImportError:
    sys.exit("Install open3d: pip install open3d")


def rotmat_to_angle_deg(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def umeyama_similarity(A, B):
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
    var_A = (A_c**2).sum() / A.shape[0]
    s = np.trace(np.diag(D) @ S) / var_A
    t = mu_B - s * (R @ mu_A)
    return s, R, t


def get_dust3r_order(scene):
    order = []
    for i, j in scene.edges:
        if i not in order:
            order.append(i)
        if j not in order:
            order.append(j)
    return order


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


def pose_inv(M):
    R = M[:3, :3]
    t = M[:3, 3]
    Minv = torch.eye(4, device=M.device, dtype=M.dtype)
    Minv[:3, :3] = R.T
    Minv[:3, 3] = -R.T @ t
    return Minv


ap = argparse.ArgumentParser()
ap.add_argument("images", nargs=5)
ap.add_argument("--outdir", required=True)
ap.add_argument("--radius", type=float, default=0.004)
ap.add_argument("--model-dir", default=None)
ap.add_argument("--align-iters", type=int, default=1000)
ap.add_argument("--align-lr", type=float, default=0.005)
ap.add_argument("--gaussian-subsample", type=int)
ap.add_argument("--icp-thresh", type=float, default=0.005)
ap.add_argument("--save-ply", action="store_true")
ap.add_argument("--no-sym", action="store_true")
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.outdir, exist_ok=True)

imgs_outdir = os.path.join(args.outdir, "imgs")
os.makedirs(imgs_outdir, exist_ok=True)
with open(os.path.join(args.outdir, "image_list.txt"), "w") as f:
    for i, src in enumerate(args.images):
        dst = os.path.join(imgs_outdir, os.path.basename(src))
        shutil.copy2(src, dst)
        f.write(f"{i} {os.path.basename(src)}\n")
print(f"✓ copied {len(args.images)} images to {imgs_outdir}")

metrics = {"per_view": [], "pairwise_pose": []}
metrics_path = os.path.join(args.outdir, "metrics.json")
csv_path = os.path.join(args.outdir, "metrics_per_view.csv")
poses_global_path = os.path.join(args.outdir, "poses_w2c.txt")
poses_icp_path = os.path.join(args.outdir, "poses_after_icp.txt")
poses_best_path = os.path.join(args.outdir, "poses_best.txt")

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


pairs = list(itertools.combinations(range(len(args.images)), 2))
pairs += [(j, i) for i, j in pairs]

v1, v2, p1, p2 = [], [], [], []
clouds_pairwise = []
for (i, j) in pairs:
    a, b, c, d, full_i, full_j = run_pair(i, j)
    v1.append(a); v2.append(b); p1.append(c); p2.append(d)
    clouds_pairwise.append((i, full_i))

bat = lambda x: collate_with_cat(x, lists=True)
d3 = {"view1": bat(v1), "view2": bat(v2), "pred1": bat(p1), "pred2": bat(p2)}
scene = global_aligner(d3, device=dev, mode=GlobalAlignerMode.PointCloudOptimizer)
log_path = os.path.join(args.outdir, "global_align_loss.txt")
scene.compute_global_alignment(log_path=log_path, init="mst", niter=args.align_iters,
                               schedule="cosine", lr=args.align_lr)

# Build mapping from image index -> row in P0 using DUSt3R's order
P0 = scene.get_im_poses().detach()  # (N,4,4) cam-to-world in scene row order
im_order = [int(x) for x in get_dust3r_order(scene)]  # list: row -> img_idx
row_by_img = {img_idx: row for row, img_idx in enumerate(im_order)}

# Choose a reference view and rebase so that its pose becomes identity
REF_VIEW = 0
if REF_VIEW not in row_by_img:
    # fall back to the first view in DUSt3R's order
    REF_VIEW = im_order[0]
row0 = row_by_img[REF_VIEW]

A = torch.eye(4, device=P0.device, dtype=P0.dtype)
A[:3, :3] = P0[row0, :3, :3].T
A[:3, 3]  = -P0[row0, :3, :3].T @ P0[row0, :3, 3]

# Optional sanity check:
# assert torch.allclose(A @ P0[row0], torch.eye(4, device=P0.device, dtype=P0.dtype), atol=1e-5)

# Dict of cam-to-world, keyed by original img index (do not assume 0..N-1 density)
c2w_by_view = {idx: (A @ P0[row_by_img[idx]]).clone() for idx in row_by_img.keys()}
poses_c2w = c2w_by_view  # backward‑compat alias; dict keyed by img_idx

w2c_by_view = {i: pose_inv(M) for i, M in c2w_by_view.items()}

with open(poses_global_path, "w") as f:
    for idx in sorted(row_by_img.keys()):
        M = w2c_by_view[idx]
        fname = os.path.basename(args.images[idx]) if idx < len(args.images) else f"view_{idx:03d}"
        f.write(f"# {fname}\n")
        np.savetxt(f, M.cpu().numpy(), fmt="%.8f")
        f.write("\n")
print("✓ wrote", poses_global_path)

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
cloud_views = []
for c_idx, (view_id, pred_full) in enumerate(clouds_pairwise):
    cl = cloud_from(pred_full)  # ensure 'means' is in the cam frame of view_id
    if args.gaussian_subsample and cl["means"].shape[0] > args.gaussian_subsample:
        sel = torch.randperm(cl["means"].shape[0], device=dev)[:args.gaussian_subsample]
        for k in list(cl):
            cl[k] = cl[k][sel]
    if view_id not in c2w_by_view:
        raise KeyError(f"view_id {view_id} not in c2w map; known keys: {sorted(c2w_by_view)}")
    Mv = c2w_by_view[view_id]
    R, t = Mv[:3, :3], Mv[:3, 3]
    cl["means"] = (cl["means"] @ R.to(dev).T) + t.to(dev)
    out_npz = os.path.join(args.outdir, f"cloud_pair_{c_idx:02d}_view{view_id}.npz")
 #   np.savez_compressed(out_npz, **{k: v.cpu().numpy() for k, v in cl.items()})
    if args.save_ply:
        save_as_ply(cl, out_npz.replace(".npz", ".ply"))
    print(f"saved {out_npz} ({cl['means'].shape[0]} pts)")
    clouds.append(cl)
    cloud_views.append(view_id)
merged_global = {}
if len(clouds) > 0:
    keys0 = list(clouds[0].keys())
    for k in keys0:
        merged_global[k] = torch.cat([c[k] for c in clouds if k in c], dim=0)

out_global_cloud = os.path.join(args.outdir, "merged_global.npz")
if merged_global:
    np.savez_compressed(out_global_cloud, **{k: v.cpu().numpy() for k, v in merged_global.items()})
    print(f"wrote {out_global_cloud} (pre-ICP cloud for GLOBAL renders)")
    try:
        render_pose_set(
            root=args.outdir,
            poses_path=poses_global_path,
            gaussians_path=out_global_cloud,
            tag="GLOBAL"
        )
    except Exception as e:
        print("⚠ Global pose render failed:", e)
else:
    out_global_cloud = None

base_idx = cloud_views.index(REF_VIEW)
if base_idx != 0:
    clouds[0], clouds[base_idx] = clouds[base_idx], clouds[0]
    cloud_views[0], cloud_views[base_idx] = cloud_views[base_idx], cloud_views[0]
base = clouds[0]
base_np = base["means"].detach().cpu().numpy()

sim3_scales = [1.0]
for cl in clouds[1:]:
    A_pts = cl["means"].detach().cpu().numpy()
    B_pts = base_np
    n = min(5000, min(len(A_pts), len(B_pts)))
    idxA = np.random.choice(len(A_pts), n, replace=False)
    idxB = np.random.choice(len(B_pts), n, replace=False)
    s, _, _ = umeyama_similarity(A_pts[idxA], B_pts[idxB])
    sim3_scales.append(float(s))
print("Sim(3) scales vs. base:", sim3_scales)

vox, th_c, th_f = 0.02, 0.08, args.icp_thresh
b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
b_coarse = b_full.voxel_down_sample(vox)
b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

icp_per_cloud = []
merged = {k: [base[k]] for k in base.keys()}

for c_idx in range(len(clouds)):
    view_id = cloud_views[c_idx]
    if view_id == 0:
        continue
    cl = clouds[c_idx]
    s_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cl["means"].cpu().numpy()))
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
    T = icp2.transformation.copy()
    M_i = poses_c2w[view_id]
    M_i = torch.from_numpy(T).to(device=dev, dtype=cl["means"].dtype) @ M_i
    poses_c2w[view_id] = M_i
    for k in merged.keys():
        if k in cl:
            merged[k].append(cl[k])
    ang, tnorm = rotmat_to_angle_deg(T[:3, :3]), np.linalg.norm(T[:3, 3])
    icp_per_cloud.append({
        "cloud_idx": c_idx,
        "view_id": int(view_id),
        "T_icp": T.copy(),
        "fitness": float(icp2.fitness),
        "rmse": float(icp2.inlier_rmse),
        "rot_deg": float(ang),
        "trans": float(tnorm)
    })
    print(f"[ICP view{view_id}] fit={icp2.fitness:.3f} rmse={icp2.inlier_rmse:.4f} dθ={ang:.2f}° dt={tnorm:.4f}")
    row = {
        "cloud_idx": int(c_idx),
        "view_id": int(view_id),
        "icp_fine_fitness": float(icp2.fitness),
        "icp_fine_rmse": float(icp2.inlier_rmse),
        "icp_rot_deg": float(ang),
        "icp_trans": float(tnorm),
        "num_points_cloud": int(cl["means"].shape[0]),
        "num_points_base_before_merge": int(base["means"].shape[0])
    }
    metrics["per_view"].append(row)
    R_t = torch.from_numpy(T[:3, :3]).to(device=dev, dtype=cl["means"].dtype)
    t_t = torch.from_numpy(T[:3, 3]).to(device=dev, dtype=cl["means"].dtype)
    cl["means"] = (cl["means"] @ R_t.T) + t_t
    base_np = np.vstack([base_np, cl["means"].cpu().numpy()])
    b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
    b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
    b_coarse = b_full.voxel_down_sample(vox)
    b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

base = {k: torch.cat(v, dim=0) for k, v in merged.items()}
num_views = 5
best_poses = [None] * num_views
for view_id in range(num_views):
    # pick the best ICP candidate for this view (highest fitness, then lowest RMSE)
    cands = [c for c in icp_per_cloud if int(c["view_id"]) == view_id]
    row_idx = row_by_img[view_id]  # was view2row[view_id]
    P_glob_raw = P0[row_idx].cpu().numpy()
    if cands:
        cand = sorted(cands, key=lambda x: (-x["fitness"], x["rmse"]))[0]
        T_icp = cand["T_icp"]
        best_poses[view_id] = T_icp @ P_glob_raw
    else:
        best_poses[view_id] = P_glob_raw

with open(poses_best_path, "w") as f:
    for row, img_idx in enumerate(im_order):
        M = best_poses[img_idx]
        fname = os.path.basename(args.images[img_idx])
        f.write(f"# {fname}\n")
        np.savetxt(f, M, fmt="%.8f")
        f.write("\n")
print("✓ wrote", poses_best_path)

P_before = [P.to(dev) for P in P0]
P_after = [poses_c2w[img_idx] for img_idx in im_order]
for a in range(len(P_before)):
    for b in range(a + 1, len(P_before)):
        Rab = pose_inv(P_before[a]) @ P_before[b]
        Rhat = pose_inv(P_after[a]) @ P_after[b]
        Delta = torch.linalg.inv(Rhat) @ Rab
        R = Delta[:3, :3].cpu().numpy()
        t = Delta[:3, 3].cpu().numpy()
        metrics["pairwise_pose"].append({
            "i": int(a), "j": int(b),
            "rel_rot_deg": float(rotmat_to_angle_deg(R)),
            "rel_trans": float(np.linalg.norm(t))
        })

with open(poses_icp_path, "w") as f:
    for row, img_idx in enumerate(im_order):
        M = poses_c2w[img_idx]
        fname = os.path.basename(args.images[img_idx])
        f.write(f"# {fname}\n")
        np.savetxt(f, M.cpu().numpy(), fmt="%.8f")
        f.write("\n")
print("✓ wrote", poses_icp_path)

if args.radius > 0:
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
    num_pts = int(m.sum().item())
else:
    N0 = base["means"].shape[0]
    if "sh" in base:
        sh = base["sh"]
        if sh.ndim == 2 and sh.shape[1] == 1 and sh.shape[0] == 3 * N0:
            base["sh"] = sh.view(N0, 3)
    num_pts = int(base["means"].shape[0])

out = os.path.join(args.outdir, "merged_icp.npz")
print("saving keys:", {k: v.shape for k, v in base.items()})
np.savez_compressed(out, **{k: v.cpu().numpy() for k, v in base.items()})
print(f"wrote {out} ({num_pts} unique pts)")


try:
    render_pose_set(root=args.outdir,
                    poses_path=poses_icp_path,
                    gaussians_path=out,
                    tag="ICP")
except Exception as e:
    print("⚠ ICP pose render failed:", e)


if args.save_ply:
    save_as_ply(base, out.replace(".npz", ".ply"))
    print("wrote PLY")

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("✓ wrote", metrics_path)

with open(csv_path, "w", newline="") as f:
    if metrics["per_view"]:
        fields = sorted({k for r in metrics["per_view"] for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_view"])
print("✓ wrote", csv_path)

# ---- Auto-plot ICP metrics PNGs (optional) ----
try:
    from plot_icp_metrics import save_icp_plots
    save_icp_plots(csv_path)  # outputs to <outdir>/metrics_per_view/
    print("✓ wrote ICP metric plots next to the CSV")
except Exception as e:
    print("⚠ plotting failed:", e)

print("Done")
