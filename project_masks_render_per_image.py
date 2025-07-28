#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from einops import rearrange, repeat
import torch.nn as nn
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import csv
from collections import defaultdict
# package‑qualified imports from your repo
from splatt3r.utils.geometry import build_covariance, normalize_intrinsics
from splatt3r.src.pixelsplat_src.cuda_splatting import render_cuda


# ──────────────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────────────
class DecoderSplattingCUDA(torch.nn.Module):
    def __init__(self, background_color=(0.0, 0.0, 0.0)):
        super().__init__()
        self.register_buffer("background_color",
                             torch.tensor(background_color, dtype=torch.float32),
                             persistent=False)

    def forward(self, batch: dict, gaussians: dict, image_shape):
        H, W = image_shape
        b = gaussians["means"].shape[0]

        extrinsics = torch.stack(
            [t_view["camera_pose"] for t_view in batch["target"]], dim=1
        )
        intrinsics = torch.stack(
            [t_view["camera_intrinsics"] for t_view in batch["target"]], dim=1
        )
        intrinsics = normalize_intrinsics(intrinsics, (H, W))[..., :3, :3]

        extrinsics = torch.inverse(extrinsics)  
        _, V, _, _ = extrinsics.shape

        means = gaussians["means"]
        covariances = gaussians["covariances"]
        harmonics = gaussians["sh"]
        if harmonics.dim() == 3:
            B, G, D = harmonics.shape
            assert D % 3 == 0
            harmonics = harmonics.view(B, G, 3, D // 3)
        opacities = gaussians["opacities"]

        near = torch.full((b, V), 0.1,  device=means.device)
        far  = torch.full((b, V), 1000., device=means.device)

        color = render_cuda(
            rearrange(extrinsics,  "b v i j -> (b v) i j"),
            rearrange(intrinsics,  "b v i j -> (b v) i j"),
            rearrange(near,        "b v -> (b v)"),
            rearrange(far,         "b v -> (b v)"),
            (H, W),
            repeat(self.background_color, "c -> (b v) c", b=b, v=V),
            repeat(means,       "b g xyz   -> (b v) g xyz",   v=V),
            repeat(covariances, "b g i j   -> (b v) g i j",   v=V),
            repeat(harmonics,   "b g c dsh -> (b v) g c dsh", v=V),
            repeat(opacities,   "b g       -> (b v) g",       v=V),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=V)
        return color, None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_png(t: torch.Tensor) -> np.ndarray:
    t = t.detach().clamp(0, 1).mul_(255).to(torch.uint8)
    return t.permute(1, 2, 0).cpu().numpy()

def load_image(path: Path, H: int, W: int, device: torch.device) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)
    t = TF.resize(t, [H, W], antialias=True)
    return t.to(device)

def sorted_images(img_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

def load_poses(path: Path, expected: int):
    arr = np.loadtxt(path).reshape(-1, 4, 4)
    assert arr.shape[0] == expected, f"Expected {expected} poses, got {arr.shape[0]}"
    return [torch.tensor(M, dtype=torch.float32) for M in arr]

def make_intrinsics(fx, fy, cx, cy, device):
    K = torch.eye(4, device=device)
    K[0,0], K[1,1] = fx, fy
    K[0,2], K[1,2] = cx, cy
    return K

def project_points(means_w, w2c, K, H, W):
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    Xc = (R @ means_w.T + t[:, None]).T
    z = Xc[:, 2]
    valid_depth = z > 0
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = fx * Xc[:,0] / z + cx
    v = fy * Xc[:,1] / z + cy
    u_int = u.long()
    v_int = v.long()
    valid_uv = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    valid = valid_depth & valid_uv
    return u_int, v_int, valid

def subset_cloud(full, idx):
    cov = build_covariance(full["log_scales"].exp()[idx],
                           full["rotations"][idx])[None]
    opac = torch.sigmoid(full["logit_opacities"].squeeze(-1))[idx][None]
    sh   = full["sh"][idx][None]
    means = full["means"][idx][None]
    return {
        "means": means,
        "covariances": cov,
        "sh": sh,
        "opacities": opac,
    }

def read_intrinsics_file(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {p}")

    vals = {}
    nums = []
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip().lower()] = float(v.strip())
            else:
                nums.extend([float(x) for x in line.split()])

    if vals:
        needed = ["w", "h", "fx", "fy", "cx", "cy"]
        missing = [k for k in needed if k not in vals]
        if missing:
            raise ValueError(f"intrinsics.txt missing keys {missing}. "
                             f"Found: {list(vals.keys())}")
        W  = int(vals["w"])
        H  = int(vals["h"])
        fx = float(vals["fx"])
        fy = float(vals["fy"])
        cx = float(vals["cx"])
        cy = float(vals["cy"])
        scale = float(vals.get("scale", 1.0))
        return W, H, fx, fy, cx, cy, scale

    if len(nums) not in (6, 7):
        raise ValueError(f"intrinsics.txt should contain 6 or 7 numbers, got {len(nums)}")
    W, H, fx, fy, cx, cy = nums[:6]
    scale = nums[6] if len(nums) == 7 else 1.0
    return int(W), int(H), float(fx), float(fy), float(cx), float(cy), float(scale)

def assign_mask_colors(num_masks, device):
    # reproducible-ish colors
    rng = torch.Generator(device=device).manual_seed(0)
    return torch.rand((num_masks, 3), generator=rng, device=device)

def build_colored_subset(full_cloud, all_idx_sorted, colors):
    """
    full_cloud: dict of tensors
    all_idx_sorted: 1D LongTensor of unique gaussian indices
    colors: [len(all_idx_sorted), 3] RGB in [0,1]
    → returns gaussians dict with SH DC painted to `colors`
    """
    pt = subset_cloud(full_cloud, all_idx_sorted)
    sh = pt["sh"]  # [1, G, 3*d]
    B, G, D = sh.shape
    d = D // 3
    sh.zero_()
    # set DC
    sh[0, :, 0]       = colors[:, 0]
    sh[0, :, d + 0]   = colors[:, 1]
    sh[0, :, 2 * d + 0] = colors[:, 2]
    return pt


def save_as_ply(full_cloud, idx, path):
    """
    Write a PLY with the same fields as your global.py writer, but only for the
    subset of indices 'idx'.
    """
    means = full_cloud["means"][idx].detach().cpu().numpy()
    rot   = full_cloud["rotations"][idx].detach().cpu().numpy()
    scale = torch.exp(full_cloud["log_scales"][idx]).detach().cpu().numpy()
    sh0   = full_cloud["sh"][idx][..., :3].detach().cpu().numpy()
    opa   = torch.sigmoid(full_cloud["logit_opacities"][idx]).detach().cpu().numpy()

    zeros = np.zeros_like(means, dtype=np.float32)
    attrs = np.concatenate([means, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)

    names = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    ]
    dtype = np.dtype([(n, "f4") for n in names])
    verts = np.empty(attrs.shape[0], dtype=dtype)
    verts[:] = list(map(tuple, attrs))

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-image mask renders (original colors) + full-cloud sanity + one composite 'colored-by-mask' render."
    )
    parser.add_argument("root", type=Path, help="Base directory, e.g. data/room0_14_16")
    parser.add_argument("--scale", type=float, default=None,
                        help="Override scale from intrinsics.txt (default: use file value or 1.0)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Cap Gaussians globally (applies to full+mask+composite).")
    args = parser.parse_args()

    root = args.root.resolve()

    # expected layout
    imgs_dir   = root / "imgs"
    gsam_dir   = root / "gsam_npz"
    gaussians  = root / "merged_global.npz"
    poses_file = root / "poses_w2c.txt"
    intr_file  = root / "intrinsics.txt"
    out_dir    = root / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # intrinsics & (default) scale
    W, H, fx, fy, cx, cy, scale_file = read_intrinsics_file(intr_file)
    scale = args.scale if args.scale is not None else scale_file

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted_images(imgs_dir)
    assert len(images) > 0, f"No images found in {imgs_dir}"
    num_views = len(images)

    # cloud
    g_np = np.load(gaussians, allow_pickle=True)
    full_cloud = {k: torch.from_numpy(v).float().to(dev) for k, v in g_np.items()}
    G = full_cloud["means"].shape[0]

    # poses
    poses_w2c = load_poses(poses_file, num_views)
    poses_w2c = [p.to(dev) for p in poses_w2c]

    # dims
    Hs = int(round(H * scale))
    Ws = int(round(W * scale))

    K = make_intrinsics(fx, fy, cx, cy, dev)
    if scale != 1.0:
        K[:2] *= scale

    renderer = DecoderSplattingCUDA().to(dev)
    intrinsics_batch = K.unsqueeze(0)


    # choose global subset if max_points is set (same subset used everywhere)
    if args.max_points is not None and G > args.max_points:
        choose_global = torch.randperm(G, device=dev)[:args.max_points]
    else:
        choose_global = torch.arange(G, device=dev)

    # Precompute a full-cloud subset structure once (original colors)
    def make_full_subset():
        return subset_cloud(full_cloud, choose_global)

    # ---- accumulate across ALL views (moved out of the loop) ----
    view_rows = []
    mask_rows = []

    for view_idx, img_path in enumerate(images):
        print(f"[view {view_idx}] {img_path.name}")

        # GT image (for renders only)
        gt = load_image(img_path, Hs, Ws, dev)
        gt_img = to_png(gt)

        # ---------- Full cloud render (pose-order sanity check) ----------
        pt_full = make_full_subset()
        batch_full = {
            "context": [{
                "camera_pose": poses_w2c[view_idx].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
            "target":  [{
                "camera_pose": poses_w2c[view_idx].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
        }
        with torch.no_grad():
            rgb_pred_full, _ = renderer(batch_full, pt_full, (Hs, Ws))
        full_img = to_png(rgb_pred_full[0, 0])
        side_full = np.concatenate([gt_img, full_img], axis=1)
        imageio.imwrite(out_dir / f"view{view_idx:03d}_FULL.png", side_full)

        # ---------- Baseline per-view metrics (ALWAYS) ----------
        means_w = full_cloud["means"]
        u, v, valid = project_points(means_w, poses_w2c[view_idx], K, Hs, Ws)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)

        # respect global subset when counting visible
        if choose_global.numel() != G:
            keep_mask = torch.zeros_like(valid, dtype=torch.bool)
            keep_mask[choose_global] = True
            valid_idx = valid_idx[keep_mask[valid_idx]]

        gauss_visible = int(valid_idx.numel())
        row_view = {
            "view_id": view_idx,
            "gauss_visible": gauss_visible,
            "gauss_assigned": 0,      # will update if masks exist
            "view_coverage": 0.0,     # will update if masks exist
        }

        # ---------- Per-mask renders & metrics ----------
        gsam_path = gsam_dir / f"{img_path.stem}_groundedsam.npz"
        if not gsam_path.exists():
            print(f"  [WARN] missing {gsam_path}, writing zero-coverage row")
            view_rows.append(row_view)
            continue

        gsam = np.load(gsam_path, allow_pickle=True)
        masks_np = gsam["masks"]
        labels_np = gsam["labels"] if "labels" in gsam else np.array([], dtype=object)
        M = masks_np.shape[0]
        if M == 0:
            print("  [WARN] no masks in GSAM file, writing zero-coverage row")
            view_rows.append(row_view)
            continue

        # resize masks to (Hs, Ws)
        masks_t = torch.from_numpy(masks_np.astype(np.float32)).to(dev)
        if masks_t.shape[-2:] != (Hs, Ws):
            masks_t = F.interpolate(masks_t[:, None], size=(Hs, Ws), mode="nearest").squeeze(1)
        masks_t = masks_t.bool()

        # gaussians visible in this view (already have valid_idx after subset)
        u_valid = u[valid_idx]
        v_valid = v[valid_idx]

        # Collect per-mask gaussians & composite assignment
        local_map = {}
        used_indices = []
        for m_id in range(M):
            inside = masks_t[m_id][v_valid, u_valid]
            g_idx = valid_idx[inside]
            if g_idx.numel() == 0:
                continue
            local_map[m_id] = g_idx
            used_indices.append(g_idx)

            # Per-mask render + metrics
            label_str = str(labels_np[m_id]) if m_id < len(labels_np) else f"mask{m_id}"
            pt = subset_cloud(full_cloud, g_idx)
            batch = {
                "context": [{"camera_pose": poses_w2c[view_idx].unsqueeze(0),
                             "camera_intrinsics": intrinsics_batch}],
                "target":  [{"camera_pose": poses_w2c[view_idx].unsqueeze(0),
                             "camera_intrinsics": intrinsics_batch}],
            }
            with torch.no_grad():
                rgb_pred, _ = renderer(batch, pt, (Hs, Ws))
            render_img = to_png(rgb_pred[0, 0])
            side = np.concatenate([gt_img, render_img], axis=1)
            safe_label = label_str[:32].replace(" ", "_")
            imageio.imwrite(out_dir / f"view{view_idx:03d}_mask{m_id:03d}_{safe_label}.png", side)

            # per-mask metrics
            u_m = u_valid[inside]
            v_m = v_valid[inside]
            lin = (v_m.long() * Ws + u_m.long())
            unique_pix = int(torch.unique(lin).numel())
            mask_pixels = int(masks_t[m_id].sum().item())
            pixel_coverage = unique_pix / float(max(1, mask_pixels))
            mask_rows.append({
                "view_id": view_idx,
                "mask_id": m_id,
                "label": label_str,
                "gauss_in_mask": int(g_idx.numel()),
                "gauss_visible_view": int(gauss_visible),
                "mask_hit_rate": float(g_idx.numel()) / float(max(1, gauss_visible)),
                "mask_pixels": mask_pixels,
                "mask_hit_pixels": unique_pix,
                "mask_pixel_coverage": pixel_coverage,
            })

        # finalize per-view row (even if some masks were empty)
        if len(used_indices):
            all_idx = torch.unique(torch.cat(used_indices, dim=0))
            gauss_assigned = int(all_idx.numel())
            row_view["gauss_assigned"] = gauss_assigned
            row_view["view_coverage"] = gauss_assigned / float(max(1, gauss_visible))

            # composite colored-by-mask render (optional)
            mask_colors = assign_mask_colors(M, dev)
            color_for = torch.zeros((all_idx.numel(), 3), device=dev)
            filled = torch.zeros(all_idx.numel(), dtype=torch.bool, device=dev)
            pos_in_all = {int(i.item()): p for p, i in enumerate(all_idx)}
            for m_id, g_idx in local_map.items():
                c = mask_colors[m_id]
                for gi in g_idx.tolist():
                    p = pos_in_all.get(gi, None)
                    if p is None or filled[p]:
                        continue
                    color_for[p] = c
                    filled[p] = True
            pt_comp = build_colored_subset(full_cloud, all_idx, color_for)
            batch_comp = {
                "context": [{"camera_pose": poses_w2c[view_idx].unsqueeze(0),
                             "camera_intrinsics": intrinsics_batch}],
                "target":  [{"camera_pose": poses_w2c[view_idx].unsqueeze(0),
                             "camera_intrinsics": intrinsics_batch}],
            }
            with torch.no_grad():
                rgb_pred_comp, _ = renderer(batch_comp, pt_comp, (Hs, Ws))
            comp_img = to_png(rgb_pred_comp[0, 0])
            imageio.imwrite(out_dir / f"view{view_idx:03d}_MASKCOMP.png", np.concatenate([gt_img, comp_img], axis=1))

        view_rows.append(row_view)

    # ---- WRITE CSVs (now contain all views) ----
    view_csv = out_dir / "view_metrics.csv"
    mask_csv = out_dir / "mask_metrics.csv"
    if view_rows:
        with open(view_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
            w.writeheader()
            w.writerows(view_rows)
        print("wrote", view_csv)
    if mask_rows:
        with open(mask_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
            w.writeheader()
            w.writerows(mask_rows)
        print("wrote", mask_csv)


if __name__ == "__main__":
    main()
