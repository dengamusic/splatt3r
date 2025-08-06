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

import itertools
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
def iou_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / float(u)

class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

def paint_dc_for_local(pt, local_ids: torch.Tensor, colors: torch.Tensor):
    """
    In-place: zero higher-order SH and set DC to `colors` for the selected local ids.
    pt["sh"]: [1, G, 3*d]
    local_ids: [K] (indices into the subset order)
    colors: [K, 3] in [0,1]
    """
    sh = pt["sh"]
    assert sh.dim() == 3 and sh.shape[0] == 1, "expected sh shape [1, G, 3*d]"
    B, G, D = sh.shape
    d = D // 3
    # zero HO and set DC per channel
    sh[:, local_ids, :] = 0
    sh[:, local_ids, 0]       = colors[:, 0]  # R DC
    sh[:, local_ids, d + 0]   = colors[:, 1]  # G DC
    sh[:, local_ids, 2 * d + 0] = colors[:, 2]  # B DC
    return pt

def save_subset_as_ply_with_sh_override(full_cloud, idx: torch.Tensor, sh_override: torch.Tensor, path: Path):
    """
    Same as save_as_ply but takes the SH from `sh_override` (shape [1, G, 3*d]),
    and extracts DCs from [0, d, 2d] to write f_dc_0..2.
    """
    means = full_cloud["means"][idx].detach().cpu().numpy()
    rot   = full_cloud["rotations"][idx].detach().cpu().numpy()
    scale = torch.exp(full_cloud["log_scales"][idx]).detach().cpu().numpy()
    opa   = torch.sigmoid(full_cloud["logit_opacities"][idx]).detach().cpu().numpy()

    _, G, D = sh_override.shape
    d = D // 3
    sh_dc = torch.stack([
        sh_override[0, :, 0],
        sh_override[0, :, d + 0],
        sh_override[0, :, 2 * d + 0],
    ], dim=-1).detach().cpu().numpy()

    zeros = np.zeros_like(means, dtype=np.float32)
    attrs = np.concatenate([means, zeros, sh_dc, opa, np.log(scale), rot], -1).astype(np.float32)

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
    parser.add_argument("--max_points", type=int, default=700000,
                        help="Cap Gaussians globally (applies to full+mask+composite).")
    parser.add_argument("--merge_iou_thresh", type=float, default=0.25,
                        help="Min IoU between gaussian-index sets to connect groups.")
    parser.add_argument("--merge_clip_thresh", type=float, default=0.30,
                        help="Min CLIP cosine similarity between masks to connect groups.")

    args = parser.parse_args()

    root = args.root.resolve()

    # expected layout
    imgs_dir   = root / "imgs"
    gsam_dir   = root / "gsam_npz"
    gaussians  = root / "merged_icp.npz"
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
    all_groups = []
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

            # Load per-mask CLIP features from preprocessing
        clip_feats_np = gsam["clip"]
        clip_feats_t = torch.from_numpy(clip_feats_np).float()
        clip_feats_t = torch.nn.functional.normalize(clip_feats_t, dim=1)  # cosine-ready
    
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
            
                        # Collect a group only if we have a CLIP feature for this mask
            if clip_feats_t is not None:
                all_groups.append({
                    "view_id": int(view_idx),
                    "mask_id": int(m_id),
                    "idx_set": set(g_idx.detach().long().tolist()),
                    "clip": clip_feats_t[m_id].cpu(),  # keep on CPU
                })

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
            
            if view_idx == 0:
                pos_in_choose = {int(i.item()): p for p, i in enumerate(choose_global)}

                # First-hit-wins assignment of a color per gaussian
                # (if a gaussian falls into overlapping masks)
                mask_colors = assign_mask_colors(M, dev)
                selected_local = []
                selected_colors = []
                taken = torch.zeros(choose_global.numel(), dtype=torch.bool, device=dev)

                for m_id, g_idx in local_map.items():
                    c = mask_colors[m_id]
                    for gi in g_idx.tolist():
                        p = pos_in_choose.get(gi, None)
                        if p is None or taken[p]:
                            continue
                        selected_local.append(p)
                        selected_colors.append(c)
                        taken[p] = True

                if len(selected_local):
                    local_ids = torch.tensor(selected_local, dtype=torch.long, device=dev)
                    colors_local = torch.stack(selected_colors, dim=0)

                    # Start from the full subset (same cloud used everywhere)
                    pt_full_color = subset_cloud(full_cloud, choose_global)
                    # Overwrite SH for only the selected Gaussians
                    paint_dc_for_local(pt_full_color, local_ids, colors_local)

                    # Render from view0
                    batch0 = {
                        "context": [{"camera_pose": poses_w2c[0].unsqueeze(0),
                                     "camera_intrinsics": intrinsics_batch}],
                        "target":  [{"camera_pose": poses_w2c[0].unsqueeze(0),
                                     "camera_intrinsics": intrinsics_batch}],
                    }
                    with torch.no_grad():
                        rgb_pred_total, _ = renderer(batch0, pt_full_color, (Hs, Ws))
                    total_img = to_png(rgb_pred_total[0, 0])
                    imageio.imwrite(out_dir / "view000_FULL_MASKCOLOR_TOTAL.png",
                                    np.concatenate([gt_img, total_img], axis=1))

                    # Optional: write a PLY of the same full subset with recolored DCs
                    save_subset_as_ply_with_sh_override(
                        full_cloud, choose_global, pt_full_color["sh"],
                        out_dir / "full_cloud_colored_by_view0.ply"
                    )


        view_rows.append(row_view)

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-view merging using IoU + CLIP cosine (precomputed features)
    # ──────────────────────────────────────────────────────────────────────────
    if all_groups:
        n = len(all_groups)
        sets  = [g["idx_set"] for g in all_groups]
        views = [g["view_id"] for g in all_groups]
        feats = torch.stack([g["clip"] for g in all_groups], dim=0)  # [n,D], unit-norm

        dsu = _DSU(n)
        for i in range(n):
            Si, fi = sets[i], feats[i]
            for j in range(i + 1, n):
                if iou_sets(Si, sets[j]) < args.merge_iou_thresh:
                    continue
                sim = float(torch.dot(fi, feats[j]).item())  # cosine (unit-norm)
                if sim < args.merge_clip_thresh:
                    continue
                dsu.union(i, j)

        # Connected components → clusters
        comp_map = {}
        for i in range(n):
            r = dsu.find(i)
            comp_map.setdefault(r, []).append(i)

        cluster_infos = []
        gaussian_to_cluster = {}
        for cid, member_idx in enumerate(comp_map.values()):
            member_sets  = [sets[k] for k in member_idx]
            member_feats = feats[member_idx]
            member_views = [views[k] for k in member_idx]

            # Minimal metrics
            union_set = set().union(*member_sets)
            views_count = len(set(member_views))
            pair_ious, pair_sims = [], []
            for a, b in itertools.combinations(range(len(member_idx)), 2):
                pair_ious.append(iou_sets(member_sets[a], member_sets[b]))
                pair_sims.append(float(torch.dot(member_feats[a], member_feats[b]).item()))
            iou_mean = float(sum(pair_ious) / len(pair_ious)) if pair_ious else 1.0
            iou_min  = float(min(pair_ious)) if pair_ious else 1.0
            clip_mean = float(sum(pair_sims) / len(pair_sims)) if pair_sims else 1.0
            clip_min  = float(min(pair_sims)) if pair_sims else 1.0
            score = 0.5 * (iou_mean + clip_mean)

            for gidx in union_set:
                gaussian_to_cluster[gidx] = cid

            cluster_infos.append({
                "cluster_id": cid,
                "n_members": len(member_idx),
                "views": views_count,
                "union_size": len(union_set),
                "iou_mean": round(iou_mean, 6),
                "iou_min":  round(iou_min, 6),
                "clip_mean": round(clip_mean, 6),
                "clip_min":  round(clip_min, 6),
                "score": round(score, 6),
            })

        # Save Gaussian → cluster assignments
        assign = np.full((G,), -1, dtype=np.int32)
        for gidx, cid in gaussian_to_cluster.items():
            if 0 <= gidx < G:
                assign[gidx] = cid
        np.save(out_dir / "cluster_assignments.npy", assign)

                # ───────────────── Save a PLY: clusters → unique colors ─────────────────
        # num clusters and a reproducible color palette
        num_clusters = len(cluster_infos)
        if num_clusters > 0:
            cluster_colors = assign_mask_colors(num_clusters, dev)  # [C,3] in [0,1]

            # Build subset to modify (same subset used everywhere in the script)
            pt_full_color = subset_cloud(full_cloud, choose_global)   # dict with "sh"
            sh = pt_full_color["sh"]                                  # [1,Gs,3*d]
            _, Gs, D = sh.shape
            d = D // 3

            # Map global gaussian index -> local position in choose_global
            pos_in_choose = {int(i.item()): p for p, i in enumerate(choose_global)}

            # Overwrite DC (and zero higher orders) for gaussians that belong to a cluster
            # Unclustered gaussians keep their original SH
            local_ids = []
            colors    = []
            for gidx, cid in gaussian_to_cluster.items():
                p = pos_in_choose.get(int(gidx))
                if p is None:  # not in the saved subset (e.g., cut by --max_points)
                    continue
                local_ids.append(p)
                colors.append(cluster_colors[int(cid)])

            if local_ids:
                local_ids = torch.tensor(local_ids, device=dev, dtype=torch.long)
                colors    = torch.stack(colors, dim=0)  # [K,3]

                # zero higher-order SH for colored gaussians, then set DC per channel
                sh[:, local_ids, :] = 0
                sh[:, local_ids, 0]       = colors[:, 0]  # R
                sh[:, local_ids, d + 0]   = colors[:, 1]  # G
                sh[:, local_ids, 2 * d + 0] = colors[:, 2]  # B

            # Write PLY
            ply_path = out_dir / "full_cloud_clusters_colored.ply"
            save_subset_as_ply_with_sh_override(full_cloud, choose_global, sh, ply_path)
            print("wrote", ply_path)

        # Save a compact CSV summary
        if cluster_infos:
            merge_csv = out_dir / "merged_groups.csv"
            with open(merge_csv, "w", newline="") as f:
                cols = ["cluster_id","n_members","views","union_size",
                        "iou_mean","iou_min","clip_mean","clip_min","score"]
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in cluster_infos:
                    w.writerow(row)
            print("wrote", merge_csv)


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
