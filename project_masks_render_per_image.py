#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from einops import rearrange, repeat

# package‑qualified imports from your repo
from splatt3r.utils.geometry import build_covariance, normalize_intrinsics
from splatt3r.src.pixelsplat_src.cuda_splatting import render_cuda
import torch.nn as nn  # base class


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

        extrinsics = torch.inverse(extrinsics)  # want w2c
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

def project_points(means_w, pose_c2w, K, H, W):
    w2c = torch.inverse(pose_c2w)
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
    """
    Accepts either:
      - 6 or 7 whitespace-separated numbers:
            W H fx fy cx cy [scale]
      - or key=value lines (W=..., fx=..., etc.)
    Returns: (W, H, fx, fy, cx, cy, scale)
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-image mask renders (original colors) + full-cloud renders for pose ordering check."
    )
    parser.add_argument("root", type=Path, help="Base directory, e.g. data/room0_14_16")
    parser.add_argument("--scale", type=float, default=None,
                        help="Override scale from intrinsics.txt (default: use file value or 1.0)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Cap Gaussians per render (applied to BOTH full-cloud and per-mask renders).")
    args = parser.parse_args()

    root = args.root.resolve()

    # expected layout
    imgs_dir   = root / "imgs"
    gsam_dir   = root / "gsam_npz"
    gaussians  = root / "merged_icp.npz"
    poses_file = root / "poses_after_icp.txt"
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
    poses_c2w = load_poses(poses_file, num_views)
    poses_c2w = [p.to(dev) for p in poses_c2w]

    # dims
    Hs = int(round(H * scale))
    Ws = int(round(W * scale))

    K = make_intrinsics(fx, fy, cx, cy, dev)
    if scale != 1.0:
        K[:2] *= scale

    renderer = DecoderSplattingCUDA().to(dev)
    intrinsics_batch = K.unsqueeze(0)

    # choose global subset if max_points is set (same subset used for full & masks)
    if args.max_points is not None and G > args.max_points:
        choose_global = torch.randperm(G, device=dev)[:args.max_points]
    else:
        choose_global = torch.arange(G, device=dev)

    # Precompute a full-cloud subset structure once (original colors)
    def make_full_subset():
        return subset_cloud(full_cloud, choose_global)

    for view_idx, img_path in enumerate(images):
        print(f"[view {view_idx}] {img_path.name}")

        # GT
        gt = load_image(img_path, Hs, Ws, dev)

        # ---------- Full cloud render (pose-order sanity check) ----------
        pt_full = make_full_subset()
        batch_full = {
            "context": [{
                "camera_pose": poses_c2w[0].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
            "target":  [{
                "camera_pose": poses_c2w[view_idx].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
        }
        with torch.no_grad():
            rgb_pred_full, _ = renderer(batch_full, pt_full, (Hs, Ws))
        full_img = to_png(rgb_pred_full[0, 0])
        gt_img   = to_png(gt)
        side_full = np.concatenate([gt_img, full_img], axis=1)
        imageio.imwrite(out_dir / f"view{view_idx:03d}_FULL.png", side_full)

        # ---------- Per-mask renders (original colors) ----------
        gsam_path = gsam_dir / f"{img_path.stem}_groundedsam.npz"
        if not gsam_path.exists():
            print(f"  [WARN] missing {gsam_path}, skipping masks")
            continue
        print(f"view {view_idx}: {img_path.stem}  →  {gsam_path.name}")
        gsam = np.load(gsam_path, allow_pickle=True)
        masks_np = gsam["masks"]
        labels_np = gsam["labels"] if "labels" in gsam else np.array([], dtype=object)
        M = masks_np.shape[0]
        if M == 0:
            print("  no masks")
            continue

        # Reproject *full* cloud (even if we later subset for rendering)
        means_w = full_cloud["means"]
        u, v, valid = project_points(means_w, poses_c2w[view_idx], K, Hs, Ws)
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        u_valid   = u[valid]
        v_valid   = v[valid]

        # But only keep those that are also in our global subset
        if choose_global.numel() != G:
            # Map: which of valid_idx are in choose_global?
            mask_keep = torch.zeros_like(valid, dtype=torch.bool)
            mask_keep[choose_global] = True
            valid_idx = valid_idx[mask_keep[valid_idx]]
            u_valid   = u[valid_idx]
            v_valid   = v[valid_idx]

        masks_t = torch.from_numpy(masks_np.astype(np.bool_)).to(dev)

        for m in range(M):
            mask = masks_t[m]
            label_str = str(labels_np[m]) if m < len(labels_np) else f"mask{m}"

            inside = mask[v_valid, u_valid]
            g_idx = valid_idx[inside]
            if g_idx.numel() == 0:
                continue

            pt = subset_cloud(full_cloud, g_idx)  # original colors

            batch = {
                "context": [{
                    "camera_pose": poses_c2w[0].unsqueeze(0),
                    "camera_intrinsics": intrinsics_batch,
                }],
                "target":  [{
                    "camera_pose": poses_c2w[view_idx].unsqueeze(0),
                    "camera_intrinsics": intrinsics_batch,
                }],
            }

            with torch.no_grad():
                rgb_pred, _ = renderer(batch, pt, (Hs, Ws))

            render_img = to_png(rgb_pred[0, 0])
            side = np.concatenate([gt_img, render_img], axis=1)
            safe_label = label_str[:32].replace(" ", "_")
            out_name = f"view{view_idx:03d}_mask{m:03d}_{safe_label}.png"
            imageio.imwrite(out_dir / out_name, side)

    print(f"Done. Wrote images to {out_dir}")


if __name__ == "__main__":
    main()
