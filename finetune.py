#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import os, imageio.v2 as imageio
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.functional import to_tensor
from plyfile import PlyData, PlyElement
from utils.geometry import build_covariance, normalize_intrinsics
from src.pixelsplat_src.cuda_splatting import render_cuda  # compiled extension

class DecoderSplattingCUDA(torch.nn.Module):
    def __init__(self, background_color=(0.0, 0.0, 0.0)):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch: dict, gaussians: dict, image_shape):
        H, W = image_shape
        b = gaussians["means"].shape[0]         # usually 1
        base_pose = batch["context"][0]["camera_pose"]      # [B,4,4]
        inv_base_pose = torch.inverse(base_pose)

        # (B, V, 4, 4)
        extrinsics = torch.stack(
            [t_view["camera_pose"] for t_view in batch["target"]], dim=1
        )
        intrinsics = torch.stack(
            [t_view["camera_intrinsics"] for t_view in batch["target"]], dim=1
        )
        intrinsics = normalize_intrinsics(intrinsics, (H, W))[..., :3, :3]

        # Poses into canonical scene frame (that of the first context view)
        # extrinsics = inv_base_pose[:, None] @ extrinsics
        extrinsics = torch.inverse(extrinsics)
        _, V, _, _ = extrinsics.shape  # number of target views

        # Gaussian buffers
        means = gaussians["means"]                 # [B,G,3]
        covariances = gaussians["covariances"]     # [B,G,3,3]
        harmonics = gaussians["sh"]                # [B,G,3*d]  ->  [B,G,3,d]
        if harmonics.dim() == 3:
            B, G, D = harmonics.shape
            assert D % 3 == 0, "`sh` length must be divisible by 3 (RGB)"
            harmonics = harmonics.view(B, G, 3, D // 3)
        opacities = gaussians["opacities"]         # [B,G]

        # Near / far planes (unused by the loss, but needed by CUDA kernel)
        near = torch.full((b, V), 0.1,  device=means.device)
        far  = torch.full((b, V), 1000., device=means.device)

        with torch.no_grad():
            cam = extrinsics[0,0]                 # w2c of the rendered view
            xyz_cam = (cam[:3,:3] @ means[0].T + cam[:3,3:4]).T
            print("depth (z) min/max  :", xyz_cam[:,2].min().item(),
                                           xyz_cam[:,2].max().item())
        # Run CUDA kernel – everything flattened to (B*V, …)
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
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_intrinsics(fx: float, fy: float, cx: float, cy: float, device) -> torch.Tensor:
    K = torch.eye(4, device=device)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    return K

def save_as_ply(cloud, path):
    """
    Parameters
    ----------
    cloud : dict of tensors produced by fine‑tuning (means, rotations, …).
    path  : Path or str  – destination .ply file.
    """
    xyz   = cloud["means"].reshape(-1, 3).detach().cpu().numpy()
    rot   = cloud["rotations"].reshape(-1, 4).detach().cpu().numpy()
    scale = np.exp(cloud["log_scales"].reshape(-1, 3).detach().cpu().numpy())
    sh0   = cloud["sh"][..., :3].reshape(-1, 3).detach().cpu().numpy()
    opa   = torch.sigmoid(cloud["logit_opacities"]).reshape(-1, 1).detach().cpu().numpy()

    zeros = np.zeros_like(xyz, dtype=np.float32)  # dummy normals
    attrs = np.concatenate([xyz, zeros, sh0, opa, np.log(scale), rot], -1).astype(np.float32)

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

def load_images(paths, device, out_hw=None, save_vis=False):
    imgs = []
    os.makedirs("vis", exist_ok=True)
    for i, p in enumerate(paths):        
        img_np = imageio.imread(p)                     # H×W×C, uint8/16/32
        if img_np.shape[-1] == 4:                      # drop alpha
            img_np = img_np[..., :3]

        if np.issubdtype(img_np.dtype, np.integer):
            img_np = img_np.astype(np.float32) / np.iinfo(img_np.dtype).max
        else:  # assume already float in [0, 1] or [0, 255]
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1.01:       # most PNG/JPEG loaders keep 0‑255
                img_np /= 255.0

        img = to_tensor(img_np).to(device)             # C×H×W, float32

        if out_hw is not None:
            img = F.interpolate(img[None], size=out_hw,
                                 mode="bilinear", align_corners=False)[0]
        imgs.append(img)
        if save_vis:
            imageio.imwrite(
                f"vis/loaded_scaled_{i:04d}.png",
                tensor_to_png(img)
            )
    return imgs

def tensor_to_png(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float in [0,1] →  uint8 H×W×3"""
    t = t.detach().clamp(0, 1).mul_(255).to(torch.uint8)   # on‑GPU is fine
    return t.permute(1, 2, 0).detach().cpu().numpy()                # H,W,3




def load_poses(path, expected):
    arr = np.loadtxt(path).reshape(-1, 4, 4)
    assert arr.shape[0] == expected, f"Expected {expected} poses, got {arr.shape[0]}"
    return [torch.tensor(M, dtype=torch.float32) for M in arr]

def save_debug_image(gt, rgb_pred, it, k):
    os.makedirs("vis", exist_ok=True)

    pred_img = tensor_to_png(rgb_pred[0, 0])
    gt_img   = tensor_to_png(gt[0])

    side_by_side = np.concatenate([pred_img, gt_img], axis=1)

    imageio.imwrite(f"vis/iter{it:05d}_view{k}.png", side_by_side)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Fine‑tune a Gaussian SplatT3r cloud on a set of images"
    )
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Uniform down‑scale applied to images and intrinsics")
    ap.add_argument("--iters", type=int, default=5_000)
    ap.add_argument("--batch", type=int, default=1,
                    help="Random target views per iteration")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--output", type=str, default="finetuned_splatt3r.npz")
    ap.add_argument("--max_points", type=int, default=None,
                    help="Render only a subset of Gaussians each iter")
    ap.add_argument("--render_only", action="store_true",
                help="Skip training and just render one image")
    ap.add_argument("--view", type=int, default=0,
                help="Which image index to render (only with --render_only)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────────────────────────────────────────────
    # 1. Gaussian cloud  (trainable parameters)
    # ──────────────────────────────────────────────────────────────────────
    g_np = np.load(args.npz)
    gaussians = {
        k: torch.nn.Parameter(
            torch.from_numpy(v).float().to(dev),
            requires_grad=(k in ["means", "rotations", "log_scales",
                                 "sh", "logit_opacities"]),
        )
        for k, v in g_np.items()
    }

    # ──────────────────────────────────────────────────────────────────────
    # 2. Data  (images resized once before the loop)
    # ──────────────────────────────────────────────────────────────────────
    scale = args.scale
    W_scaled = int(round(args.W * scale))
    H_scaled = int(round(args.H * scale))

    imgs = load_images(args.images, dev, out_hw=(H_scaled, W_scaled), save_vis=True)
    
    for i, p in enumerate(args.images):
        print(f"{i}: {Path(p).name}")

    num_views = len(imgs)

    poses_c2w   = load_poses(args.poses, num_views)

    intrinsics = make_intrinsics(
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        dev)
    intrinsics[:2] *= scale
    intrinsics = intrinsics.unsqueeze(0)  # [1,4,4]

    # ──────────────────────────────────────────────────────────────────────
    # 3. Renderer & optimiser
    # ──────────────────────────────────────────────────────────────────────
    renderer = DecoderSplattingCUDA(background_color=(0.0, 0.0, 0.0)).to(dev)
    optim = torch.optim.Adam(
        [p for p in gaussians.values() if p.requires_grad], lr=args.lr
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2.5  Quick render‑only path (debugging)
    # ──────────────────────────────────────────────────────────────────────
    if args.render_only:
        choose = torch.randperm(gaussians["means"].shape[0], device=dev)[:args.max_points]

        pt_dict = {
            "means":       gaussians["means"][choose][None],                 # [1, G, 3]
            "covariances": build_covariance(
                               gaussians["log_scales"].exp()[choose],
                               gaussians["rotations"][choose]
                           )[None],                                          # [1, G, 3, 3]
            "sh":          gaussians["sh"][choose][None],                    # [1, G, d]
            "opacities":   torch.sigmoid(gaussians["logit_opacities"]
                                         .squeeze(-1))[choose][None],        # [1, G]
        }
        #pt_dict["covariances"] *= 0.05
        with torch.no_grad():
            # cov: [1, G, 3, 3]
            cov = pt_dict["covariances"]          # [B=1, G, 3, 3]
            diag = cov.diagonal(dim1=-2, dim2=-1) # [1, G, 3]  (σ²_x, σ²_y, σ²_z)
            sigmas = diag.sqrt()                  # standard deviations
            sigma_eff = sigmas.max(-1).values     # pick the largest axis → [1, G]

            print("median σ_eff :", sigma_eff.median().item())
            depths = pt_dict["means"][0].norm(dim=1)   # point → camera distance
            print("median depth :", depths.median().item())

        k = int(np.clip(args.view, 0, len(imgs) - 1))

        batch = {
            "context": [{
                "camera_pose": poses_c2w[0].unsqueeze(0).to(dev),
                "camera_intrinsics": intrinsics,
            }],
            "target":  [{
                "camera_pose": poses_c2w[k].unsqueeze(0).to(dev),
                "camera_intrinsics": intrinsics,
            }],
        }

        with torch.no_grad():
            rgb_pred, _ = renderer(batch, pt_dict, (H_scaled, W_scaled))

        out_png = Path(args.output).with_suffix(f".view{k:03d}.png")
        imageio.imwrite(out_png, tensor_to_png(rgb_pred[0, 0]))
        print(f"✓ wrote {out_png}")
        return            # ---------- NOTHING BELOW HERE RUNS ----------

    # ──────────────────────────────────────────────────────────────────────
    # 4. Training loop
    # ──────────────────────────────────────────────────────────────────────
    for it in range(args.iters):
        optim.zero_grad()
        loss = 0.0

        idxs = np.random.choice(num_views, args.batch, replace=False)
        covariances = build_covariance(
            gaussians["log_scales"].exp(), gaussians["rotations"]
        )                                           # [N,3,3]
        opacities = torch.sigmoid(
            gaussians["logit_opacities"]).squeeze(-1)   # [N]

        # Optionally keep only a subset of Gaussians each iteration
        if args.max_points and args.max_points < gaussians["means"].shape[0]:
            choose = torch.randperm(
                gaussians["means"].shape[0], device=dev)[:args.max_points]
        else:
            choose = slice(None)

        # Gather the subset and add batch dimension
        m = gaussians["means"][choose].contiguous().unsqueeze(0)        # [1,G,3]
        C = covariances[choose].contiguous().unsqueeze(0)               # [1,G,3,3]
        S = gaussians["sh"][choose].contiguous().unsqueeze(0)           # [1,G,sh]
        O = opacities[choose].contiguous().unsqueeze(0)                 # [1,G]

        pt_dict = {"means": m, "covariances": C, "sh": S, "opacities": O}

        for k in idxs:
            batch = {
                "context": [{
                    "camera_pose": poses_c2w[0].unsqueeze(0).to(dev),
                    "camera_intrinsics": intrinsics,
                }],
                "target": [{
                    "camera_pose": poses_c2w[k].unsqueeze(0).to(dev),
                    "camera_intrinsics": intrinsics,
                }],
            }

            rgb_pred, _ = renderer(
                batch, pt_dict, (H_scaled, W_scaled)
            )                                  # [1,1,3,H,W]
            gt = imgs[k].unsqueeze(0)          # [1,3,H,W]
            loss += torch.nn.functional.l1_loss(rgb_pred[:, 0], gt)
            save_debug_image(gt, rgb_pred, it, k)
        (loss / args.batch).backward()
        optim.step()

        if it % 200 == 0 or it == args.iters - 1:
            print(f"[{it:5d}/{args.iters}] L1={loss.item() / args.batch:.6f}")

    np.savez_compressed(
        args.output, **{k: v.detach().cpu().numpy() for k, v in gaussians.items()}
    )
    print(f"\u2713 wrote {args.output}")
    
    ply_path = Path(args.output).with_suffix(".ply")
    save_as_ply(gaussians, ply_path)
    print(f"\u2713 wrote {ply_path}")

if __name__ == "__main__":
    main()
