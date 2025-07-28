#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

# deps from your repo
from splatt3r.utils.geometry import build_covariance, normalize_intrinsics
from splatt3r.src.pixelsplat_src.cuda_splatting import render_cuda


# ──────────────────────────────────────────────────────────────────────────────
# Fixed intrinsics (requested)
# ──────────────────────────────────────────────────────────────────────────────
W_FIX  = 1200
H_FIX  = 680
FX_FIX = 600.0
FY_FIX = 600.0
CX_FIX = 600.0
CY_FIX = 340.0


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
        b = gaussians["means"].shape[0]

        extrinsics = torch.stack([t["camera_pose"] for t in batch["target"]], dim=1)
        intrinsics = torch.stack([t["camera_intrinsics"] for t in batch["target"]], dim=1)
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

        near = torch.full((b, V), 0.1, device=means.device)
        far  = torch.full((b, V), 1000.0, device=means.device)

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


def _to_png(t: torch.Tensor) -> np.ndarray:
    t = t.detach().clamp(0, 1).mul_(255).to(torch.uint8)
    return t.permute(1, 2, 0).cpu().numpy()


def _load_image(path: Path, H: int, W: int, device: torch.device) -> torch.Tensor:
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


def _ordered_images(img_dir: Path, list_file: Path):
    if list_file.exists():
        order = []
        with list_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx_str, name = line.split(maxsplit=1)
                order.append((int(idx_str), img_dir / name))
        order.sort(key=lambda x: x[0])
        return [p for _, p in order]
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def _load_poses(path: Path, expected: int):
    arr = np.loadtxt(path).reshape(-1, 4, 4)
    assert arr.shape[0] == expected, f"Expected {expected} poses, got {arr.shape[0]}"
    return [torch.tensor(M, dtype=torch.float32) for M in arr]


def _subset_from_full(full_cloud: dict, idx: torch.Tensor):
    cov = build_covariance(full_cloud["log_scales"].exp()[idx],
                           full_cloud["rotations"][idx])[None]
    opac = torch.sigmoid(full_cloud["logit_opacities"].squeeze(-1))[idx][None]
    sh   = full_cloud["sh"][idx][None]
    means = full_cloud["means"][idx][None]
    return {"means": means, "covariances": cov, "sh": sh, "opacities": opac}


def _make_intrinsics(device: torch.device):
    K = torch.eye(4, device=device)
    K[0, 0], K[1, 1] = FX_FIX, FY_FIX
    K[0, 2], K[1, 2] = CX_FIX, CY_FIX
    return K


def render_pose_set(
    root,
    poses_path,
    gaussians_path,
    tag=None,
    scale_override=None,   # kept for API compatibility; ignored (fixed intrinsics)
    max_points=None,
):
    """
    Write side-by-side GT | render images for a given pose + gaussians pair.
    Uses fixed intrinsics: W=1200, H=680, fx=fy=600, cx=600, cy=340.

    root            : scene directory (contains imgs/ and image_list.txt)
    poses_path      : path to poses file (4x4 per view)
    gaussians_path  : npz with keys: means, log_scales, rotations, sh, logit_opacities
    tag             : e.g., "GLOBAL" or "ICP" (names output folder)
    max_points      : optional cap on number of Gaussians (random subset)
    """
    root = Path(root)
    poses_path = Path(poses_path)
    gaussians_path = Path(gaussians_path)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs_dir = root / "imgs"
    out_dir  = root / f"vis_{(tag or 'POSE').lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _ordered_images(imgs_dir, root / "image_list.txt")
    assert len(images) > 0, f"No images found in {imgs_dir}"
    num_views = len(images)

    Hs, Ws = H_FIX, W_FIX
    K = _make_intrinsics(dev)
    intrinsics_batch = K.unsqueeze(0)

    g_np = np.load(gaussians_path, allow_pickle=True)
    full_cloud = {k: torch.from_numpy(v).float().to(dev) for k, v in g_np.items()}
    G = full_cloud["means"].shape[0]
    if max_points is not None and G > max_points:
        choose_idx = torch.randperm(G, device=dev)[:max_points]
    else:
        choose_idx = torch.arange(G, device=dev)

    poses_c2w = _load_poses(poses_path, num_views)
    poses_c2w = [p.to(dev) for p in poses_c2w]

    renderer = DecoderSplattingCUDA().to(dev)

    def make_subset():
        return _subset_from_full(full_cloud, choose_idx)

    for view_idx, img_path in enumerate(images):
        gt = _load_image(img_path, Hs, Ws, dev)
        gt_img = _to_png(gt)

        pt = make_subset()
        batch = {
            "context": [{
                "camera_pose": poses_c2w[view_idx].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
            "target":  [{
                "camera_pose": poses_c2w[view_idx].unsqueeze(0),
                "camera_intrinsics": intrinsics_batch,
            }],
        }

        with torch.no_grad():
            rgb_pred, _ = renderer(batch, pt, (Hs, Ws))

        pred_img = _to_png(rgb_pred[0, 0])
        side = np.concatenate([gt_img, pred_img], axis=1)
        imageio.imwrite(out_dir / f"view{view_idx:03d}_FULL.png", side)

    print(f"[render_pose_set] wrote {num_views} images to {out_dir}")
