import torch
from einops import rearrange, repeat

from .cuda_splatting import render_cuda
from splatt3r.utils.geometry import normalize_intrinsics


class DecoderSplattingCUDA(torch.nn.Module):
    """Differentiable renderer for 3‑D Gaussian splats (CUDA backend).

    Shapes
    ------
    Context & target in ``batch``
      camera_pose        : [b,4,4]  (c2w)
      camera_intrinsics  : [b,4,4]

    ``pred`` dictionary
      means              : [b,n,3]
      covariances        : [b,n,3,3]
      sh                 : [b,n,C] (spherical‑harmonics coefficients)
      opacities          : [b,n]

    Returns
    -------
      color : [b,v,3,H,W]  RGB for each target view
    """

    def __init__(self, background_color=(0.0, 0.0, 0.0)):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch, pred, image_shape):
        base_c2w = batch["context"][0]["camera_pose"]            # [b,4,4]
        inv_base = torch.inverse(base_c2w)                         # [b,4,4]

        # Stack target extrinsics & intrinsics along view dim v
        extrinsics = torch.stack(
            [t["camera_pose"] for t in batch["target"]], dim=1
        )  # [b,v,4,4]
        intrinsics = torch.stack(
            [t["camera_intrinsics"] for t in batch["target"]], dim=1
        )  # [b,v,4,4]

        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]  # [b,v,3,3]
        extrinsics = inv_base[:, None] @ extrinsics                                # world→cam_rel

        # Expand point cloud along v (so we can flatten to (b*v))
        b, v = extrinsics.shape[:2]
        means = rearrange(pred["means"], "b n d -> b 1 n d").expand(-1, v, -1, -1)
        covs = rearrange(pred["covariances"], "b n i j -> b 1 n i j").expand(-1, v, -1, -1, -1)
        sh = rearrange(pred["sh"], "b n c -> b 1 n c").expand(-1, v, -1, -1)
        opac = rearrange(pred["opacities"], "b n -> b 1 n").expand(-1, v, -1)

        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            torch.full((b * v,), 0.1, device=extrinsics.device),
            torch.full((b * v,), 1000.0, device=extrinsics.device),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            rearrange(means, "b v n d    -> (b v) n d"),
            rearrange(covs, "b v n i j -> (b v) n i j"),
            rearrange(sh, "b v n c     -> (b v) n c 1"),
            rearrange(opac, "b v n      -> (b v) n"),
        )

        return rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v), None
