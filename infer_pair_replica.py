#!/usr/bin/env python3
import argparse, os, re, sys, torch, numpy as np
from huggingface_hub import hf_hub_download

# ── repo imports ────────────────────────────────────────────────────────────
sys.path.extend(['src/mast3r_src', 'src/mast3r_src/dust3r', 'src/pixelsplat_src'])
from dust3r.utils.image import load_images
import main
import utils.export as export

# try P3D for quaternion maths; warn if unavailable
try:
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
    HAVE_P3D = True
except ImportError:
    print("⚠️  PyTorch3D not found – rotations will NOT be lifted to world frame.")
    HAVE_P3D = False

# ── helpers ────────────────────────────────────────────────────────────────
def load_c2w(traj_file: str, frame_idx: int) -> torch.Tensor:
    """Return 4×4 camera→world matrix from traj_w_cgl.txt."""
    mats = np.loadtxt(traj_file, dtype=np.float32).reshape(-1, 4, 4)
    w2c = mats[frame_idx]
    return torch.from_numpy(np.linalg.inv(w2c))  # (4,4)

def parse_idx(img_path: str) -> int:
    """rgb/000123.jpg -> 123"""
    m = re.search(r'(\d+)\D*$', os.path.splitext(os.path.basename(img_path))[0])
    return int(m.group(1)) if m else 0

def lift_cloud(cloud, R, t):
    """In-place: move mean & rot into world frame."""
    cloud['mean'] = (R @ cloud['mean'].T + t[:, None]).T
    if HAVE_P3D and 'qvec' in cloud:
        R_cam = quaternion_to_matrix(cloud['qvec'])
        cloud['qvec'] = matrix_to_quaternion((R @ R_cam.cpu()).to(R.device))

# ── CLI ────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('img1'); p.add_argument('img2')
p.add_argument('--traj-file', required=True, help='Replica traj_w_cgl.txt')
p.add_argument('--outdir', default='results')
p.add_argument('--model-dir', default=None)
p.add_argument('--save-npz', action='store_true')
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── checkpoint ────────────────────────────────────────────────────────────
ckpt = hf_hub_download("brandonsmart/splatt3r_v1.0",
                       "epoch=19-step=1200.ckpt",
                       local_dir=args.model_dir,
                       local_dir_use_symlinks=False,
                       resume_download=True)

model = main.MAST3RGaussians.load_from_checkpoint(ckpt, device).eval().to(device)

# ── images  & forward ──────────────────────────────────────────────────────
imgs = load_images([args.img1, args.img2], size=512, verbose=False)
for im in imgs:
    im['img'], im['original_img'] = im['img'].to(device), im['original_img'].to(device)
    im['true_shape'] = torch.as_tensor(im['true_shape'])

pred1, pred2 = model(imgs[0], imgs[1])  # camera-1 frame

# ── lift to Replica world frame ────────────────────────────────────────────
idx1, idx2 = parse_idx(args.img1), parse_idx(args.img2)
c2w1 = load_c2w(args.traj_file, idx1).to(device); R1, t1 = c2w1[:3,:3], c2w1[:3,3]

lift_cloud(pred1, R1, t1)
lift_cloud(pred2, R1, t1)        # pred2 still expressed in cam-1 frame

# ── write outputs ──────────────────────────────────────────────────────────
os.makedirs(args.outdir, exist_ok=True)
ply_path = os.path.join(args.outdir, 'gaussians.ply')
export.save_as_ply(pred1, pred2, ply_path)
print(f"✓ wrote {ply_path}")

if args.save_npz:
    npz_path = os.path.join(args.outdir, 'gaussians.npz')
    np.savez_compressed(
        npz_path,
        xyz=torch.cat((pred1['mean'],   pred2['mean']  )).cpu().numpy(),
        rot=torch.cat((pred1['qvec'],   pred2['qvec']  )).cpu().numpy() if 'qvec' in pred1 else None,
        scale=torch.cat((pred1['scale'], pred2['scale'])).cpu().numpy(),
        sh=torch.cat((pred1['sh_feat'], pred2['sh_feat'])).cpu().numpy(),
        opacity=torch.cat((pred1['opacity'], pred2['opacity'])).cpu().numpy(),
    )
    print(f"✓ wrote {npz_path}")
