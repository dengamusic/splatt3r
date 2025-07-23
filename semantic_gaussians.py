#!/usr/bin/env python3
import argparse, numpy as np, cv2, os, glob, re
from collections import defaultdict

'''
def load_gaussians(npz_path):
    data = np.load(npz_path)
    return data['means'], data.get('rotations', None)
'''
def load_gaussians(npz_path):
    data = np.load(npz_path)
    means = data['means']
    # Flatten view + spatial dims into N×3
    #assert means.ndim == 4 and means.shape[3] == 3
    rotations = data.get('rotations', None)
    if rotations is not None:
        if rotations.ndim > 2:
            rotations = rotations.reshape(-1, rotations.shape[-1])
    return means, rotations


def load_pose(traj_file, idx):
    mats = np.loadtxt(traj_file, dtype=np.float32).reshape(-1,4,4)
    if idx >= len(mats):
        raise IndexError(f"Frame {idx} > traj contains {len(mats)} poses")
    return mats[idx]


def load_masks_features(folder, img_name):
    """Load all mask-feature pairs for the given image."""
    results = {}
    base = os.path.splitext(img_name)[0]  # e.g., rgb_1400

    mask_paths = sorted(glob.glob(os.path.join(folder, f"{base}_mask_*.npy")))
    
    for mask_path in mask_paths:
        # Extract index from mask filename
        match = re.search(r'_mask_(\d+)\.npy$', mask_path)
        if not match:
            continue
        idx = match.group(1)
        feat_path = os.path.join(folder, f"{base}_feat_{idx}.npy")



        mask = np.load(mask_path)
        feature = np.load(feat_path)
        results[mask_path] = (mask, feature)

    return results

def parse_frame_idx(img_name):
    m = re.search(r'(\d+)(?=\.png$)', img_name)
    if not m:
        raise ValueError(f"Can't find frame idx in {img_name}")
    return int(m.group(1))

def project_to_pixels(means, K, c2w):
    #print("means.shape:", means.shape)
    w2c = np.linalg.inv(c2w)
    xyz_cam = (means @ w2c[:3,:3].T) + w2c[:3,3]
    uv = (K @ xyz_cam.T).T
    uv = uv[:,:2] / uv[:,2:3]
    
    return uv, xyz_cam[:,2]

def load_5_poses(pose_txt_path):
    poses = np.loadtxt(pose_txt_path, dtype=np.float32).reshape(5, 4, 4)
    return poses

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gauss-npz', required=True)
    #p.add_argument('--traj', required=True)
    p.add_argument('--poses', required=True, help="text file with 5 poses, one 4x4 matrix per pose (in order)")
    p.add_argument('--mask-folder', required=True, help="folder containing masks and .txt files")
    p.add_argument('--img1', required=True)
    p.add_argument('--img2', required=True)
    p.add_argument('--img3', required=True)
    p.add_argument('--img4', required=True)
    p.add_argument('--img5', required=True)
    p.add_argument('--K', default=[700, 0, 600, 0, 700, 340, 0, 0, 1], nargs=9, type=float)
    p.add_argument('--out', default='gauss_sem.npz')
    args = p.parse_args()

    means, rotations = load_gaussians(args.gauss_npz)
    N = means.shape[0]
    K = np.array(args.K).reshape(3,3)


    feat_dim = None  # to be set after loading first feature
    feat_sums = np.zeros((N, 512), dtype=np.float32)  # assuming 512-d CLIP
    feat_counts = np.zeros(N, dtype=np.int32)

    poses = load_5_poses(args.poses)
    for img, c2w in zip([args.img1, args.img2, args.img3, args.img4, args.img5], poses):
        #frame_idx = parse_frame_idx(img)
        #c2w = load_pose(args.traj, frame_idx)
        uv, depth = project_to_pixels(means, K, c2w)
        mask_info = load_masks_features(args.mask_folder, img)

        ''''''
        for mask_fn, (mask, feature) in mask_info.items():
            H, W = mask.shape[:2]
            valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

            for j in np.where(valid)[0]:
                u, v = uv[j]
                x, y = int(round(u))-1, int(round(v))-1
                if mask[y, x] > 0:
                    if feat_dim is None:
                        feat_dim = feature.shape[-1]
                    feat_sums[j] += feature.squeeze()
                    feat_counts[j] += 1
            '''
            for j, (u, v, d) in enumerate(zip(uv[:,0], uv[:,1], depth)):
                #print("uv min/max:", uv.min(), uv.max())
                #print("image shape:", H, W)

                
                if d <= 0 or u < 0 or v < 0 or u >= W or v >= H:
                    continue
                x, y = int(round(u)), int(round(v))
                if 0 <= x < W and 0 <= y < H and mask[y, x] > 0:
                    if feat_dim is None:
                        feat_dim = feature.shape[-1]
                    feat_sums[j] += feature.squeeze()
                    feat_counts[j] += 1
                '''    


    # Compute mean features
    valid = feat_counts > 0
    mean_feats = np.zeros_like(feat_sums)
    mean_feats[valid] = feat_sums[valid] / feat_counts[valid][:, None]

    out = dict(np.load(args.gauss_npz, allow_pickle=True))
    out["clip_feats"] = mean_feats  # shape: [N, 512]
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out} with per-Gaussian CLIP features.")

    num_assigned = (feat_counts > 0).sum()
    print(f"Assigned CLIP features to {num_assigned} / {N} Gaussians")


if __name__ == "__main__":
    main()
