
import os
import itertools
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download

# The dust3r/pixelsplat modules are provided by external paths injected by the CLI.
from dust3r.utils.device import collate_with_cat
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from .geometry import rotmat_to_angle_deg, umeyama_similarity, pose_inv, get_dust3r_order
from .clouds import cloud_from_pred, transform_cloud_inplace, save_as_ply, dedup_xyz
from .io_utils import ensure_outdirs, copy_images_and_write_list, write_poses_txt, write_metrics_json, write_metrics_csv

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("Install open3d: pip install open3d")

@dataclass
class Config:
    images: List[str]
    outdir: str
    radius: float = 0.004
    model_dir: str = None
    align_iters: int = 1000
    align_lr: float = 0.005
    gaussian_subsample: int = None
    icp_thresh: float = 0.005
    save_ply: bool = False
    no_sym: bool = False  # retained for CLI parity

class Splatt3rPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = {"per_view": [], "pairwise_pose": []}

        # Output paths
        self.metrics_path = os.path.join(cfg.outdir, "metrics.json")
        self.csv_path = os.path.join(cfg.outdir, "metrics_per_view.csv")
        self.poses_global_path = os.path.join(cfg.outdir, "poses_global.txt")
        self.poses_icp_path = os.path.join(cfg.outdir, "poses_after_icp.txt")
        self.poses_best_path = os.path.join(cfg.outdir, "poses_best.txt")
        self.log_path = os.path.join(cfg.outdir, "global_align_loss.txt")

        # To be filled
        self.imgs: List[Dict] = []
        self.im_order: List[int] = []
        self.view2row: Dict[int, int] = {}
        self.num_views: int = 0
        self.P0: torch.Tensor = None
        self.poses_c2w: List[torch.Tensor] = []

    # -- Setup -------------------------------------------------------------
    def setup_outdirs(self) -> None:
        imgs_outdir = ensure_outdirs(self.cfg.outdir)
        list_path = os.path.join(self.cfg.outdir, "image_list.txt")
        copy_images_and_write_list(self.cfg.images, imgs_outdir, list_path)

    def load_model(self):
        # Upstream provides MAST3RGaussians in a module named 'main'.
        # We expose it under a Splatt3R* alias for naming consistency.
        import main as _mmod
        Splatt3RGaussians = getattr(_mmod, "MAST3RGaussians")
        ckpt = hf_hub_download(
            "brandonsmart/splatt3r_v1.0",
            "epoch=19-step=1200.ckpt",
            local_dir=self.cfg.model_dir,
            local_dir_use_symlinks=False,
        )
        model = Splatt3RGaussians.load_from_checkpoint(ckpt, self.dev).eval().to(self.dev)
        return model

    def load_and_prepare_images(self, size: int = 512):
        imgs = load_images(self.cfg.images, size=size, verbose=False)
        for idx, im in enumerate(imgs):
            im["img"] = im["img"].to(self.dev)
            im["original_img"] = im["original_img"].to(self.dev)
            im["true_shape"], im["idx"] = torch.as_tensor(im["true_shape"]), idx
        self.imgs = imgs

    # -- Pairwise / Global alignment --------------------------------------
    def build_pairwise(self, model):
        from .predictor import run_pair
        pairs = list(itertools.combinations(range(len(self.cfg.images)), 2))
        v1, v2, p1, p2 = [], [], [], []
        clouds_pairwise = []

        for (i, j) in pairs:
            a, b, c, d, full_i, full_j = run_pair(model, self.imgs, i, j)
            v1.append(a); v2.append(b); p1.append(c); p2.append(d)
            clouds_pairwise.append((i, full_i))  # matches original behavior

        bat = lambda x: collate_with_cat(x, lists=True)
        d3 = {"view1": bat(v1), "view2": bat(v2), "pred1": bat(p1), "pred2": bat(p2)}
        return d3, clouds_pairwise

    def run_global_alignment(self, d3):
        scene = global_aligner(d3, device=self.dev, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(
            log_path=self.log_path, init="mst", niter=self.cfg.align_iters,
            schedule="cosine", lr=self.cfg.align_lr
        )
        self.P0 = scene.get_im_poses().detach()
        self.im_order = get_dust3r_order(scene)
        self.view2row = {img_idx: row for row, img_idx in enumerate(self.im_order)}
        self.num_views = len(self.im_order)
        return scene

    def compute_c2w(self, ref_view: int = 0):
        row0 = self.view2row[ref_view]
        A = torch.eye(4, device=self.dev, dtype=self.P0.dtype)
        A[:3, :3] = self.P0[row0, :3, :3].T
        A[:3, 3] = -self.P0[row0, :3, :3].T @ self.P0[row0, :3, 3]

        c2w_by_view = {}
        for row, img_idx in enumerate(self.im_order):
            c2w_by_view[img_idx] = (A @ self.P0[row]).clone()
        self.poses_c2w = [c2w_by_view[i] for i in range(self.num_views)]

    # -- Export ------------------------------------------------------------
    def export_global_poses(self):
        write_poses_txt(self.poses_global_path, self.poses_c2w, self.im_order, self.cfg.images)

    def build_and_save_pair_clouds(self, clouds_pairwise, ref_view: int = 0):
        clouds = []
        cloud_views = []
        for c_idx, (view_id, pred_full) in enumerate(clouds_pairwise):
            cl = cloud_from_pred(pred_full)
            if self.cfg.gaussian_subsample and cl["means"].shape[0] > self.cfg.gaussian_subsample:
                sel = torch.randperm(cl["means"].shape[0], device=self.dev)[: self.cfg.gaussian_subsample]
                for k in list(cl.keys()):
                    cl[k] = cl[k][sel]

            Mv = self.poses_c2w[view_id]
            R, t = Mv[:3, :3], Mv[:3, 3]
            transform_cloud_inplace(cl, R, t)

            out_npz = os.path.join(self.cfg.outdir, f"cloud_pair_{c_idx:02d}_view{view_id}.npz")
            np.savez_compressed(out_npz, **{k: v.detach().cpu().numpy() for k, v in cl.items()})
            if self.cfg.save_ply:
                save_as_ply(cl, out_npz.replace(".npz", ".ply"))
            print(f"saved {out_npz} ({cl['means'].shape[0]} pts)")

            clouds.append(cl)
            cloud_views.append(view_id)

        base_idx = cloud_views.index(ref_view)
        if base_idx != 0:
            clouds[0], clouds[base_idx] = clouds[base_idx], clouds[0]
            cloud_views[0], cloud_views[base_idx] = cloud_views[base_idx], cloud_views[0]
        return clouds, cloud_views

    def print_sim3_scales(self, clouds):
        base_np = clouds[0]["means"].detach().cpu().numpy()
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

    def refine_with_icp(self, clouds, cloud_views, ref_view: int = 0):
        base_np = clouds[0]["means"].detach().cpu().numpy()

        vox, th_c, th_f = 0.02, 0.08, self.cfg.icp_thresh
        b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
        b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
        b_coarse = b_full.voxel_down_sample(vox)
        b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

        icp_per_cloud = []
        merged = {k: [clouds[0][k]] for k in clouds[0].keys()}

        for c_idx in range(len(clouds)):
            view_id = cloud_views[c_idx]
            if view_id == ref_view:
                continue

            cl = clouds[c_idx]
            s_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cl["means"].detach().cpu().numpy()))
            s_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
            s_coarse = s_full.voxel_down_sample(vox)
            s_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

            icp1 = o3d.pipelines.registration.registration_icp(
                s_coarse, b_coarse, th_c, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
            )
            icp2 = o3d.pipelines.registration.registration_icp(
                s_full, b_full, th_f, icp1.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
            )
            T = icp2.transformation.copy()

            # Update pose
            M_i = self.poses_c2w[view_id]
            M_i = torch.from_numpy(T).to(device=self.dev, dtype=clouds[0]["means"].dtype) @ M_i
            self.poses_c2w[view_id] = M_i

            # Merge tensors (pre-transform), parity with original implementation
            for k in merged.keys():
                if k in cl:
                    merged[k].append(cl[k])

            ang = rotmat_to_angle_deg(T[:3, :3])
            tnorm = float(np.linalg.norm(T[:3, 3]))
            icp_per_cloud.append({
                "cloud_idx": int(c_idx),
                "view_id": int(view_id),
                "T_icp": T.copy(),
                "fitness": float(icp2.fitness),
                "rmse": float(icp2.inlier_rmse),
                "rot_deg": float(ang),
                "trans": float(tnorm),
            })
            print(f"[ICP view{view_id}] fit={icp2.fitness:.3f} rmse={icp2.inlier_rmse:.4f} dθ={ang:.2f}° dt={tnorm:.4f}")

            # Row for CSV
            row = {
                "cloud_idx": int(c_idx),
                "view_id": int(view_id),
                "icp_fine_fitness": float(icp2.fitness),
                "icp_fine_rmse": float(icp2.inlier_rmse),
                "icp_rot_deg": float(ang),
                "icp_trans": float(tnorm),
                "num_points_cloud": int(cl["means"].shape[0]),
                "num_points_base_before_merge": int(clouds[0]["means"].shape[0]),
            }
            self.metrics["per_view"].append(row)

            # Apply T to cl for subsequent ICP merges
            R_t = torch.from_numpy(T[:3, :3]).to(device=self.dev, dtype=cl["means"].dtype)
            t_t = torch.from_numpy(T[:3, 3]).to(device=self.dev, dtype=cl["means"].dtype)
            cl["means"] = (cl["means"] @ R_t.T) + t_t

            base_np = np.vstack([base_np, cl["means"].detach().cpu().numpy()])
            b_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(base_np))
            b_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
            b_coarse = b_full.voxel_down_sample(vox)
            b_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

        base = {k: torch.cat(v, dim=0) for k, v in merged.items()}
        return base, icp_per_cloud

    def export_best_and_icp_poses(self, icp_per_cloud):
        # Choose best ICP per view (fitness desc, rmse asc), else original global
        best_poses = [None] * self.num_views
        for view_id in range(self.num_views):
            cands = [c for c in icp_per_cloud if c["view_id"] == view_id]
            row_idx = self.view2row[view_id]
            P_glob_raw = self.P0[row_idx].detach().cpu().numpy()
            if cands:
                cand = sorted(cands, key=lambda x: (-x["fitness"], x["rmse"]))[0]
                T_icp = cand["T_icp"]
                best_poses[view_id] = T_icp @ P_glob_raw
            else:
                best_poses[view_id] = P_glob_raw

        # Write best poses
        with open(self.poses_best_path, "w") as f:
            for row, img_idx in enumerate(self.im_order):
                M = best_poses[img_idx]
                fname = os.path.basename(self.cfg.images[img_idx])
                f.write(f"# {fname}
")
                import numpy as _np
                _np.savetxt(f, M, fmt="%.8f")
                f.write("
")
        print("✓ wrote", self.poses_best_path)

        # Pairwise delta metrics pre vs post
        P_before = [P.to(self.dev) for P in self.P0]
        P_after = [self.poses_c2w[img_idx] for img_idx in self.im_order]
        for a in range(len(P_before)):
            for b in range(a + 1, len(P_before)):
                Rab = pose_inv(P_before[a]) @ P_before[b]
                Rhat = pose_inv(P_after[a]) @ P_after[b]
                Delta = torch.linalg.inv(Rhat) @ Rab
                R = Delta[:3, :3].detach().cpu().numpy()
                t = Delta[:3, 3].detach().cpu().numpy()
                self.metrics["pairwise_pose"].append({
                    "i": int(a), "j": int(b),
                    "rel_rot_deg": float(rotmat_to_angle_deg(R)),
                    "rel_trans": float(np.linalg.norm(t))
                })

        # Final ICP poses in im_order
        write_poses_txt(self.poses_icp_path, self.poses_c2w, self.im_order, self.cfg.images)

    def dedup_and_save_merged(self, base):
        if self.cfg.radius > 0:
            m = dedup_xyz(base["means"], self.cfg.radius)
            N = m.shape[0]

            if "sh" in base:
                sh = base["sh"]
                if sh.ndim == 2 and sh.shape[1] == 1 and sh.shape[0] % 3 == 0 and sh.shape[0] // 3 == N:
                    base["sh"] = sh.view(-1, 3)

            def mask_axes(t):
                ax = [a for a, s in enumerate(t.shape) if s == N]
                if not ax:
                    return t
                sl = [slice(None)] * t.ndim
                idx = torch.nonzero(m, as_tuple=False).squeeze(1)
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

        out = os.path.join(self.cfg.outdir, "merged_icp.npz")
        print("saving keys:", {k: tuple(v.shape) for k, v in base.items()})
        import numpy as _np
        _np.savez_compressed(out, **{k: v.detach().cpu().numpy() for k, v in base.items()})
        print(f"wrote {out} ({num_pts} unique pts)")

        if self.cfg.save_ply:
            save_as_ply(base, out.replace(".npz", ".ply"))
            print("wrote PLY")

    def run(self):
        self.setup_outdirs()
        model = self.load_model()
        self.load_and_prepare_images(size=512)

        d3, clouds_pairwise = self.build_pairwise(model)
        self.run_global_alignment(d3)
        self.compute_c2w(ref_view=0)
        self.export_global_poses()

        clouds, cloud_views = self.build_and_save_pair_clouds(clouds_pairwise, ref_view=0)
        self.print_sim3_scales(clouds)

        base, icp_per_cloud = self.refine_with_icp(clouds, cloud_views, ref_view=0)
        self.export_best_and_icp_poses(icp_per_cloud)

        self.dedup_and_save_merged(base)

        write_metrics_json(self.metrics_path, self.metrics)
        write_metrics_csv(self.csv_path, self.metrics["per_view"])

        print("Done")
