#!/usr/bin/env python3
"""
Multi-view Splatt3r demo: CLI entrypoint.

- Keeps the same flags/behavior as your original script
- Uses package code in src/multiview_splatt3r
- Also appends external project paths for dust3r/pixelsplat as before
"""
import argparse
import os
import sys

# Resolve project directories relative to this script
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "src")

# Ensure our package ("multiview_splatt3r") can be imported
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Preserve external search paths (relative to ./src)
# These are kept for parity with your original project layout.
_ext_paths = [
    os.path.join(_SRC_DIR, "mast3r_src"),
    os.path.join(_SRC_DIR, "mast3r_src", "dust3r"),
    os.path.join(_SRC_DIR, "pixelsplat_src"),
]
for p in _ext_paths:
    if p not in sys.path:
        sys.path.append(p)

from multiview_splatt3r import Splatt3rPipeline, Config

def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs=5, help="Five image paths")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--radius", type=float, default=0.004, help="Dedup radius (in scene units)")
    ap.add_argument("--model-dir", default=None, help="Local dir for HF model cache")
    ap.add_argument("--align-iters", type=int, default=1000, help="Global align iterations")
    ap.add_argument("--align-lr", type=float, default=0.005, help="Global align learning rate")
    ap.add_argument("--gaussian-subsample", type=int, help="Random subsample per cloud (gaussians)")
    ap.add_argument("--icp-thresh", type=float, default=0.005, help="ICP fine threshold (meters)")
    ap.add_argument("--save-ply", action="store_true", help="Also write PLY point clouds")
    ap.add_argument("--no-sym", action="store_true", help="(Unused) kept for CLI parity")
    args = ap.parse_args()

    return Config(
        images=args.images,
        outdir=args.outdir,
        radius=args.radius,
        model_dir=args.model_dir,
        align_iters=args.align_iters,
        align_lr=args.align_lr,
        gaussian_subsample=args.gaussian_subsample,
        icp_thresh=args.icp_thresh,
        save_ply=args.save_ply,
        no_sym=args.no_sym,
    )

def main():
    cfg = parse_args()
    pipeline = Splatt3rPipeline(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()
