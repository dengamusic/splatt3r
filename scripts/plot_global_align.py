#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

NUMBER_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_loss_file(path: Path):
    """
    Parse a global_align_loss.txt file.
    Accepts either "iter loss" (two+ numbers per line) or "loss" (one number per line).
    Returns arrays (x: iterations, y: loss).
    """
    xs, ys = [], []
    idx = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        nums = [float(n) for n in NUMBER_RE.findall(line)]
        if not nums:
            continue
        if len(nums) >= 2:
            # Assume first number is iteration, last number is loss
            x = nums[0]
            y = nums[-1]
        else:
            # Only one number per line -> treat as loss; iteration is row index (1-based)
            idx += 1
            x = idx
            y = nums[0]
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

def moving_average(y: np.ndarray, window: int):
    if window is None or window <= 1:
        return None
    window = int(window)
    if window > len(y):
        return None
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="valid")

def main():
    ap = argparse.ArgumentParser(description="Plot all global_align_loss.txt in run folders under a root directory.")
    ap.add_argument("root", type=Path, help="Directory that contains run subfolders, each with a global_align_loss.txt file.")
    ap.add_argument("--pattern", default="global_align_loss.txt", help="Loss filename to look for in each run folder.")
    ap.add_argument("--recursive", action="store_true", help="Search recursively instead of only first-level subfolders.")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window N (e.g., 20). Use 1 to disable smoothing.")
    ap.add_argument("--every", type=int, default=1, help="Plot every Nth point (decimation). Use 1 to plot all points.")
    ap.add_argument("--first", type=int, default=None, help="Plot only the first N iterations (per run).")
    ap.add_argument("--outfile", type=Path, default=None, help="Where to save the figure (PNG). Default: <root>/global_align_loss_summary.png")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving.")
    ap.add_argument("--ylog", action="store_true", help="Use logarithmic y-scale for the loss.")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    if args.recursive:
        files = sorted(root.rglob(args.pattern))
    else:
        files = sorted((d / args.pattern for d in root.iterdir() if d.is_dir() and (d / args.pattern).exists()))

    if not files:
        raise SystemExit(f"No '{args.pattern}' files found under: {root}")

    if args.every < 1:
        raise SystemExit("--every must be >= 1")
    if args.smooth < 1:
        raise SystemExit("--smooth must be >= 1")

    plt.figure(figsize=(10, 6))
    n_plotted = 0
    for f in files:
        run_name = f.parent.name  # use run folder name for curve label
        x, y = parse_loss_file(f)
        if x.size == 0:
            print(f"Skipping empty file: {f}")
            continue

        # Limit to the first N iterations, if requested
        if args.first is not None:
            # Keep entries with iteration <= first N
            mask = x <= float(args.first)
            x = x[mask]
            y = y[mask]

        if x.size == 0:
            print(f"Skipping (no data within --first limit): {f}")
            continue

        # Optional decimation (plot every Nth point)
        if args.every > 1:
            x = x[::args.every]
            y = y[::args.every]

        # Optional smoothing
        if args.smooth > 1:
            y_s = moving_average(y, args.smooth)
            if y_s is not None:
                x = x[args.smooth - 1 :]
                y = y_s

        plt.plot(x, y, label=run_name)
        n_plotted += 1

    if n_plotted == 0:
        raise SystemExit("Nothing to plot.")

    plt.xlabel("iteration")
    plt.ylabel("global aligner loss")
    plt.title("Global aligner loss per run")
    if args.ylog:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    out = args.outfile or (root / "global_align_loss_summary.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
