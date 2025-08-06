#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Generic helpers
# -------------------------------
def make_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def find_csvs(root: Path, pattern: str):
    return sorted(root.rglob(pattern))

def infer_run_name(csv_path: Path):
    parts = csv_path.parts
    if "vis" in parts:
        i = parts.index("vis")
        if i - 1 >= 0:
            return parts[i - 1]
    return csv_path.parent.name

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

MARKERS = ["o","s","^","D","v","P","X","<",">","*","h","H","+","x","1","2","3","4"]

# -------------------------------
# Loaders for each csv type
# -------------------------------
def load_merged(csv_path: Path, run_name: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None
    df.columns = [c.strip() for c in df.columns]
    if "score" not in df.columns:
        if {"iou_mean","clip_mean"}.issubset(df.columns):
            df["score"] = 0.5 * (pd.to_numeric(df["iou_mean"], errors="coerce") +
                                 pd.to_numeric(df["clip_mean"], errors="coerce"))
        else:
            return None
    df = coerce_numeric(df, ["score","iou_mean","clip_mean","union_size","views","n_members"])
    df = df[np.isfinite(df["score"])]
    df.insert(0, "run", run_name); df.insert(1, "source", str(csv_path))
    return df

def load_view_metrics(csv_path: Path, run_name: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None
    df.columns = [c.strip() for c in df.columns]
    df = coerce_numeric(df, ["view_id","gauss_visible","gauss_assigned","view_coverage"])
    if {"gauss_visible","gauss_assigned"}.issubset(df.columns):
        df["gauss_unassigned"] = df["gauss_visible"] - df["gauss_assigned"]
    df.insert(0, "run", run_name); df.insert(1, "source", str(csv_path))
    return df

def load_mask_metrics(csv_path: Path, run_name: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = [
        "view_id","mask_id","gauss_in_mask","gauss_visible_view",
        "mask_hit_rate","mask_pixels","mask_hit_pixels","mask_pixel_coverage"
    ]
    df = coerce_numeric(df, numeric_cols)
    if "label" not in df.columns:
        df["label"] = df["mask_id"].apply(lambda x: f"mask{int(x)}")
    df.insert(0, "run", run_name); df.insert(1, "source", str(csv_path))
    return df

# -------------------------------
# Summaries / Plots we keep (same as before)
# -------------------------------
def summarize_merged(combined: pd.DataFrame):
    agg = {"score": ["count","mean","median","std"]}
    if "iou_mean" in combined.columns: agg["iou_mean"] = ["mean"]
    if "clip_mean" in combined.columns: agg["clip_mean"] = ["mean"]
    if "union_size" in combined.columns: agg["union_size"] = ["mean"]
    if "views" in combined.columns: agg["views"] = ["mean"]
    summary = combined.groupby("run").agg(agg)
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
    summary = summary.rename(columns={"score_count": "clusters"}).reset_index()
    return summary

def plot_merged(summary, combined, outdir: Path):
    # Bar: clusters per run
    fig = plt.figure(figsize=(8,4.5))
    x = np.arange(len(summary)); y = summary["clusters"].to_numpy()
    plt.bar(x, y); plt.xticks(x, summary["run"], rotation=45, ha="right")
    plt.ylabel("# clusters"); plt.title("Clusters per run"); plt.tight_layout()
    fig.savefig(outdir / "clusters_per_run_bar.png", dpi=200); plt.close(fig)

    # Box: score distribution per run
    runs = sorted(combined["run"].unique())
    data = [combined.loc[combined["run"] == r, "score"].dropna().to_numpy() for r in runs]
    fig = plt.figure(figsize=(9,5)); plt.boxplot(data, labels=runs, showfliers=False)
    plt.ylabel("Score"); plt.title("Score distribution per run")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(outdir / "score_boxplot_by_run.png", dpi=200); plt.close(fig)

    # Scatter: iou_mean vs clip_mean (if present)
    if {"iou_mean","clip_mean"}.issubset(combined.columns):
        fig = plt.figure(figsize=(7.5,6))
        for i, r in enumerate(runs):
            sub = combined.loc[combined["run"] == r]
            if sub.empty: continue
            plt.scatter(sub["iou_mean"], sub["clip_mean"], label=r,
                        marker=MARKERS[i % len(MARKERS)], alpha=0.75)
        plt.xlabel("iou_mean"); plt.ylabel("clip_mean"); plt.title("IoU vs CLIP by run")
        plt.legend(fontsize=8); plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout(); fig.savefig(outdir / "iou_vs_clip_scatter.png", dpi=200); plt.close(fig)

def summarize_view(combined: pd.DataFrame):
    agg = { "view_id": ["count"], "view_coverage": ["mean","median","std"] }
    if "gauss_visible" in combined.columns:  agg["gauss_visible"]  = ["mean"]
    if "gauss_assigned" in combined.columns: agg["gauss_assigned"] = ["mean"]
    if "gauss_unassigned" in combined.columns: agg["gauss_unassigned"] = ["mean"]
    summary = combined.groupby("run").agg(agg)
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
    summary = summary.rename(columns={"view_id_count":"views"}).reset_index()
    return summary

def plot_view(summary, combined, outdir: Path):
    runs = sorted(combined["run"].unique())
    # Box: view_coverage per run
    data = [combined.loc[combined["run"] == r, "view_coverage"].dropna().to_numpy() for r in runs]
    fig = plt.figure(figsize=(9,5)); plt.boxplot(data, labels=runs, showfliers=False)
    plt.ylabel("view_coverage"); plt.title("View coverage per run")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    fig.savefig(outdir / "view_coverage_boxplot_by_run.png", dpi=200); plt.close(fig)
    # Bar: mean view_coverage per run
    fig = plt.figure(figsize=(8,4.5))
    x = np.arange(len(summary)); y = summary["view_coverage_mean"].to_numpy()
    plt.bar(x, y); plt.xticks(x, summary["run"], rotation=45, ha="right")
    plt.ylabel("Mean view_coverage"); plt.title("Mean view coverage per run"); plt.tight_layout()
    fig.savefig(outdir / "view_coverage_mean_bar.png", dpi=200); plt.close(fig)

# -------------------------------
# NEW: Label-level table from mask_metrics
# -------------------------------
def label_summary(mask_combined: pd.DataFrame):
    # keep only rows that have the needed fields
    need = {"label","gauss_in_mask","gauss_visible_view","mask_hit_rate","mask_pixels","mask_hit_pixels","view_id"}
    if not need.issubset(mask_combined.columns):
        missing = need - set(mask_combined.columns)
        raise ValueError(f"mask_metrics is missing columns: {sorted(missing)}")

    # percentages in [0..100]
    def _agg(group: pd.DataFrame):
        sum_vis  = group["gauss_visible_view"].sum()
        sum_in   = group["gauss_in_mask"].sum()
        sum_px   = group["mask_pixels"].sum()
        sum_hitp = group["mask_hit_pixels"].sum()

        hit_rate_w = 100.0 * (sum_in / sum_vis) if sum_vis > 0 else np.nan
        pix_cov_w  = 100.0 * (sum_hitp / sum_px) if sum_px > 0 else np.nan

        return pd.Series({
            "masks": group.shape[0],
            "views": group["view_id"].nunique(),
            "hit_rate_pct_w": hit_rate_w,
            "hit_rate_pct_mean": 100.0 * group["mask_hit_rate"].mean(),
            "pixel_cov_pct_w": pix_cov_w,
            "pixel_cov_pct_mean": 100.0 * group["mask_pixel_coverage"].mean(),
        })

    by_run_label = mask_combined.groupby(["run","label"], sort=False).apply(_agg).reset_index()

    # also aggregated across runs
    by_label = mask_combined.groupby(["label"], sort=False).apply(_agg).reset_index()

    return by_run_label, by_label

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Collect merged/view/mask CSVs from subdirectories; build summaries and label tables."
    )
    ap.add_argument("root", nargs="?", default=".", help="Root folder to scan recursively (default: .)")
    ap.add_argument("--out", default="analysis_out", help="Output directory for tables and plots")
    ap.add_argument("--merged_pattern", default="merged_groups.csv", help="Filename to search for merged clusters")
    ap.add_argument("--view_pattern", default="view_metrics.csv", help="Filename to search for view metrics")
    ap.add_argument("--mask_pattern", default="mask_metrics.csv", help="Filename to search for mask metrics")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = make_outdir(Path(args.out))

    # ---------- MERGED ----------
    merged_paths = find_csvs(root, args.merged_pattern)
    merged_rows = []
    for p in merged_paths:
        run = infer_run_name(p)
        df = load_merged(p, run)
        if df is not None: merged_rows.append(df)
    if merged_rows:
        merged_combined = pd.concat(merged_rows, ignore_index=True)
        merged_combined.to_csv(outdir / "combined_merged_groups.csv", index=False)
        merged_summary = summarize_merged(merged_combined)
        merged_summary.to_csv(outdir / "summary_by_run_merged.csv", index=False)
        plot_merged(merged_summary, merged_combined, outdir)

    # ---------- VIEW ----------
    view_paths = find_csvs(root, args.view_pattern)
    view_rows = []
    for p in view_paths:
        run = infer_run_name(p)
        df = load_view_metrics(p, run)
        if df is not None: view_rows.append(df)
    if view_rows:
        view_combined = pd.concat(view_rows, ignore_index=True)
        view_combined.to_csv(outdir / "combined_view_metrics.csv", index=False)
        view_summary = summarize_view(view_combined)
        view_summary.to_csv(outdir / "summary_by_run_view.csv", index=False)
        # keep only the more meaningful plots
        plot_view(view_summary, view_combined, outdir)

    # ---------- MASK & LABEL TABLE ----------
    mask_paths = find_csvs(root, args.mask_pattern)
    mask_rows = []
    for p in mask_paths:
        run = infer_run_name(p)
        df = load_mask_metrics(p, run)
        if df is not None: mask_rows.append(df)
    if mask_rows:
        mask_combined = pd.concat(mask_rows, ignore_index=True)
        mask_combined.to_csv(outdir / "combined_mask_metrics.csv", index=False)

        # Label-level percentages
        by_run_label, by_label = label_summary(mask_combined)
        by_run_label.sort_values(["run","hit_rate_pct_w","pixel_cov_pct_w"], ascending=[True,False,False], inplace=True)
        by_label.sort_values(["hit_rate_pct_w","pixel_cov_pct_w"], ascending=[False,False], inplace=True)

        by_run_label.to_csv(outdir / "label_summary_by_run.csv", index=False)
        by_label.to_csv(outdir / "label_summary_overall.csv", index=False)

        print(f"[label] wrote label_summary_by_run.csv and label_summary_overall.csv to {outdir}")

    print("Done.")

if __name__ == "__main__":
    main()
