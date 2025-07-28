#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helpers
# -------------------------------
def find_csvs(root: Path, pattern: str = "merged_groups.csv"):
    return sorted(root.rglob(pattern))

def infer_run_name(csv_path: Path):
    # Try to use the parent of a "vis" folder as the run name: .../<RUN>/vis/merged_groups.csv
    parts = csv_path.parts
    if "vis" in parts:
        i = parts.index("vis")
        if i - 1 >= 0:
            return parts[i - 1]
    # Fallback: parent directory name
    return csv_path.parent.name

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_and_tag(csv_path: Path, run_name: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return None

    # Normalize column names (strip spaces, lowercase where relevant)
    df.columns = [c.strip() for c in df.columns]

    # Required columns: either score, or enough to derive it
    have_score = "score" in df.columns
    if not have_score:
        if {"iou_mean", "clip_mean"}.issubset(df.columns):
            df["score"] = 0.5 * (pd.to_numeric(df["iou_mean"], errors="coerce") +
                                 pd.to_numeric(df["clip_mean"], errors="coerce"))
        else:
            print(f"[WARN] {csv_path} missing score and iou_mean/clip_mean; skipping.", file=sys.stderr)
            return None

    # Coerce useful numerics
    df = coerce_numeric(df, ["score","iou_mean","clip_mean","union_size","views","n_members"])

    # Minimal required row filter: score must be finite
    df = df[np.isfinite(df["score"])]

    # Tag the run
    df.insert(0, "run", run_name)
    # Add a source column for traceability
    df.insert(1, "source", str(csv_path.relative_to(csv_path.parents[0]) if csv_path.is_absolute() else csv_path))

    return df

def make_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# -------------------------------
# Plotting (matplotlib)
# -------------------------------
def plot_bar_clusters_per_run(summary, out_path: Path):
    fig = plt.figure(figsize=(8, 4.5))
    x = np.arange(len(summary))
    y = summary["clusters"].to_numpy()
    plt.bar(x, y)
    plt.xticks(x, summary["run"], rotation=45, ha="right")
    plt.ylabel("Number of clusters")
    plt.title("Clusters per run")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_box_scores_per_run(combined, out_path: Path):
    runs = sorted(combined["run"].unique())
    data = [combined.loc[combined["run"] == r, "score"].dropna().to_numpy() for r in runs]
    fig = plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=runs, showfliers=False)
    plt.ylabel("Score")
    plt.title("Score distribution per run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_scatter_iou_vs_clip(combined, out_path: Path):
    if not {"iou_mean","clip_mean"}.issubset(combined.columns):
        return  # skip if not available
    runs = sorted(combined["run"].unique())
    markers = ["o","s","^","D","v","P","X","<",">","*","h","H","+","x","1","2","3","4"]
    fig = plt.figure(figsize=(7.5, 6))
    for i, r in enumerate(runs):
        sub = combined.loc[combined["run"] == r]
        if sub.empty:
            continue
        m = markers[i % len(markers)]
        plt.scatter(sub["iou_mean"], sub["clip_mean"], label=r, marker=m, alpha=0.75)
    plt.xlabel("iou_mean")
    plt.ylabel("clip_mean")
    plt.title("iou_mean vs clip_mean by run")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect merged_groups.csv from subdirectories and plot combined stats.")
    ap.add_argument("root", nargs="?", default=".", help="Root folder to scan recursively (default: current dir).")
    ap.add_argument("--pattern", default="merged_groups.csv", help="Filename pattern to search for (default: merged_groups.csv).")
    ap.add_argument("--out", default="analysis_out", help="Output directory for tables and plots.")
    ap.add_argument("--min_views", type=int, default=None, help="If set, keep only clusters with views >= this.")
    ap.add_argument("--topk", type=int, default=10, help="Top-K clusters by score per run to save.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = make_outdir(Path(args.out))

    csv_paths = find_csvs(root, args.pattern)
    if not csv_paths:
        print(f"[ERROR] No '{args.pattern}' found under {root}")
        sys.exit(1)

    rows = []
    for p in csv_paths:
        run = infer_run_name(p)
        df = load_and_tag(p, run)
        if df is not None:
            rows.append(df)

    if not rows:
        print("[ERROR] No usable CSVs after loading; abort.")
        sys.exit(2)

    combined = pd.concat(rows, ignore_index=True)

    # Optional filter by min_views
    if args.min_views is not None and "views" in combined.columns:
        combined = combined[combined["views"].fillna(0).astype(int) >= int(args.min_views)]

    if combined.empty:
        print("[ERROR] Combined dataframe is empty after filtering.")
        sys.exit(3)

    # Save combined table
    combined_csv = outdir / "combined_merged_groups.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"wrote {combined_csv}")

    # Summary by run
    agg = {
        "score": ["count", "mean", "median", "std"],
    }
    if "iou_mean" in combined.columns:
        agg["iou_mean"] = ["mean"]
    if "clip_mean" in combined.columns:
        agg["clip_mean"] = ["mean"]
    if "union_size" in combined.columns:
        agg["union_size"] = ["mean"]
    if "views" in combined.columns:
        # we'll compute extra counts separately
        agg["views"] = ["mean"]

    summary = combined.groupby("run").agg(agg)
    # Flatten columns
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
    summary = summary.rename(columns={"score_count": "clusters"})
    summary = summary.reset_index()

    # Extra: count of clusters with views >= 2 (if present)
    if "views" in combined.columns:
        v2 = (combined["views"] >= 2).groupby(combined["run"]).sum().rename("clusters_views_ge2")
        summary = summary.merge(v2.reset_index(), on="run", how="left")

    summary_csv = outdir / "summary_by_run.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")

    # Top-K clusters by score per run
    top_rows = []
    for r, sub in combined.groupby("run"):
        top = sub.sort_values("score", ascending=False).head(args.topk)
        top_rows.append(top.assign(rank=range(1, len(top) + 1)))
    top_df = pd.concat(top_rows, ignore_index=True)
    top_csv = outdir / "top_clusters_by_run.csv"
    top_df.to_csv(top_csv, index=False)
    print(f"wrote {top_csv}")

    # Plots
    plot_bar_clusters_per_run(summary[["run","clusters"]], outdir / "clusters_per_run_bar.png")
    print(f"wrote {outdir / 'clusters_per_run_bar.png'}")

    plot_box_scores_per_run(combined, outdir / "score_boxplot_by_run.png")
    print(f"wrote {outdir / 'score_boxplot_by_run.png'}")

    plot_scatter_iou_vs_clip(combined, outdir / "iou_vs_clip_scatter.png")
    print(f"wrote {outdir / 'iou_vs_clip_scatter.png'}")

    print("Done.")

if __name__ == "__main__":
    main()
