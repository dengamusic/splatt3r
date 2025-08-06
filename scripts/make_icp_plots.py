import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# -------------------- argparse --------------------
parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path, help="Path to metrics_per_view.csv")
args = parser.parse_args()

CSV_PATH = args.csv.resolve()
OUTDIR = CSV_PATH.parent / CSV_PATH.stem
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------- config --------------------
DPI      = 300
FONTSIZE = 16
plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE * 0.9,
    "ytick.labelsize": FONTSIZE * 0.9,
    "legend.fontsize": FONTSIZE * 0.9,
})

# -------------------- load data --------------------
df = pd.read_csv(CSV_PATH)
if "view_id" in df.columns:
    df = df.sort_values("view_id")

# -------------------- helpers --------------------
def savefig(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)

def summarize(col):
    return {
        "mean": float(np.mean(col)),
        "std": float(np.std(col, ddof=1)),
        "median": float(np.median(col)),
        "p90": float(np.percentile(col, 90)),
        "min": float(np.min(col)),
        "max": float(np.max(col)),
    }

# -------------------- plots --------------------
views = df["view_id"] if "view_id" in df.columns else np.arange(len(df))

# A. Bar plots: rotation / translation
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axes[0].bar(views, df["icp_rot_deg"])
axes[0].set_ylabel("Rotation Δθ (deg)")
axes[0].set_title("ICP correction per view")
axes[1].bar(views, df["icp_trans"])
axes[1].set_ylabel("Translation ‖Δt‖")
axes[1].set_xlabel("View ID")
savefig(fig, "Fig-A_rotation_translation")

# B. Line plots: fitness / RMSE
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axes[0].plot(views, df["icp_fine_fitness"], "o-")
axes[0].set_ylabel("Fine fitness")
axes[0].set_title("ICP fine-stage metrics")
axes[1].plot(views, df["icp_fine_rmse"], "o-")
axes[1].set_ylabel("Fine RMSE")
axes[1].set_xlabel("View ID")
savefig(fig, "Fig-B_fitness_rmse")

# C. Scatter: rot vs trans
fig, ax = plt.subplots(figsize=(5.5, 5.0))
ax.scatter(df["icp_rot_deg"], df["icp_trans"])
for v, x, y in zip(views, df["icp_rot_deg"], df["icp_trans"]):
    ax.annotate(str(v), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=FONTSIZE * 0.75)
ax.set_xlabel("Rotation Δθ (deg)")
ax.set_ylabel("Translation ‖Δt‖")
ax.set_title("ICP correction magnitude")
savefig(fig, "Fig-C_rot_vs_trans_scatter")

# D. Optional histogram: log Sim(3) scale
if "log_sim3_scale" in df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["log_sim3_scale"], bins=10)
    ax.set_xlabel("log Sim(3) scale vs. base")
    ax.set_ylabel("#views")
    ax.set_title("Per-view Sim(3) scale drift")
    savefig(fig, "Fig-D_log_sim3_scale_hist")

# -------------------- stats summary --------------------
summary = {
    "icp_rot_deg": summarize(df["icp_rot_deg"]),
    "icp_trans": summarize(df["icp_trans"]),
    "icp_fine_fitness": summarize(df["icp_fine_fitness"]),
    "icp_fine_rmse": summarize(df["icp_fine_rmse"]),
}
if "log_sim3_scale" in df.columns:
    summary["log_sim3_scale"] = summarize(df["log_sim3_scale"])

txt_lines = []
for k, v in summary.items():
    line = (f"{k}: mean={v['mean']:.4f} ± {v['std']:.4f} | "
            f"median={v['median']:.4f} | p90={v['p90']:.4f} | "
            f"min={v['min']:.4f} | max={v['max']:.4f}")
    txt_lines.append(line)

stats_path = OUTDIR / "stats_summary.txt"
stats_path.write_text("\n".join(txt_lines))
print("✓ Wrote:", OUTDIR.resolve())
print(stats_path.read_text())
