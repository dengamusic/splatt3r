# plot_icp_metrics.py
from pathlib import Path
import csv
import math

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _read_metrics(csv_path: Path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract and sort by view_id when present
    def as_int(v, fallback):
        try:
            return int(float(v))
        except Exception:
            return fallback

    indexed = []
    for i, r in enumerate(rows):
        vid = as_int(r.get("view_id", i), i)
        indexed.append((vid, r))
    indexed.sort(key=lambda t: t[0])

    views = []
    rot_deg = []
    trans = []
    fit = []
    rmse = []

    for vid, r in indexed:
        views.append(vid)
        rot_deg.append(_safe_float(r.get("icp_rot_deg")))
        trans.append(_safe_float(r.get("icp_trans")))
        fit.append(_safe_float(r.get("icp_fine_fitness")))
        rmse.append(_safe_float(r.get("icp_fine_rmse")))

    # Filter out Nones consistently
    def compact(xs, ys):
        out_x, out_y = [], []
        for x, y in zip(xs, ys):
            if y is not None:
                out_x.append(x); out_y.append(y)
        return out_x, out_y

    v_rot, rot_deg = compact(views, rot_deg)
    v_trn, trans   = compact(views, trans)
    v_fit, fit     = compact(views, fit)
    v_rms, rmse    = compact(views, rmse)

    return {
        "views_rot": v_rot, "rot_deg": rot_deg,
        "views_trn": v_trn, "trans": trans,
        "views_fit": v_fit, "fit": fit,
        "views_rms": v_rms, "rmse": rmse,
    }

def save_icp_plots(csv_path, outdir=None, dpi=300, fontsize=14):
    """
    Read metrics_per_view.csv and save PNG plots next to it.
    - csv_path: path to metrics_per_view.csv
    - outdir: optional output dir; default = csv_dir / csv_stem
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("[plot_icp_metrics] matplotlib not installed; skip plotting.")
        return

    csv_path = Path(csv_path).resolve()
    outdir = Path(outdir+"/vis") if outdir else (csv_path.parent / csv_path.stem)
    outdir.mkdir(parents=True, exist_ok=True)

    data = _read_metrics(csv_path)

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "legend.fontsize": fontsize * 0.9,
    })

    # ---- Fig A: rotation / translation (bars) ----
    if data["rot_deg"] or data["trans"]:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        if data["rot_deg"]:
            axes[0].bar(data["views_rot"], data["rot_deg"])
            axes[0].set_ylabel("Rotation Δθ (deg)")
            axes[0].set_title("ICP correction per view")
            axes[0].grid(True, axis="y", alpha=0.3)
        if data["trans"]:
            axes[1].bar(data["views_trn"], data["trans"])
            axes[1].set_ylabel("Translation ‖Δt‖")
            axes[1].set_xlabel("View ID")
            axes[1].grid(True, axis="y", alpha=0.3)
        fig.savefig(outdir / "icp_rotation_translation.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    # ---- Fig B: fitness / RMSE (lines) ----
    if data["fit"] or data["rmse"]:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        if data["fit"]:
            axes[0].plot(data["views_fit"], data["fit"], "o-")
            axes[0].set_ylabel("Fine fitness")
            axes[0].set_title("ICP fine-stage metrics")
            axes[0].grid(True, axis="both", alpha=0.3)
        if data["rmse"]:
            axes[1].plot(data["views_rms"], data["rmse"], "o-")
            axes[1].set_ylabel("Fine RMSE")
            axes[1].set_xlabel("View ID")
            axes[1].grid(True, axis="both", alpha=0.3)
        fig.savefig(outdir / "icp_fitness_rmse.png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    # ---- Fig C: rotation vs translation (scatter) ----
    if data["rot_deg"] and data["trans"]:
        # align lists by view id that exists in both
        common_views = sorted(set(data["views_rot"]).intersection(set(data["views_trn"])))
        v_to_rot = {v: r for v, r in zip(data["views_rot"], data["rot_deg"])}
        v_to_trn = {v: t for v, t in zip(data["views_trn"], data["trans"])}
        xs, ys, labs = [], [], []
        for v in common_views:
            r = v_to_rot.get(v)
            t = v_to_trn.get(v)
            if r is not None and t is not None:
                xs.append(r); ys.append(t); labs.append(v)
        if xs:
            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            ax.scatter(xs, ys)
            # Annotate with view id
            for v, x, y in zip(labs, xs, ys):
                ax.annotate(str(v), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=fontsize*0.75)
            ax.set_xlabel("Rotation Δθ (deg)")
            ax.set_ylabel("Translation ‖Δt‖")
            ax.set_title("ICP correction magnitude")
            ax.grid(True, alpha=0.3)
            fig.savefig(outdir / "icp_rot_vs_trans_scatter.png", bbox_inches="tight", dpi=dpi)
            plt.close(fig)

    print(f"[plot_icp_metrics] wrote PNGs to {outdir}")
