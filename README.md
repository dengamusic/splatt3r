# Extending Splatt3R: Multi-View Semantic Gaussian Splatting - Pipeline Usage

This project provides a one-command Bash pipeline **and** per-script entry points.

## Called Scripts (in order)

1. `global.py` — global alignment / splatt3r over five RGB frames  
2. `scripts/preprocess_grounded_sam.py` — Grounded-SAM preprocessing (writes `.npz`)  
3. `project_masks.py` — semantic projection

---
## Setup Dependencies and Submodules
Follow documentations of [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [Splatt3r](https://github.com/btsmart/splatt3r)

## Download Replica Dataset
```bash
cd data
./download_replica.sh
```
## Run the whole pipeline

```bash
./run_pipeline.sh
```

---

## Run steps manually (example with filled-in args)

```bash
# Example selected frames (adjust paths if needed)
i0="data/replica_v2/room_0/imap/00/rgb/rgb_000405.png"
i1="data/replica_v2/room_0/imap/00/rgb/rgb_000436.png"
i2="data/replica_v2/room_0/imap/00/rgb/rgb_000512.png"
i3="data/replica_v2/room_0/imap/00/rgb/rgb_000688.png"
i4="data/replica_v2/room_0/imap/00/rgb/rgb_000904.png"

OUTDIR="data/room_0_run1_000405_000904"
mkdir -p "$OUTDIR"
```

### 1) Global alignment / splatt3r (`global.py`)

```bash
python global.py \
  "$i0" "$i1" "$i2" "$i3" "$i4" \
  --outdir "$OUTDIR" \
  --radius 0.003 \
  --save-ply \
  --icp-thresh 0.002
```

### 2) Write intrinsics file

```bash
cat > "$OUTDIR/intrinsics.txt" <<'EOF'
1200 680 600 600 600 340 1.0
# W  H   fx  fy  cx  cy  scale
EOF
```

### 3) Grounded-SAM preprocessing (`scripts/preprocess_grounded_sam.py`)

```bash
mkdir -p "$OUTDIR/gsam_npz"

python scripts/preprocess_grounded_sam.py \
  "$OUTDIR/imgs" \
  "$OUTDIR/gsam_npz" \
  "Grounded-Segment-Anything"
```

### 4) Project masks (`project_masks.py`)

```bash
python project_masks.py "$OUTDIR"
```

---

## Script signatures (for quick copy)

```bash
python global.py <img0> <img1> <img2> <img3> <img4> \
  --outdir <run_outdir> \
  --radius <float> \
  [--save-ply] \
  --icp-thresh <float>

python scripts/preprocess_grounded_sam.py <imgs_dir> <gsam_out_dir> <gsa_repo_dir>

python project_masks.py <run_outdir>
```

---

## Notes

- The five image paths must exist and follow `rgb_*.png`.  
- `global.py` writes `imgs/` into `<run_outdir>`, which is input for GSAM preprocessing.  
- Adjust `RADIUS`, `ICP_THRESH`, and `INTRINSICS_LINE` to match your dataset.
  
### Environment Variables Setup (try in case of issues)

Before running the pipeline or scripts, set the environment paths correctly:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$(dirname "$(pwd)"):$PYTHONPATH
```

- `LD_LIBRARY_PATH`: Prioritizes libraries from your active Conda environment.
- `PYTHONPATH`: Adds the parent directory of the current project to Python's module search paths.

