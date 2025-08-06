#!/usr/bin/env bash
set -euo pipefail

# --- config -------------------------------------------------------------------
PY=python
GLOBAL=global.py
GSAM_PRE=scripts/preprocess_grounded_sam.py
RENDER=project_masks.py

DATA_ROOT=data/replica_v2
OUT_ROOT=data
SCENES=(room_0)

RUNS_PER_SCENE=5          # <—— set how many random runs you want per scene

RADIUS=0.003
ICP_THRESH=0.002
SAVE_PLY=--save-ply
WINDOW_START=405
# window size
WINDOW_SIZE=500

# Grounded-Segment-Anything repo/checkpoints
GSA_DIR=Grounded-Segment-Anything

# Folder name for GSAM npz files inside each run folder
GSAM_DIR_NAME=gsam_npz

# Renderer memory cap
MAX_POINTS=400000

# Intrinsics to drop into each run
INTRINSICS_LINE="1200 680 600 600 600 340 1.0"
# ------------------------------------------------------------------------------

pick_5_window() {
  local dir="$1"
  local start="${2:-}"

  mapfile -t files < <(ls -1 "${dir}"/rgb_*.png 2>/dev/null | sort -V)
  local N=${#files[@]}
  if (( N < 5 )); then
    echo "ERROR: Need at least 5 images in ${dir}, found ${N}" >&2
    return 1
  fi

  local -a idxs=()
  for f in "${files[@]}"; do
    idxs+=( "$(basename "${f%.*}" | awk -F_ '{print $2}')" )
  done

  # ---------- explicit window start ----------
  if [[ -n "$start" ]]; then
    if ! [[ "$start" =~ ^[0-9]+$ ]]; then
      echo "ERROR: WINDOW_START must be an integer, got '$start'" >&2
      return 1
    fi
    local low="$start"
    local high=$(( low + WINDOW_SIZE ))
    local -a subset_files=()
    local start_file=""

    for ((k=0; k<N; ++k)); do
      local id=${idxs[$k]}
      if (( id >= low && id <= high )); then
        subset_files+=( "${files[$k]}" )
      fi
      if (( id == low )); then
        start_file="${files[$k]}"
      fi
    done

    if ((${#subset_files[@]} >= 5)); then
      # Ensure the window start is one of the 5 frames and is FIRST
      if [[ -n "$start_file" ]]; then
        mapfile -t rest < <(printf "%s\n" "${subset_files[@]}" | grep -Fxv -- "$start_file" | shuf -n 4)
        mapfile -t rest_sorted < <(printf "%s\n" "${rest[@]}" | sort -V)
        mapfile -t sample < <(printf "%s\n" "$start_file" "${rest_sorted[@]}")
        echo "${sample[0]}|${sample[1]}|${sample[2]}|${sample[3]}|${sample[4]}"
      else
        echo "WARN: No exact frame with id=${low} found; including earliest in window instead." >&2
        local earliest="${subset_files[0]}"
        mapfile -t rest < <(printf "%s\n" "${subset_files[@]}" | grep -Fxv -- "$earliest" | shuf -n 4)
        mapfile -t rest_sorted < <(printf "%s\n" "${rest[@]}" | sort -V)
        mapfile -t sample < <(printf "%s\n" "$earliest" "${rest_sorted[@]}")
        echo "${sample[0]}|${sample[1]}|${sample[2]}|${sample[3]}|${sample[4]}"
      fi
      return 0
    else
      # Not enough images in the window; still include start frame if it exists
      if [[ -n "$start_file" ]]; then
        mapfile -t rest < <(printf "%s\n" "${files[@]}" | grep -Fxv -- "$start_file" | shuf -n 4)
        mapfile -t rest_sorted < <(printf "%s\n" "${rest[@]}" | sort -V)
        mapfile -t sample < <(printf "%s\n" "$start_file" "${rest_sorted[@]}")
        echo "${sample[0]}|${sample[1]}|${sample[2]}|${sample[3]}|${sample[4]}"
      else
        echo "WARN: WINDOW_START=${low} produced only ${#subset_files[@]} images in window size ${WINDOW_SIZE}; taking first 5." >&2
        echo "${files[0]}|${files[1]}|${files[2]}|${files[3]}|${files[4]}"
      fi
      return 0
    fi
  fi

  # ---------- random window start mode ----------
  for _ in {1..200}; do
    local r=$(( RANDOM % N ))
    local a=${idxs[$r]}
    local low=$a
    local high=$(( a + WINDOW_SIZE ))

    local -a subset_files=()
    local start_file="${files[$r]}"   # exact file for the chosen start index
    for ((k=0; k<N; ++k)); do
      local id=${idxs[$k]}
      if (( id >= low && id <= high )); then
        subset_files+=( "${files[$k]}" )
      fi
    done

    if ((${#subset_files[@]} >= 5)); then
      # Include the chosen start frame and make it FIRST
      mapfile -t rest < <(printf "%s\n" "${subset_files[@]}" | grep -Fxv -- "$start_file" | shuf -n 4)
      mapfile -t rest_sorted < <(printf "%s\n" "${rest[@]}" | sort -V)
      mapfile -t sample < <(printf "%s\n" "$start_file" "${rest_sorted[@]}")
      echo "${sample[0]}|${sample[1]}|${sample[2]}|${sample[3]}|${sample[4]}"
      return 0
    fi
  done

  echo "WARN: Could not find a <=${WINDOW_SIZE} range window with 5 images in ${dir}, taking first 5." >&2
  echo "${files[0]}|${files[1]}|${files[2]}|${files[3]}|${files[4]}"
}

for S in "${SCENES[@]}"; do
  RGB_DIR="${DATA_ROOT}/${S}/imap/00/rgb"
  if [[ ! -d "$RGB_DIR" ]]; then
    echo "WARN: ${RGB_DIR} does not exist, skipping ${S}"
    continue
  fi

  echo "==> Scene: ${S}"

  for RUN in $(seq 1 "${RUNS_PER_SCENE}"); do
    sel=$(pick_5_window "$RGB_DIR" "$WINDOW_START") || { echo "Skipping run ${RUN} for ${S}"; continue; }
    IFS='|' read -r i0 i1 i2 i3 i4 <<< "$sel"

    first_idx=$(basename "${i0%.*}" | awk -F_ '{print $2}')
    last_idx=$(basename "${i4%.*}" | awk -F_ '{print $2}')
    OUTDIR="${OUT_ROOT}/${S}_run${RUN}_${first_idx}_${last_idx}"
    mkdir -p "$OUTDIR"

    echo "  ├─ run ${RUN}"
    echo "  │   imgs:"
    printf "  │     %s\n  │     %s\n  │     %s\n  │     %s\n  │     %s\n" "$i0" "$i1" "$i2" "$i3" "$i4"
    echo "  │   out : ${OUTDIR}"

    # 1) Run global alignment / splatt3r
    $PY "$GLOBAL" \
      "$i0" "$i1" "$i2" "$i3" "$i4" \
      --outdir "$OUTDIR" \
      --radius "$RADIUS" \
      $SAVE_PLY \
      --icp-thresh "$ICP_THRESH"

    # 2) Write intrinsics.txt
    cat > "${OUTDIR}/intrinsics.txt" <<EOF
$INTRINSICS_LINE
# W  H   fx  fy  cx  cy  scale
EOF

    # 3) Grounded-SAM preprocessing
    GSAM_OUT="${OUTDIR}/${GSAM_DIR_NAME}"
    mkdir -p "$GSAM_OUT"
    $PY "$GSAM_PRE" \
      "${OUTDIR}/imgs" \
      "$GSAM_OUT" \
      "$GSA_DIR"

    # 4) Render per-image/per-mask diagnostics (+ composite colored-by-mask)
    $PY "$RENDER" \
      "$OUTDIR"
    echo "  │   ✓ Done run ${RUN}"
  done

  echo "==> Finished scene ${S}"
done
