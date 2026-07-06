#!/usr/bin/env bash
#
# Self-contained CPU-only runner for the TRADITIONAL (Tier-1) segmentation methods on
# MSD Task09 Spleen. Does everything itself:
#   clone/update repo -> locate code (works whether it sits at repo ROOT or under
#   tradseg/) -> isolated venv + deps -> download Spleen (once) -> dual-track
#   preprocessing (once) -> run ALL methods x regimes x tracks x folds -> results in a
#   SEPARATE folder per method -> pooled-Dice summary.
#
# The baseline U-Net is DELIBERATELY NOT run here. No GPU needed.
#
# Bootstrap on the VM (survives an SSH disconnect) -- use the raw URL of THIS file in
# your repo (adjust the path if it is not under run/):
#   curl -O https://raw.githubusercontent.com/ThanoSnake/CV_SemesterProject_TradMethods/main/run/run_all_tier1.sh
#   nohup bash run_all_tier1.sh &        # progress: tail -f ~/tradseg-run/run_*.log
# Copy results off afterwards, e.g.:
#   gcloud compute scp --recurse <user>@<vm>:~/tradseg-run/results ./
#
# Handy overrides (env):  FOLDS="0"  TRACKS="A"  METHODS="multiotsu watershed"  ADVANCED=0  BRANCH=master
#
set -uo pipefail   # NOT -e: one failing experiment must not discard the finished ones.

# ============================ CONFIG (override via env) =====================
REPO_URL="${REPO_URL:-https://github.com/ThanoSnake/CV_SemesterProject_TradMethods.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/tradseg-run}"
TASK="${TASK:-Task09_Spleen}"
DATA_TAR_URL="${DATA_TAR_URL:-https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar}"

METHODS="${METHODS:-otsu multiotsu kmeans gmm region_growing watershed}"
REGIMES="${REGIMES:-auto oracle}"
TRACKS="${TRACKS:-A B}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ADVANCED="${ADVANCED:-1}"
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############### TradSeg Tier-1 (CPU, no U-Net)  $(date '+%F %T') ###############"
echo "repo=$REPO_URL branch=$BRANCH workdir=$WORKDIR task=$TASK"
echo "methods='$METHODS' regimes='$REGIMES' tracks='$TRACKS' folds='$FOLDS' advanced=$ADVANCED"

run() {  # run "label" cmd... : header + timing, CONTINUE on failure
  local label="$1"; shift
  echo ""; echo "===== [$(date '+%F %T')] $label ====="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "----- $label done in $((SECONDS - t0))s (exit $rc) -----"
  [ $rc -eq 0 ] || echo "!!! FAILED: $label (continuing) !!!"
  return 0
}

# ---- 1. clone (first time) or update the repo ----
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
  echo "repo present -> updating to origin/$BRANCH"
  git -C "$REPO_DIR" fetch origin "$BRANCH" \
    && git -C "$REPO_DIR" checkout "$BRANCH" \
    && git -C "$REPO_DIR" reset --hard FETCH_HEAD \
    || echo "WARN: could not update; using existing checkout"
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
    || { echo "git clone of '$BRANCH' failed -> aborting"; exit 1; }
fi

# ---- 1b. locate the code and make it importable as the package 'tradseg' ----
# Works whether the repo keeps the code under tradseg/ OR flat at the repo root.
if [ -f "$REPO_DIR/tradseg/run_tier1.py" ]; then
  PKG_PARENT="$REPO_DIR"; CODE_DIR="$REPO_DIR/tradseg"
  echo "layout: code under tradseg/ subfolder"
elif [ -f "$REPO_DIR/run_tier1.py" ]; then
  ln -sfn "$REPO_DIR" "$WORKDIR/tradseg"          # expose the flat repo as package 'tradseg'
  PKG_PARENT="$WORKDIR"; CODE_DIR="$WORKDIR/tradseg"
  echo "layout: flat repo -> symlinked as $WORKDIR/tradseg"
else
  echo "ERROR: run_tier1.py not found in repo root or repo/tradseg/. Aborting."
  exit 1
fi
cd "$PKG_PARENT"
touch "$CODE_DIR/__init__.py" "$CODE_DIR/methods/__init__.py" 2>/dev/null || true
REQ="$CODE_DIR/requirements.txt"
[ -f "$REQ" ] || { echo "ERROR: requirements.txt not found at $REQ"; exit 1; }
echo "package parent: $PKG_PARENT | code: $CODE_DIR"

# ---- 2. isolated venv + deps (falls back to pip --user) ----
VENV="$WORKDIR/venv"
if python3 -m venv "$VENV" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"; PY=python; PIPU=""
  echo "using venv: $VENV"
else
  echo "WARN: venv unavailable -> system python3 + pip --user"; PY=python3; PIPU="--user"
fi
run "pip upgrade"              $PY -m pip install $PIPU -q --upgrade pip
run "pip install requirements" $PY -m pip install $PIPU -q -r "$REQ"
$PY - <<'PYCHK' || { echo "FATAL: core deps failed to import."; exit 1; }
import numpy, scipy, skimage, sklearn, medpy, nibabel, SimpleITK
print("deps OK | numpy", numpy.__version__, "| skimage", skimage.__version__)
PYCHK

# ---- 3. data: download + extract once ----
export TASK
export DATA_DIR="$PKG_PARENT/data/$TASK"
if [ ! -d "$DATA_DIR/imagesTr" ]; then
  echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
  mkdir -p "$PKG_PARENT/data"
  ( cd "$PKG_PARENT/data" && curl -L -O "$DATA_TAR_URL" && tar -xf "${TASK}.tar" ) \
    || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
fi

# ---- 4. dual-track preprocessing, once per track ----
for track in $TRACKS; do
  prep="$DATA_DIR/preprocessed_$track"
  if [ -f "$DATA_DIR/splits.pkl" ] && ls "$prep"/*.npy >/dev/null 2>&1; then
    echo "skip preprocess track $track (already present)"
  else
    run "preprocess track $track" $PY -m tradseg.preprocessing --track "$track"
  fi
done

# ---- 5. experiments: method x regime x track x fold; ONE folder per method ----
RESULTS="$PKG_PARENT/results"
ADV=""; [ "$ADVANCED" = "1" ] && ADV="--advanced"
for method in $METHODS; do
  out="$RESULTS/$method"; mkdir -p "$out"
  for track in $TRACKS; do
    prep="$DATA_DIR/preprocessed_$track"
    ls "$prep"/*.npy >/dev/null 2>&1 || { echo "skip $method track $track (no data)"; continue; }
    for regime in $REGIMES; do
      for fold in $FOLDS; do
        stem="tier1_${method}_${regime}_track${track}_f${fold}"
        if [ -f "$out/$stem.json" ]; then echo "skip $stem (exists)"; continue; fi
        run "$stem" $PY -m tradseg.run_tier1 --method "$method" --fold "$fold" \
            --regime "$regime" --track "$track" --preprocessed-dir "$prep" \
            --out-dir "$out" $ADV
      done
    done
  done
done

# ---- 6. summary: pooled per-case Dice per method/regime/track across folds ----
echo ""; echo "===== [$(date '+%F %T')] summary ====="
RESULTS="$RESULTS" $PY - <<'PYSUM' || echo "WARN: summary failed"
import os, json, glob
from statistics import mean
root = os.environ["RESULTS"]
rows = {}
for jf in glob.glob(os.path.join(root, "*", "*.json")):
    try:
        d = json.load(open(jf))
    except Exception:
        continue
    m = d.get("meta", {})
    key = (m.get("method"), m.get("regime"), m.get("track"))
    vals = [c["Dice"] for c in d.get("per_case", {}).values()
            if c.get("Dice") is not None and c["Dice"] == c["Dice"]]
    rows.setdefault(key, []).extend(vals)
print(f"\n{'method':16s}{'regime':8s}{'track':6s}{'meanDice':>9}{'n':>5}")
for key in sorted(k for k in rows if all(x is not None for x in k)):
    v = rows[key]
    print(f"{key[0]:16s}{key[1]:8s}{key[2]:6s}{(mean(v) if v else float('nan')):>9.4f}{len(v):>5d}")
PYSUM

echo ""; echo "############### ALL DONE  $(date '+%F %T') ###############"
echo "results (one folder per method): $RESULTS/<method>/"
ls -1 "$RESULTS" 2>/dev/null | sed 's/^/  /'
cp "$LOG" "$RESULTS/" 2>/dev/null || true
