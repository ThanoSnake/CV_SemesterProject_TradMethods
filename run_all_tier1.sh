#!/usr/bin/env bash
#
# Self-contained CPU-only experiment on MSD Task09 Spleen with the TRADITIONAL
# (Tier-1) segmentation methods. Same spirit as run_losses.sh, but no GPU / no U-Net:
#
# It does EVERYTHING itself: git clone (or update) -> deps -> download Spleen (once) ->
# dual-track preprocessing (once) -> run ALL methods x regimes x tracks x folds ->
# results in a SEPARATE folder per method under <WORKDIR>/repo/results/<method>/ ->
# print a pooled-Dice summary. The baseline U-Net is DELIBERATELY NOT run here.
#
# The repo layout is FLAT (config.py, run_tier1.py, preprocessing.py, methods/ ... at
# the repo root), so scripts are launched directly as `python3 <script>.py`.
#
# You do NOT clone anything by hand -- this script clones for you. Put it on the VM and
# launch it so it survives an SSH disconnect:
#   nohup bash run_all_tier1.sh &        # progress: tail -f ~/tradseg-run/run_*.log
# Copy results off afterwards:
#   gcloud compute scp --recurse <user>@<vm>:~/tradseg-run/repo/results ./
#
set -uo pipefail   # -u: error on unset vars; pipefail through tee. NOT -e: one failing
                   # experiment must not throw away the runs that already finished.

# ============================ CONFIG (edit these) ============================
REPO_URL="${REPO_URL:-https://github.com/ThanoSnake/CV_SemesterProject_TradMethods.git}"
BRANCH="${BRANCH:-main}"                                # set BRANCH=master if that is your default
WORKDIR="${WORKDIR:-$HOME/tradseg-run}"                 # where the repo + logs live
TASK="${TASK:-Task09_Spleen}"

METHODS="${METHODS:-otsu multiotsu kmeans gmm region_growing watershed}"
REGIMES="${REGIMES:-auto oracle}"                       # auto = fully automatic; oracle = GT-seeded upper bound
TRACKS="${TRACKS:-A B}"                                 # A = identical to U-Net (fair); B = trad-optimised
FOLDS="${FOLDS:-0 1 2 3 4}"                             # full 5-fold CV; FOLDS="0" for a quick pass
ADVANCED="${ADVANCED:-1}"                               # 1 -> also HD95 / ASSD (needs medpy)
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1                 # log EVERYTHING (incl. clone) to console AND file

echo "################ TradSeg Tier-1 (CPU, no U-Net)  $(date '+%F %T') ################"
echo "repo=$REPO_URL  branch=$BRANCH  workdir=$WORKDIR  task=$TASK"
echo "methods='$METHODS' regimes='$REGIMES' tracks='$TRACKS' folds='$FOLDS' advanced=$ADVANCED"
echo "log -> $LOG"

run() {   # run() "label" cmd... : header + timing, CONTINUE on failure (background-safe)
    local label="$1"; shift
    echo ""; echo "===== [$(date '+%F %T')] $label ====="
    local t0=$SECONDS
    "$@"; local rc=$?
    echo "----- $label done in $((SECONDS - t0))s (exit $rc) -----"
    [ $rc -eq 0 ] || echo "!!! FAILED: $label (continuing) !!!"
    return 0
}

# ---- 1. clone (or update) the repo ----
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo present -> force-updating to origin/$BRANCH (tracked code only; data/ & results/ untouched)"
    git -C "$REPO_DIR" fetch origin "$BRANCH" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD \
        || echo "WARN: could not update; using existing checkout"
else
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone of branch '$BRANCH' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR" || { echo "cannot cd $REPO_DIR"; exit 1; }
echo "on branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)  |  cwd: $(pwd)"

# sanity: the flat code must be at the repo root
[ -f "run_tier1.py" ] || { echo "ERROR: run_tier1.py not at repo root (flat layout expected)."; exit 1; }
# the methods/ package marker may be .gitignored -> recreate; touch never truncates an existing file
mkdir -p methods; touch methods/__init__.py

# ---- 2. dependencies (CPU-only; no torch needed) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
python3 -m pip install --break-system-packages -q -r requirements.txt 2>/dev/null \
    || python3 -m pip install -q -r requirements.txt \
    || echo "WARN: pip install returned non-zero; continuing (deps may already be present)"
python3 - <<'PYCHK' || { echo "FATAL: core deps failed to import."; exit 1; }
import numpy, scipy, skimage, sklearn, medpy, nibabel, SimpleITK
print("deps OK | numpy", numpy.__version__, "| skimage", skimage.__version__)
PYCHK

# ---- 3. data: download + extract once ----
export TASK
export DATA_DIR="$REPO_DIR/data/$TASK"
if [ ! -d "$DATA_DIR/imagesTr" ]; then
    echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
    mkdir -p data
    ( cd data && curl -L -O "https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar" \
        && tar -xf "${TASK}.tar" ) \
        || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
fi

# ---- 4. dual-track preprocessing, once per track ----
for track in $TRACKS; do
    prep="$DATA_DIR/preprocessed_$track"
    if [ -f "$DATA_DIR/splits.pkl" ] && ls "$prep"/*.npy >/dev/null 2>&1; then
        echo "skip preprocess track $track (already present)"
    else
        run "preprocess track $track" python3 preprocessing.py --track "$track"
    fi
done

# ---- 5. experiments: method x regime x track x fold; ONE folder per method ----
#         skip-if-exists makes re-runs idempotent.
ADV=""; [ "$ADVANCED" = "1" ] && ADV="--advanced"
for method in $METHODS; do
    mkdir -p "results/$method"
    for track in $TRACKS; do
        prep="$DATA_DIR/preprocessed_$track"
        ls "$prep"/*.npy >/dev/null 2>&1 || { echo "skip $method track $track (no preprocessed data)"; continue; }
        for regime in $REGIMES; do
            for fold in $FOLDS; do
                stem="tier1_${method}_${regime}_track${track}_f${fold}"
                if [ -f "results/$method/$stem.json" ]; then
                    echo "skip $stem (exists)"; continue
                fi
                run "$stem" python3 run_tier1.py --method "$method" --fold "$fold" \
                    --regime "$regime" --track "$track" --preprocessed-dir "$prep" \
                    --out-dir "results/$method" $ADV
            done
        done
    done
done

# ---- 6. summary: pooled per-case Dice per method/regime/track across folds ----
echo ""; echo "===== [$(date '+%F %T')] summary ====="
python3 - <<'PYSUM' || echo "WARN: summary failed"
import os, json, glob
from statistics import mean
rows = {}
for jf in glob.glob(os.path.join("results", "*", "*.json")):
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

cp "$LOG" results/ 2>/dev/null || true
echo ""; echo "################ ALL DONE  $(date '+%F %T') ################"
echo "results (one folder per method): $REPO_DIR/results/<method>/"
ls -1 results/ 2>/dev/null | sed 's/^/  /'
