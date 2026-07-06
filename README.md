# TradSeg — Traditional (non-neural) segmentation on MSD Spleen

Classical segmentation methods (Tier 1 & 2) for MSD Task09 Spleen, scored **per-volume** in
the same 256×256 space and 5-fold CV as the baseline U-Net, so results are directly
comparable. CPU-only, no GPU needed.

The repo is **flat**: the modules live at the root and are launched directly as
`python3 <script>.py` (absolute imports — do NOT use `python -m`).

## Install
```bash
pip install -r requirements.txt
```

## Self-contained run, one script per tier (recommended)
Each `run_tier<N>.sh` does everything itself and the same way — clone/update repo → deps →
download data (once) → dual-track preprocess (once) → run that tier's methods → results in
`results/tier<N>/<method>/`. Run them one at a time (they share the repo checkout + data):
```bash
nohup bash run_tier1.sh &     # Tier 1 baselines   (progress: tail -f ~/tradseg-run/run_tier*_*.log)
nohup bash run_tier2.sh &     # Tier 2 level sets / graph cut / random walker
nohup bash run_tier3.sh &     # Tier 3 (reports "nothing to run" until its methods are implemented)
```
Overrides: `FOLDS="0"`, `TRACKS="A"`, `METHODS="graphcut random_walker"`, `ADVANCED=0`, `BRANCH=master`.

## Manual run
```bash
export DATA_DIR=$PWD/data/Task09_Spleen      # imagesTr/ + labelsTr/ live here

# dual-track preprocessing -> (2,Z,S,S) npy + splits.pkl
python3 preprocessing.py --track A           # fair: identical to the U-Net (window c40/w400, resize 256)
python3 preprocessing.py --track B           # traditional-optimised: window c50/w150 + median denoise

# one method, one fold
python3 run_experiment.py --method multiotsu --fold 0 --regime auto \
    --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced
```

## Methods (`--method`)
| method | family | notes |
|---|---|---|
| `otsu`, `multiotsu` | intensity threshold | weak by design (soft-tissue isointensity) |
| `kmeans`, `gmm` | intensity clustering | keep cluster nearest spleen intensity |
| `region_growing` | seeded flood fill | seed = prior peak (auto) / deepest GT voxel (oracle) |
| `watershed` | marker-controlled | gradient relief + fg/bg markers |
| `chanvese` | region level set (MorphACWE) | fg marker = init contour; may leak (no edge stop) |
| `morphgac` | edge level set (Morph-GAC) | balloons from init to gradient edges |
| `graphcut` | GMM + MRF/Potts, min-cut | GMM unaries + contrast-Potts + fg/bg hard seeds (needs PyMaxflow) |
| `random_walker` | graph diffusion (Grady) | fg/bg markers |

**Regimes.** `auto` = fully automatic (fair vs the automatic U-Net); the spleen is
localised via a probabilistic location + intensity prior built from the **training**
fold. `oracle` = seeds / component from the case GT → per-method upper bound.

## Files
```
config.py         paths + data contract (env-overridable)
preprocessing.py  raw NIfTI -> (2,Z,S,S) npy + splits (Track A / B / Bnr)
io_utils.py       load npy + splits
metrics.py        Dice/Jaccard/HD95/ASSD, per-volume, baseline-matching NaN rules
postprocess.py    3D largest-CC, small-object removal, hole fill, prior/overlap select
seeding.py        SpatialPrior (auto) + oracle markers / seed points
methods/          Tier 1: otsu, multiotsu, kmeans, gmm, region_growing, watershed
                  Tier 2: chanvese, morphgac, graphcut, random_walker
run_experiment.py  run ONE method x fold x regime -> scores JSON (generic; used by run_tier*.sh)
list_methods.py    print a tier's registered method names (used by run_tier*.sh)
run_tier{1,2,3}.sh self-contained per-tier driver: clone->deps->download->preprocess->experiments
```

## Notes
- Evaluation is per-volume, in the preprocessed 256×256 space, **without voxel spacing**
  (HD95/ASSD in resized-voxel units) — identical to the U-Net, hence fair. Dice/Jaccard
  are the headline; surface metrics are secondary.
- The `auto` spatial prior needs a fixed in-plane size → use Track A or B (resized).
