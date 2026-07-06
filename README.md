# TradSeg — Traditional (non-neural) segmentation on MSD Spleen

Classical segmentation methods (Tier 1) for MSD Task09 Spleen, scored **per-volume** in
the same 256×256 space and 5-fold CV as the baseline U-Net, so results are directly
comparable. CPU-only, no GPU needed.

The repo is **flat**: the modules live at the root and are launched directly as
`python3 <script>.py` (absolute imports — do NOT use `python -m`).

## Install
```bash
pip install -r requirements.txt
```

## One-shot self-contained run (recommended)
Downloads data, preprocesses, runs every method, writes one folder per method:
```bash
nohup bash run_all_tier1.sh &          # progress: tail -f ~/tradseg-run/run_*.log
```
Overrides: `FOLDS="0"`, `TRACKS="A"`, `METHODS="multiotsu watershed"`, `ADVANCED=0`, `BRANCH=master`.

## Manual run
```bash
export DATA_DIR=$PWD/data/Task09_Spleen      # imagesTr/ + labelsTr/ live here

# dual-track preprocessing -> (2,Z,S,S) npy + splits.pkl
python3 preprocessing.py --track A           # fair: identical to the U-Net (window c40/w400, resize 256)
python3 preprocessing.py --track B           # traditional-optimised: window c50/w150 + median denoise

# one method, one fold
python3 run_tier1.py --method multiotsu --fold 0 --regime auto \
    --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced
```

## Methods (`--method`)
| method | family | notes |
|---|---|---|
| `otsu`, `multiotsu` | intensity threshold | weak by design (soft-tissue isointensity) |
| `kmeans`, `gmm` | intensity clustering | keep cluster nearest spleen intensity |
| `region_growing` | seeded flood fill | seed = prior peak (auto) / deepest GT voxel (oracle) |
| `watershed` | marker-controlled | gradient relief + fg/bg markers |

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
methods/          otsu, multiotsu, kmeans, gmm, region_growing, watershed
run_tier1.py      run one method x fold x regime -> scores JSON
run_all_tier1.sh  self-contained clone->deps->download->preprocess->all experiments
```

## Notes
- Evaluation is per-volume, in the preprocessed 256×256 space, **without voxel spacing**
  (HD95/ASSD in resized-voxel units) — identical to the U-Net, hence fair. Dice/Jaccard
  are the headline; surface metrics are secondary.
- The `auto` spatial prior needs a fixed in-plane size → use Track A or B (resized).
