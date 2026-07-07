# unets/ — U-Net baseline + MC-Dropout uncertainty (for the hybrids)

Self-contained port of the spleen U-Net from `../../my_unet-uncertainty/` (nothing there is
modified). Provides the architecture, training, testing, and — most importantly — the
**communication layer** (`predictor.py`, `infer.py`) that hands the U-Net's segmentation +
per-voxel uncertainty to the traditional refiner, for the two hybrid ideas:
**(1) uncertainty-gated refinement** and **(2) refining a weakened baseline**.

- **Preprocessing is REUSED** from tradseg **Track A** (`preprocessing.py --track A` →
  `preprocessed_A/*.npy`), which is byte-identical to the U-Net's original contract. No
  separate preprocessing here.
- **Torch-only** (+ batchgenerators for training). GPU recommended. Weights + predictions land
  in `results/unets/`. Morph blocks from the original are ignored.

## Files
```
arch.py        MCDropoutUNet (deep MC dropout at 5 points; torch-only)
mc_dropout.py  enable_dropout / mc_forward / uncertainty_maps / fit_temperature
losses.py      SoftDiceLoss (+ helpers)          # + torch CE in the engine
data.py        batchgenerators loaders (fg-filtered train, all-slice val/test) + augmentation
engine.py      build_loaders + run_epoch (Dice+CE, bf16)     # train_frac -> weakened baseline
train.py       train one fold; SKIP if <tag>_f<fold>_best.pth exists (--force to retrain)
test.py        deterministic per-volume Dice (tradseg metrics)
predictor.py   UNetPredictor: predict() / predict_mc() on a (Z,H,W) volume  <-- the hybrid API
infer.py       MC inference over a fold -> per-case seg + uncertainty .npz (for the hybrid)
run_unet.sh    self-contained: clone -> deps -> data -> preprocess A -> train/test/infer per fold
```

## Run EVERYTHING (VM, GPU) — recommended
`run_all.sh` is the one-shot orchestrator: clone/refresh → data (skip if present) →
preprocess Track A (skip if present) → per fold train+test **3 nets** (skip-if-weights-exist)
→ run **both hybrids** (each with random walker + level set) → aggregate one comparison table.
```bash
curl -O https://raw.githubusercontent.com/ThanoSnake/CV_SemesterProject_TradMethods/main/unets/run_all.sh
nohup bash run_all.sh &          # tail -f ~/tradseg-run/run_all_*.log
# defaults: ONE fold, WEAK_FRAC=0.5, UNC=entropy. Knobs: FOLDS="0" EPOCHS=150 WEAK_FRAC=0.5 MC=30 UNC=entropy FORCE=1
# full 5-fold CV later:  FOLDS="0 1 2 3 4" nohup bash run_all.sh &
```
The three nets: `baseline` (pure, p=0), `mcdropout` (p=0.4), `weak` (pure, WEAK_FRAC of train).
Hybrid #1 = uncertainty-gated refinement of `mcdropout`; Hybrid #2 = morphological anchored
refinement of `weak`. Results: weights `results/unets/`, hybrids `results/hybrid/`, final
table `results/comparison/summary.{csv,json}`.

The hybrid engine itself lives at the repo root: **`hybrid.py`** (band + anchored refiners),
**`run_hybrid.py`** (per-fold runner), **`agg_compare.py`** (cross-fold table). Every refiner
may only relabel voxels *inside the band*; outside it the U-Net label is kept (anchoring).

## Run a single net (legacy per-tag primitive)
```bash
# full baseline (skip-if-exists), all folds:
nohup bash unets/run_unet.sh &
# weakened baseline (Idea 2): fewer TRAIN cases and/or fewer epochs, distinct tag
TAG=mcdropout_f25 FRAC=0.25 nohup bash unets/run_unet.sh &
TAG=mcdropout_ep20 EPOCHS=20 nohup bash unets/run_unet.sh &
```
Manual, one fold:
```bash
export DATA_DIR=$PWD/data/Task09_Spleen
python3 unets/train.py --tag mcdropout --fold 0 --preprocessed-dir $DATA_DIR/preprocessed_A --out-dir results/unets
python3 unets/test.py  --tag mcdropout --fold 0 --preprocessed-dir $DATA_DIR/preprocessed_A --weights-dir results/unets --advanced
python3 unets/infer.py --tag mcdropout --fold 0 --preprocessed-dir $DATA_DIR/preprocessed_A --weights-dir results/unets --mc-samples 30
```

## Communication API (used by the hybrid)
```python
import sys; sys.path.insert(0, "unets")
from predictor import UNetPredictor
pr = UNetPredictor("results/unets/mcdropout_f0_best.pth", num_classes=2, dropout_p=0.4)
det = pr.predict(image_vol)                 # {'seg', 'prob'}         (dropout OFF)
mc  = pr.predict_mc(image_vol, T=30)        # {'seg','mean_prob','entropy','mutual_info','fg_var'}
```
`infer.py` dumps the same MC maps as `results/unets/preds/<tag>_f<fold>_<case>.npz`
(`seg, fg_prob, entropy, mutual_info, fg_var`) for the traditional refiner to consume.
