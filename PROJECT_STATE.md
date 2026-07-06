# TradSeg — Project State & Handoff

> Read this file first. It fully describes the project's purpose, the fairness
> contract with the baseline, the code architecture (file by file), what is done,
> what is not, the implementation plan, and the concrete next steps. A new agent
> should be able to continue the work from this document alone.

---

## 1. TL;DR for a new agent

- **Goal:** implement *traditional* (non-neural) image-segmentation methods and compare
  them against an already-built baseline **U-Net** on a **medical** task from the
  **Medical Segmentation Decathlon (MSD)**, along three axes: **accuracy, time, memory**.
  Later (not now): *hybrid* methods combining the two schools.
- **Task chosen:** **MSD Task09 Spleen** (abdominal CT, single organ, ~41 labelled volumes,
  spleen = label 1). Work **2D per-slice first**, extend to 3D later.
- **Status:** **Tier 1 is fully implemented and verified on synthetic data** (unit +
  end-to-end integration). It has **NOT yet been run on the real MSD data.** The next
  concrete action is to run `run_all_tier1.sh` on the GPU/CPU VM and inspect real numbers.
- **This folder (`tradseg/`) IS the online git repo** (`github.com/ThanoSnake/CV_SemesterProject_TradMethods`,
  branch `main`). The code is **flat** (modules at the repo root) and is run as
  `python3 <script>.py` (absolute imports — **do not** use `python -m`). Keep every new
  deliverable inside this folder.
- **CPU-only.** No GPU is needed for the traditional methods. That is precisely the
  "space/time" story vs the GPU-hungry U-Net.

---

## 2. Purpose & academic context

- NTUA (ΕΜΠ) 8th-semester **Computer Vision / Όραση Υπολογιστών** course of **Prof. P. Maragos**;
  this is the semester project.
- The project deliberately leans on the course's toolbox: **mathematical morphology**,
  **nonlinear (lattice / (max,+) / morphological-PDE) algebra**, **level sets / active
  contours / variational methods**, and **texture** (Gabor / AM-FM energy operators, MRF,
  granulometries). Two textbook chapters are the source of ideas (see §14).
- The research angle is not "old methods lose to U-Net" (known), but the **trade-off**:
  traditional methods need **no training data, no GPU, little memory**, are **interpretable**,
  and give a **time-to-first-result** advantage. The story is the *gap vs the cost*.

---

## 3. Dataset & task

- **MSD Task09 Spleen**, downloaded as `Task09_Spleen.tar` (~1.5 GB) from
  `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar`.
- Contents: `imagesTr/` (41 CT volumes, NIfTI), `labelsTr/` (spleen = 1, background = 0),
  `imagesTs/` (unlabelled → not used; MSD test labels are hidden).
- Chosen because it is the standard "easy" MSD organ: single structure, decent contrast,
  small dataset → good for prototyping traditional methods. Tumour tasks (pancreas, lung,
  colon, liver-tumour, brain) are expected to *fail* with traditional methods and are left
  as future "negative-result" motivation for hybrids.
- **Physical difficulty to keep in mind:** on CT the spleen (~40–60 HU) is nearly
  **isointense** with liver, kidney, muscle. Pure-intensity methods therefore *cannot*
  separate it from neighbours and will "leak"; this is inherent (not a bug) and is the
  reason we add spatial priors, connected-component selection, and seeded methods.

---

## 4. The baseline to beat, and the FAIRNESS CONTRACT

A U-Net baseline **already exists** (built earlier, separate work) in the sibling folder
`../my_unet-uncertainty/` — a custom 2D **RecursiveUNet** (DKFZ/batchgenerators style)
plus MC-Dropout uncertainty variants (GitHub `ThanoSnake/my_unet`, branch
`uncertainty_spleen`). **We do not import or run it here** (that happens separately, on GPU).

We reproduce its **data + evaluation contract** so numbers are directly comparable:

- **Preprocessed volume format:** per case a `(2, Z, S, S)` `.npy`, channel 0 = CT image in
  `[0,1]`, channel 1 = integer label. `S = 256`.
- **Fixed CT window:** center 40, width 400 HU → `[-160, 240]` → `[0,1]` (air → 0). Fixed,
  not per-volume, so the intensity→tissue mapping is identical across cases.
- **Axial-first**, **label-free body crop** (drop air/table), **square-pad + resize 256**
  (image bilinear, label nearest).
- **Splits:** seeded (42) **5-fold CV**, each case in the test set of exactly one fold
  (`splits.pkl`). Our `create_splits` reproduces this algorithm exactly → identical folds.
- **Evaluation:** **per-volume**, in the preprocessed 256×256 space, **without voxel
  spacing** (so HD95/ASSD are in *resized-voxel units*, z treated isotropically), scored on
  the **foreground label only (spleen=1)**, averaged across cases with **nanmean**.
  Dice = 2TP/(2TP+FP+FN); NaN iff both pred & ref empty. HD95/ASSD = NaN if pred or ref is
  empty/full. **Dice/Jaccard are the honest headline; surface metrics are secondary.**
- **The U-Net's train-time foreground-slice curation does NOT transfer** to us (traditional
  methods have no training). Consequence: our automatic methods must fight **empty-slice
  false positives** via a presence prior + **3D largest-connected-component** post-processing.
  Always evaluate **per-volume, never per-slice** (per-slice Dice is mostly NaN).

**Two preprocessing tracks** were agreed (see §6) so the fair comparison is preserved while
still giving traditional methods a fair shot.

---

## 5. Repository & environment

- **The online repo == this `tradseg/` folder.** Its contents sit at the *repo root*
  (flat). Files outside it in the parent `TradSegMethods/` working dir
  (`../my_unet-uncertainty/`, the two lecture PDFs) are **context only, not in the repo.**
- **Flat layout + absolute imports.** Run modules directly: `python3 run_tier1.py …`,
  `python3 preprocessing.py …`. `config.PROJECT_ROOT = dir(config.py)` = repo root when flat.
  Do **not** use `python -m tradseg.X` (that would need a package wrapper we deliberately
  removed).
- **Execution env:** the user runs on a **GCP Deep Learning VM (L4 24 GB GPU)**, but the
  traditional methods are **CPU-bound** and need no GPU. Deps pin **`numpy<2`** (aligns with
  the baseline stack). Local dev machine is Windows + Anaconda (`C:\Users\feida\anaconda3\python.exe`).
- **Self-contained runner** `run_all_tier1.sh` clones the repo, installs deps, downloads the
  data once, preprocesses once, runs every experiment, and writes one folder per method.

---

## 6. Design decisions (locked)

1. **Dual-track preprocessing** (standalone, re-implemented; not the U-Net's code):
   - **Track A ("fair")** — identical to the U-Net contract (window c40/w400, resize 256).
     Use for the headline comparison.
   - **Track B ("trad")** — traditional-optimised: narrower soft-tissue window (**c50/w150**,
     more fat/organ contrast) + **median denoise**, still resized 256.
   - **Track Bnr** — Track B but **no resize + spacing sidecar** (`*.spacing.json`), for
     Tier-3 texture / mm-accurate surface metrics later. (Auto spatial prior needs a fixed
     in-plane size, so Bnr is not for the `auto` regime.)
2. **Two seeding/eval regimes:**
   - **`auto`** — fully automatic (fair vs the automatic U-Net). The spleen is localised via
     a **spatial prior built from the training fold**: an in-plane (H,W) location prior + a
     Gaussian intensity model of spleen HU. Uses training data only (like the U-Net) — never
     the test GT.
   - **`oracle`** — seeds / component derived from the case GT → **per-method upper bound**.
3. **Per-volume evaluation** exactly matching the U-Net (see §4).
4. **Results are stored one folder per method** (`results/<method>/…json`), not all mixed.
5. **Emphasis on morphology / nonlinear algebra** threads through: morphological
   post-processing, marker-controlled watershed (eikonal flavour), and the planned Tier-3
   granulometries / morphological level sets.

---

## 7. Architecture — file by file (all in `tradseg/`, repo root)

| File | Role |
|---|---|
| `config.py` | Paths + the data contract (env-overridable: `DATA_DIR`, `TASK`, window, size, `FOREGROUND_LABEL`). `spleen_target_intensity()` maps ~50 HU into the `[0,1]` window (~0.52). |
| `preprocessing.py` | Standalone raw-NIfTI → `(2,Z,S,S)` npy + `splits.pkl`. `PreprocConfig` + `TRACKS={A,B,Bnr}`. Steps: `normalize_ct` → `axial_axis` → `body_bbox` → optional `denoise` → optional `square_resize`. `create_splits` reproduces the baseline's seeded 5-fold. CLI: `--track`. |
| `io_utils.py` | `load_splits`, `fold_cases`, `load_case` (→ image `(Z,H,W)` float `[0,1]`, label int), `iter_cases`. |
| `metrics.py` | `dice/jaccard/precision/recall` (numpy) + `HD95/ASSD` (via `medpy`, lazy). `evaluate_case` (foreground label, per volume), `aggregate` (nanmean + std + n). NaN rules match the baseline. |
| `postprocess.py` | `keep_largest_cc`, `keep_k_largest_cc`, `remove_small_objects`, `fill_holes` (3D), `fill_holes_2d` (per slice), `binary_closing/opening`, `select_component_by_prior`, `select_component_by_overlap`, and a convenience `clean(...)`. |
| `seeding.py` | `SpatialPrior.from_training` (in-plane prior + intensity model); `.select` (prior component pick), `.auto_markers` (watershed fg/bg), `.auto_seed_points` (region-growing seeds). Oracle helpers: `oracle_markers`, `oracle_seed_points`. |
| `methods/base.py` | `Segmenter` ABC: `segment_volume(image, seeds=None) -> bool (Z,H,W)`; attrs `requires_seeds`, `seed_type` (`points`/`markers`). Helpers `apply_per_slice`, `body_mask`. **Segmenters must never read GT.** |
| `methods/thresholding.py` | `OtsuSegmenter`, `MultiOtsuSegmenter` (keeps the intensity band containing the spleen target). Weak by design. |
| `methods/clustering.py` | `KMeansSegmenter`, `GMMSegmenter` on `[0,1]` intensities; keep the cluster nearest the spleen intensity. |
| `methods/region_growing.py` | `RegionGrowingSegmenter` — `skimage.segmentation.flood` from seed points, intensity tolerance. `seed_type="points"`. |
| `methods/watershed.py` | `WatershedSegmenter` — marker-controlled watershed on gradient-magnitude relief, fg/bg markers. `seed_type="markers"`. |
| `methods/levelset.py` | **Tier 2** — `ChanVeseSegmenter` (MorphACWE, region) + `MorphGACSegmenter` (edge). fg marker = init level set, evolved **2D per slice**. |
| `methods/graphcut.py` | **Tier 2** — `GraphCutSegmenter` = GMM unaries + contrast-sensitive Potts pairwise + fg/bg hard seeds; PyMaxflow min-cut, 2D per slice (this IS the GMM+MRF/Potts method). |
| `methods/randomwalker.py` | **Tier 2** — `RandomWalkerSegmenter` (Grady): fg/bg markers, 2D per slice. |
| `methods/__init__.py` | `REGISTRY` = 6 Tier-1 + 4 Tier-2 (`chanvese, morphgac, graphcut, random_walker`). |
| `run_tier1.py` | Generic orchestrator CLI (Tier 1 **and** Tier 2): load fold → (auto) build `SpatialPrior` from train → per test case: build seeds (auto/oracle) → `segment_volume` → localise (prior/overlap component pick + `clean`) → `evaluate_case` → aggregate → write `results/<method>/<method>_<regime>_track<T>_f<fold>.json` with meta (`tier`, sec/case, peak RSS). The Segmenter `tier` attribute + `seed_type` drive naming/seeding generically, so new methods need no runner changes beyond a `build_method` branch. |
| `run_all_tier1.sh` | Self-contained runner (see §12); iterates all registered methods. |
| `requirements.txt` | `numpy<2, scipy, scikit-image, scikit-learn, PyMaxflow, medpy, nibabel, SimpleITK, pandas, matplotlib`. |
| `README.md` | User-facing usage. |
| `.gitignore` | Ignores `__pycache__/`, `data/`, `results/`, `*.npy`, `*.log`, etc. |
| `run_losses.sh` | A reference script from the U-Net project (not part of this pipeline; can be deleted). |

**Data/results layout at runtime** (git-ignored):
```
data/Task09_Spleen/{imagesTr,labelsTr, preprocessed_A/, preprocessed_B/, splits.pkl}
results/<method>/<method>_<regime>_track<T>_f<fold>.json
```

---

## 8. Status — what is DONE

- **Tier 1 fully implemented**: 6 methods (otsu, multiotsu, kmeans, gmm, region_growing,
  watershed) + preprocessing (A/B/Bnr) + seeding (auto prior + oracle) + per-volume metrics
  (Dice/Jaccard/HD95/ASSD) + 3D post-processing + orchestrator + self-contained runner.
- **Verified locally on synthetic data** (Anaconda Python), three levels:
  - unit tests of `metrics` (Dice/Jaccard/precision/recall, NaN rules, aggregate) and
    `postprocess` (largest-CC, small-object removal, prior/overlap select, hole fill);
  - a smoke test of the thresholding methods (shapes/dtypes, `scikit-image`/`sklearn` present);
  - an **end-to-end integration test**: synthetic `(2,Z,256,256)` cases with an isointense
    "distractor", run through the real `python3 run_tier1.py` CLI for all method×regime combos
    → all exit 0; the `auto` regime correctly selected the target over the distractor via the
    spatial prior. (Test files were ephemeral/scratch, not committed.)
- **Absolute-import refactor** done so the code runs flat (`python3 script.py`), matching the
  flat repo and the `run_losses.sh` style.
- `run_all_tier1.sh` `bash -n` syntax-checked.
- **Tier 2 implemented (2026-07-06):** `chanvese` (MorphACWE, region level set), `morphgac`
  (Morph-GAC, edge level set), `graphcut` (GMM unaries + contrast-Potts + PyMaxflow min-cut
  = **GMM+MRF/Potts**), `random_walker` (Grady) — all **2D per-slice**, seeded via fg/bg
  markers, registered in `REGISTRY`, runnable through the **same** `run_tier1.py`. Synthetic
  integration passed for all 4 (auto+oracle). **Not yet run on real data.** `chanvese` is
  init-sensitive (region-based, no edge stop) → expect variance; `morphgac`/`graphcut`/
  `random_walker` are more stable. **Results filenames dropped the `tier1_` prefix** → now
  `<method>_<regime>_track<T>_f<fold>.json` (the tier lives in the JSON `meta`).

---

## 9. Status — what is NOT done / risks / caveats

- **Never run on real MSD data.** The only path not exercisable locally is
  `preprocessing.py` on real NIfTI (no data/Linux locally). If anything breaks it will show
  early in the VM log at "preprocess track A". Watch `nibabel` loading / axial-axis / spacing.
- **Real-data numbers unknown.** Synthetic Dice ≈ 1.0 is NOT indicative. On real CT expect
  intensity methods (otsu/multiotsu/kmeans/gmm) to be weak and leak into touching isointense
  organs; watershed/region-growing depend heavily on seed quality; `oracle` gives the upper
  bound. See §10 for rough expectations.
- **Surface metrics not in mm** (no spacing) — consistent with the U-Net, so fair, but note it.
- **Tier-3 texture** needs the non-resized (`Bnr`) track for physically consistent scale;
  the `auto` prior does not apply there.
- No CSV aggregation yet (only per-run JSON + a pooled-Dice summary printed by the runner).

---

## 10. Implementation plan (tiers) & expectations

| Tier | Methods | Status |
|---|---|---|
| **1 — baselines** | Otsu, multi-Otsu, K-means, GMM, seeded region growing, marker-controlled watershed | **DONE** |
| **2 — core** | **Chan–Vese / Morphological ACWE** (region level set), **Geodesic Active Contours / Morph-GAC** (edge level set, Ch.17), **graph cuts** (MRF max-flow), **random walker** (Grady), **GMM + MRF/Potts** | **DONE** (2D per-slice; graphcut = GMM+MRF via PyMaxflow) |
| **3 — texture / morphology (Maragos flavour)** | **Gabor / AM-FM (Teager energy, DESA) features → curve evolution / clustering** (Ch.13), **granulometries / pattern spectrum**, **anisotropic diffusion (Perona–Malik)** preproc | TODO |
| **4 — hybrids** (later) | CNN prob-map + level-set/CRF refinement; traditional features as CNN channels; watershed/superpixels ↔ network seeds; differentiable morphology / active-contour loss; unrolled level sets | TODO |

**Rough Dice expectations, easy organ, `auto` regime** (real data): threshold/K-means
0.3–0.6; region-growing/watershed 0.6–0.8; (Tier-2 region level sets / graph cuts) 0.75–0.90;
`oracle` upper bounds higher (~0.9+); **U-Net ≈ 0.90–0.95**. Tumours (other tasks): traditional
often < 0.4.

Tier 2/3 plug into the **same `Segmenter` interface and runner** with no changes to the
harness (`run_tier1.py` handles seeded vs intensity methods and both regimes generically).

---

## 11. Immediate next steps (in order)

1. **Real run on the VM** (see §12). Push the repo, `curl` the runner, `nohup` it. Inspect
   the log for the preprocessing step and the pooled-Dice summary. Fix any real-NIfTI issues.
2. **Bring back the `results/<method>/*.json`** and sanity-check real numbers + failure modes
   (leakage, empty-slice FPs, seed quality). These guide Tier 2.
3. **(Optional) CSV/table aggregation** of all JSONs for the report (method×regime×track×fold →
   Dice/HD95/ASSD/time/RAM).
4. **Tier 2 implementation** — start with **Chan–Vese / Morphological ACWE** (region-based,
   robust on weak boundaries) and **graph cuts / random walker** (strong seeded methods), all
   as new `methods/*.py` registered in `REGISTRY`. Then GAC/Morph-GAC.
5. Later: Tier 3 (texture/morphology), then the **U-Net baseline run** (separate, GPU) for the
   head-to-head table, then Tier 4 hybrids.

---

## 12. How to run (commands)

**Self-contained (recommended), on the VM, in background:**
```bash
# after pushing this repo (branch main):
curl -O https://raw.githubusercontent.com/ThanoSnake/CV_SemesterProject_TradMethods/main/run_all_tier1.sh
nohup bash run_all_tier1.sh &          # progress: tail -f ~/tradseg-run/run_*.log
# env overrides: FOLDS="0"  TRACKS="A"  METHODS="multiotsu watershed"  ADVANCED=0  BRANCH=master
```
It clones→installs deps→downloads Spleen (once)→preprocesses A & B (once)→runs
`method × regime × track × fold`→`results/<method>/`→prints a pooled-Dice summary. Idempotent
(skip-if-exists). **Does not run the U-Net.**

**Manual (from the repo root, i.e. inside this folder on the VM):**
```bash
pip install -r requirements.txt
export DATA_DIR=$PWD/data/Task09_Spleen        # must contain imagesTr/ + labelsTr/
python3 preprocessing.py --track A
python3 preprocessing.py --track B
python3 run_tier1.py --method multiotsu --fold 0 --regime auto \
    --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced
```

---

## 13. How to extend (for the next LLM)

- **Add a method:** create `methods/<name>.py` with a `Segmenter` subclass implementing
  `segment_volume(image, seeds=None)`. If it needs seeds, set `requires_seeds=True` and
  `seed_type="points"|"markers"`. Register it in `methods/__init__.py:REGISTRY`. The runner
  then handles auto/oracle seeding and component selection automatically.
- **Keep the fairness contract:** operate on channel-0 `[0,1]` images, never read GT in the
  segmenter, evaluate per-volume with `metrics.evaluate_case`, keep the 256 space for the
  headline comparison.
- **Everything lives in this folder** (`tradseg/` == repo root). Do not scatter files into the
  parent `TradSegMethods/` (that is outside the repo and the user does not track it).
- **Run flat:** `python3 <script>.py`, absolute imports. If you add a subpackage, its internal
  imports may be relative (`from .x import`), but references to top-level modules must be
  absolute (`import config`), never `from .. import config`.
- **Verify** new code the same way: quick numpy/scipy unit checks + a synthetic
  `(2,Z,256,256)` integration run through `run_tier1.py` before claiming it works.

---

## 14. External references & memory

- **Idea sources** (in the parent dir, not in the repo): the two Maragos textbook chapters —
  `../book-cv05_13_chapter13_Texture.pdf` (Textons, Gabor/AM-FM energy operators & DESA,
  MRF/GRF, granulometries) and `../book-cv05_17_chapter17_ActivContourLevelsetVar.pdf`
  (snakes, curve evolution & level sets, geometric/geodesic active contours, morphological
  PDEs, watershed/eikonal, variational methods). Tier-2/3 draw directly from these.
- **Baseline** (parent dir, separate repo): `../my_unet-uncertainty/` — the U-Net + its
  preprocessing/splits/eval that define the fairness contract in §4.
- **Persistent agent memory** for this workspace lives under the Claude projects memory dir
  (auto-loaded each session via `MEMORY.md`): keys `project-tradsegmethods`,
  `unet-baseline-data-contract`, `env-python-anaconda`, `user-profile`. This document is the
  human/agent-readable superset.

---

*Last updated: 2026-07-06. Tier 1 + Tier 2 implemented and synthetic-verified; awaiting first real run on the VM.*
