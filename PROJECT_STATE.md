# TradSeg — Project State & Handoff

> Read this file first. It fully describes the project's purpose, the fairness contract with
> the baseline, the code architecture, what is done, the REAL-DATA results & findings so far,
> what is pending, and the concrete next steps. A new agent should be able to continue from
> this document alone. (Updated only on explicit request, not every change.)

---

## 1. TL;DR for a new agent

- **Goal:** implement *traditional* (non-neural) segmentation methods and compare them to an
  already-built **U-Net** baseline on **MSD Task09 Spleen** (abdominal CT, ~41 labelled
  volumes, spleen = label 1), along three axes: **accuracy, time, memory**. Later: *hybrids*.
- **Status:** **Tiers 1, 2, 3 are ALL implemented** and have been **run on the real MSD data
  on the VM**. Tier 1, Tier 2, and Tier-3 `gabor` results have been evaluated (see §9). A
  **seeding overhaul** (adaptive auto-markers) has just been implemented to fix the dominant
  auto-regime failure; it has **NOT yet been re-run on real data**.
- **Headline finding:** traditional **seeded** methods (watershed, morphgac, random_walker)
  reach **~0.88–0.92 Dice given good seeds (`oracle`)** — competitive with U-Net (~0.90–0.95),
  at **CPU / seconds / <1.1 GB**. The whole gap is **automatic localization** (`auto` ≪ `oracle`),
  and its dominant cause was an **empty-marker bug** now fixed by the new seeding.
- **This folder (`tradseg/`) IS the online git repo** (`github.com/ThanoSnake/CV_SemesterProject_TradMethods`,
  branch `main`). Code is **flat** (modules at repo root), run as `python3 <script>.py`
  (absolute imports — **not** `python -m`). Keep every new deliverable inside this folder.
- **CPU-only.** No GPU needed for the traditional methods — that is the "space/time" story.

---

## 2. Purpose & academic context

- NTUA (ΕΜΠ) 8th-semester **Computer Vision** course of **Prof. P. Maragos**; semester project.
- Leans on the course toolbox: **mathematical morphology**, **nonlinear (lattice / (max,+) /
  morphological-PDE) algebra**, **level sets / active contours / variational methods**, and
  **texture** (Gabor / AM-FM energy operators, MRF, granulometries). Sources: two textbook
  chapters (§15).
- The angle is the **trade-off**: traditional methods need **no training data, no GPU, little
  memory**, are **interpretable**, and give **time-to-first-result**. The story is *gap vs cost*.

---

## 3. Dataset & task

- **MSD Task09 Spleen** — `Task09_Spleen.tar` (~1.5 GB) from
  `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar`.
- `imagesTr/` (41 CT NIfTI), `labelsTr/` (spleen=1, bg=0), `imagesTs/` (unlabelled → unused).
- Standard "easy" MSD organ (single structure, small dataset). Tumour tasks are expected to
  fail with traditional methods (future negative-result motivation for hybrids).
- **Physical difficulty:** on CT the spleen (~40–60 HU) is nearly **isointense** with liver,
  kidney, muscle → pure-intensity methods cannot separate it (they "predict everything"). This
  is inherent, not a bug; it motivates spatial priors + seeded methods.

---

## 4. The baseline to beat, and the FAIRNESS CONTRACT

A U-Net baseline **already exists** in the sibling folder `../my_unet-uncertainty/` — a custom
2D **RecursiveUNet** + MC-Dropout uncertainty (GitHub `ThanoSnake/my_unet`, branch
`uncertainty_spleen`). **We do not import or run it here** (separate, GPU). We reproduce its
**data + evaluation contract** for direct comparability:

- **Preprocessed volume:** per case `(2, Z, S, S)` npy — ch0 = CT in `[0,1]`, ch1 = int label. S=256.
- **Fixed CT window** center 40 / width 400 HU → `[-160,240]` → `[0,1]` (air→0). Fixed, not per-volume.
- **Axial-first**, **label-free body crop**, **square-pad + resize 256** (image bilinear, label nearest).
- **Splits:** seeded (42) **5-fold CV** (`splits.pkl`); `create_splits` reproduces the baseline's exactly.
- **Evaluation:** **per-volume**, in the 256×256 space, **without voxel spacing** (HD95/ASSD in
  resized-voxel units, z isotropic), foreground label only, averaged with **nanmean**.
  Dice=2TP/(2TP+FP+FN); NaN iff both empty. HD95/ASSD NaN if pred/ref empty or full.
  **Dice/Jaccard are the honest headline; surface metrics secondary.**
- **Fairness rule:** the `auto` regime uses ONLY (training prior + training intensity model +
  test image), **never the test GT** → same "GT-blind" class as U-Net. The `oracle` regime uses
  the test GT → **upper bound / diagnostic only**, never reported as the fair number.

---

## 5. Repository & environment

- **The online repo == this `tradseg/` folder** (the user's local git is here; only its contents
  are pushed). Files outside it in the parent `TradSegMethods/` (`../my_unet-uncertainty/`, the
  two lecture PDFs) are **context only, not in the repo.**
- **Flat layout + absolute imports.** Run modules directly (`python3 run_experiment.py …`,
  `python3 preprocessing.py …`). `config.PROJECT_ROOT = dir(config.py)` = repo root when flat.
  Do **not** use `python -m`.
- **Execution env:** user runs on a **GCP Deep Learning VM (L4 24 GB GPU)**, but the traditional
  methods are **CPU-bound** (no GPU). Deps pin **`numpy<2`**. Local dev = Windows + Anaconda
  (`C:\Users\feida\anaconda3\python.exe`), used only for synthetic verification.
- **Per-tier self-contained runners** `run_tier1.sh` / `run_tier2.sh` / `run_tier3.sh` (identical
  except the `TIER` default line): clone/update repo → deps → download data (once) → dual-track
  preprocess (once) → run that tier's methods → `results/tier<N>/<method>/`. They resolve the
  tier's methods via `list_methods.py <tier>`. No GPU, no U-Net.

---

## 6. Design decisions (locked)

1. **Dual-track PREPROCESSING** (a preprocessing axis — NOT a seeding policy):
   - **Track A ("fair")** — identical to U-Net (window c40/w400, resize 256). Headline comparison.
   - **Track B ("trad")** — traditional-optimised: narrower window **c50/w150** (more fat/organ
     contrast, sharper boundaries) + **median denoise**, resize 256.
   - **Track Bnr** — Track B, no resize + spacing sidecar (for Tier-3 / mm metrics; not for `auto`).
   - Both tracks use the SAME seeding; they only change the image the methods operate on.
2. **Two seeding/eval REGIMES:**
   - **`auto`** — fully automatic, fair: spleen localised via a **training-fold spatial prior**
     (in-plane location atlas + Gaussian intensity model). No test GT.
   - **`oracle`** — seeds/component from the case GT → **per-method upper bound** (diagnostic).
     Initialised PURELY from GT (does NOT pass through the auto policy; `prior` is not even built
     in an oracle run).
   - *(A third `heuristic` zero-training regime — fixed anatomical prior, no dataset labels — was
     discussed and DEFERRED by the user.)*
3. **Adaptive, always-non-empty auto seeding** (overhaul 2026-07-07, see §8) — replaced the old
   fixed-threshold marker that produced empty fg on ~1/3 of cases.
4. **Per-volume evaluation** exactly matching the U-Net (§4).
5. **Results one folder per method**, under `results/tier<N>/<method>/`.
6. **Emphasis on morphology / nonlinear algebra** throughout (morphological post-processing,
   marker-watershed↔eikonal, morphological level sets, granulometries, Teager energy).

---

## 7. Architecture — file by file (all in `tradseg/`, repo root)

| File | Role |
|---|---|
| `config.py` | Paths + data contract (env-overridable). `spleen_target_intensity()` ≈ 0.52. |
| `preprocessing.py` | raw NIfTI → `(2,Z,S,S)` npy + `splits.pkl`. `TRACKS={A,B,Bnr}`. CLI `--track`. |
| `io_utils.py` | `load_splits`, `fold_cases`, `load_case`, `iter_cases`. |
| `metrics.py` | Dice/Jaccard/Precision/Recall (numpy) + HD95/ASSD (medpy, lazy). `evaluate_case`, `aggregate`. |
| `postprocess.py` | 3D largest-CC, small-object removal, fill_holes(2D/3D), closing/opening, `select_component_by_prior`/`_overlap`, `clean(...)`. |
| `seeding.py` | `SpatialPrior.from_training` (atlas + intensity model); `select` (prior component pick); **NEW adaptive** `auto_markers`/`auto_seed_points` (per-slice top-p% of prior×match, z-gating, guaranteed non-empty, centred) via `_seed_from_score`. Oracle: `oracle_markers`, `oracle_seed_points` (pure GT). |
| `methods/base.py` | `Segmenter` ABC: `segment_volume(image, seeds=None)->bool`; attrs `tier`, `requires_seeds`, `seed_type` (`points`/`markers`). **Never read GT.** |
| `methods/thresholding.py` | **T1** `OtsuSegmenter`, `MultiOtsuSegmenter` (intensity). |
| `methods/clustering.py` | **T1** `KMeansSegmenter`, `GMMSegmenter` (intensity clustering). |
| `methods/region_growing.py` | **T1** `RegionGrowingSegmenter` (flood from points). `seed_type="points"`. |
| `methods/watershed.py` | **T1** `WatershedSegmenter` (marker-controlled, gradient relief). `seed_type="markers"`. |
| `methods/levelset.py` | **T2** `ChanVeseSegmenter` (MorphACWE, region) + `MorphGACSegmenter` (edge). fg marker = init, 2D/slice. |
| `methods/graphcut.py` | **T2** `GraphCutSegmenter` = GMM unaries + contrast-Potts + fg/bg hard seeds (PyMaxflow); = **GMM+MRF**. |
| `methods/randomwalker.py` | **T2** `RandomWalkerSegmenter` (Grady): fg/bg markers, 2D/slice. |
| `methods/texture.py` | **T3** `GaborSegmenter` (Gabor energy), `AmFmSegmenter` (Teager-energy DCA). Feature→KMeans. |
| `methods/granulometry.py` | **T3** `GranulometrySegmenter` (multiscale morphological top-hats). |
| `texture_utils.py` | T3 helpers: Perona-Malik, Gabor bank, Teager energy, granulometry top-hats. |
| `methods/__init__.py` | `REGISTRY` = 6 T1 + 4 T2 + 3 T3 = **13 methods**. |
| `run_experiment.py` | **Generic** per-method runner (any tier): load fold → (auto) build prior → seeds → `segment_volume` → localise → `evaluate_case` → write `results/…/<method>_<regime>_track<T>_f<fold>.json` (+meta tier/time/RSS). New methods need only a `build_method` branch. |
| `list_methods.py` | Prints a tier's method names (by the `tier` attr); used by `run_tier*.sh`. |
| `run_tier1.sh` / `run_tier2.sh` / `run_tier3.sh` | Self-contained per-tier drivers (identical but the `TIER` default). |
| `requirements.txt` | numpy<2, scipy, scikit-image, scikit-learn, PyMaxflow, medpy, nibabel, SimpleITK, pandas, matplotlib. |
| `README.md`, `.gitignore` | usage / ignores (`data/`, `results/`, `__pycache__`, …). |
| `run_losses.sh` | Reference from the U-Net project (unused; can delete). |

**Runtime layout** (git-ignored):
```
data/Task09_Spleen/{imagesTr,labelsTr, preprocessed_A/, preprocessed_B/, splits.pkl}
results/tier<N>/<method>/<method>_<regime>_track<T>_f<fold>.json
```

---

## 8. Status — what is DONE

- **Tiers 1, 2, 3 all implemented** (13 methods) + dual-track preprocessing + auto/oracle seeding
  + per-volume metrics + 3D post-processing + generic runner + per-tier self-contained scripts.
  All verified on synthetic `(2,Z,256,256)` data (unit + end-to-end integration, all exit 0).
- **Run on the real MSD data on the VM** and evaluated: **Tier 1** (all 6), **Tier 2** (all 4),
  **Tier 3 `gabor`** (see §9). *(Tier-1 real results were produced by an earlier `run_all_tier1.sh`
  → old folder/naming `results/<method>/tier1_…`; Tier 2/3 use the current per-tier layout.)*
- **Seeding overhaul (2026-07-07):** `auto_markers`/`auto_seed_points` rewritten — **adaptive
  per-slice top-p%** of the prior×match score, **z-gating** (a slice is seeded only if its peak
  score ≥ `slice_gate`×global-peak), **guaranteed non-empty** (global argmax fallback), **centred
  seeds** (one per spleen slice for region growing). Fixes the ~1/3 empty-fg-marker failures.
  Fairness unchanged (training prior + intensity + test image, no GT). Constants: `top_frac=0.02,
  slice_gate=0.35, min_seed=20`. Verified on synthetic (seeded auto 0.91–1.0). **NOT yet re-run
  on real data.** Affects ONLY `auto` of the seeded methods (T1 watershed/region_growing;
  T2 chanvese/morphgac/graphcut/random_walker).

---

## 9. Real-data RESULTS & key findings (pooled per-case Dice, 5-fold, 41 cases)

**Tier 1** (old seeding):

| method | auto-A | oracle-A | auto-B | oracle-B | note |
|---|---|---|---|---|---|
| otsu / multiotsu / kmeans / gmm | ~0.03–0.06 | ~0.03–0.11 | ~0.01–0.06 | ~0.03–0.11 | **predict-everything** (prec ~0.02); auto≈oracle (one giant component → component-selection no-op) |
| region_growing | 0.233 | 0.372 | 0.159 | 0.484 | seeded, moderate |
| **watershed** | 0.476 | **0.922** | 0.355 | **0.928** | **flagship**; oracle ≈ U-Net |

**Tier 2** (old seeding):

| method | auto-A | oracle-A | auto-B | oracle-B | s/case |
|---|---|---|---|---|---|
| chanvese | 0.490 | 0.783 | 0.389 | 0.849 | ~17s |
| **morphgac** | **0.588** | **0.898** | 0.504 | 0.898 | ~30s |
| graphcut | 0.231 | 0.772 | 0.199 | 0.799 | ~4s |
| **random_walker** | 0.422 | 0.881 | 0.337 | **0.902** | ~2s |

**Tier 3 `gabor`** (amfm / granulometry not yet reviewed): Dice **~0.015–0.06** (auto≈oracle,
prec/rec both low → large mislocated blob), **worse than the intensity baselines**, and the
**slowest / heaviest** (~39 s/case, ~1.2–1.6 GB). Expected: spleen is texture-less, so Gabor
features act as noise; standardizing 8 texture features vs 1 intensity lets noise dominate
(esp. Track A). A valid **negative baseline** ("texture doesn't help for a homogeneous organ").

**Key insights:**
1. **Seeded methods ≈ U-Net given good seeds** (watershed/morphgac/random_walker oracle ~0.88–0.92)
   → the ceiling is high; the whole gap is **automatic localization**.
2. **The `auto` failures were 100% EMPTY predictions** (empty auto fg-markers on ~1/3 of cases);
   0% were mis-located. Conditional on a non-empty marker, morphgac auto ≈ 0.86 (≈ its oracle).
   → the **new adaptive seeding should close most of the auto gap** (re-run pending).
3. **Track A vs B:** in `auto`, A > B (old seeding produced MORE empty markers on B's shifted
   intensities); in `oracle`, **B ≥ A** (B's sharper contrast helps delineation → B works as
   designed). The auto-B deficit is a seeding artefact, expected to vanish after the re-run.
4. **Intensity (T1) and texture (T3) methods "predict everything"** because their candidate is one
   connected soft-tissue blob → component selection can't isolate the spleen (auto≈oracle) → the
   pending **spatial-prior masking** (§10) is the fix.
5. **Cost:** all CPU, 0.4–40 s/case, <1.6 GB. random_walker is the sweet spot (fast + high oracle).

---

## 10. What is NOT done / pending improvements / risks

- **Re-run `auto` of the 6 seeded methods** with the new seeding (delete old `*_auto_*.json`
  first). Oracle / intensity / Tier-3 are unaffected by the seeding change.
- **Pending improvement #1 — spatial-prior MASKING** for `requires_seeds=False` methods
  (T1 intensity ×4 + T3 texture ×3): restrict the candidate to the high `prior×match` region
  before component selection, so they stop predicting the whole body. Applies in
  `run_experiment.py:localise()` (auto branch). Fair (training prior only). NOT implemented.
- **Pending #2 (optional):** a stricter single-GT-seed oracle (tests growth, not just boundary).
- **Pending #3 (deferred by user):** `heuristic` zero-training regime (fixed anatomical prior) to
  showcase the "no training data" advantage over U-Net.
- **amfm / granulometry** real results not yet reviewed (likely similar to gabor: near-zero).
- **U-Net head-to-head table** not yet assembled (needs the U-Net numbers, run separately on GPU).
- **No CSV/table aggregation** committed (ad-hoc scripts only). **Tier 4 hybrids** not started.
- Surface metrics in resized-voxel units (not mm) — consistent, so fair, but note it.

---

## 11. Implementation plan (tiers) & expectations

| Tier | Methods | Status |
|---|---|---|
| **1 — baselines** | otsu, multiotsu, kmeans, gmm, region_growing, watershed | **DONE + evaluated** |
| **2 — core** | chanvese (MorphACWE), morphgac (Morph-GAC), graphcut (=GMM+MRF), random_walker | **DONE + evaluated** |
| **3 — texture/morphology** | gabor, amfm (Teager/DCA), granulometry (pattern spectrum) | **DONE**; gabor evaluated |
| **4 — hybrids** (later) | CNN prob-map + level-set/CRF refine; traditional features as CNN channels; seeds ↔ network; differentiable morphology / active-contour loss; unrolled level sets | TODO |

**Real-data expectations now confirmed:** seeded methods `oracle` ~0.77–0.92; seeded `auto`
0.23–0.59 with old seeding (empty-marker limited) → expected to rise substantially after the
seeding fix; intensity & texture methods ~0.02–0.06 (predict-everything) until masking is added;
U-Net ≈ 0.90–0.95.

---

## 12. Immediate next steps (in order)

1. **Push the updated `seeding.py`; re-run `auto` for the 6 seeded methods** on the VM
   (delete old `*_auto_*.json` first). Verify the auto gap closes.
2. **Implement + run spatial-prior masking** for the intensity + texture methods (fixes predict-
   everything).
3. **Aggregate all results to a CSV/table** (method×regime×track×fold → Dice/HD95/ASSD/time/RSS).
4. **Assemble the U-Net vs traditional head-to-head** table (accuracy / time / memory).
5. Later: optional stricter oracle and/or heuristic zero-training regime; then Tier-4 hybrids.

---

## 13. How to run (commands)

**Self-contained per tier (VM, background):**
```bash
curl -O https://raw.githubusercontent.com/ThanoSnake/CV_SemesterProject_TradMethods/main/run_tier2.sh
nohup bash run_tier2.sh &          # progress: tail -f ~/tradseg-run/run_tier*_*.log
# env overrides: FOLDS="0"  TRACKS="A"  METHODS="morphgac random_walker"  REGIMES="auto"  ADVANCED=0  BRANCH=master
```
Each `run_tier<N>.sh` clones/updates → deps → downloads (once) → preprocesses A&B (once) →
runs that tier's methods → `results/tier<N>/<method>/` → pooled-Dice summary. Idempotent
(skip-if-exists). Run tiers one at a time (shared repo + data). **Re-run only what changed via
`METHODS=... REGIMES=...` overrides + deleting the stale JSONs.**

**Manual (from the repo root on the VM):**
```bash
pip install -r requirements.txt
export DATA_DIR=$PWD/data/Task09_Spleen
python3 preprocessing.py --track A ; python3 preprocessing.py --track B
python3 run_experiment.py --method morphgac --fold 0 --regime auto \
    --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced
```

---

## 14. How to extend (for the next agent)

- **Add a method:** `methods/<name>.py` with a `Segmenter` subclass + `tier` attribute; if seeded,
  set `requires_seeds=True` and `seed_type="points"|"markers"`. Register in `REGISTRY`; add a
  `build_method` branch in `run_experiment.py`. `list_methods.py` + `run_tier<tier>.sh` pick it up.
- **Keep the fairness contract:** operate on ch0 `[0,1]`; never read GT in the segmenter; evaluate
  per-volume via `metrics.evaluate_case`; keep the 256 space for the headline.
- **Everything lives in `tradseg/`** (== repo root). Flat layout, absolute imports (`import config`,
  never `from .. import config`; subpackage-internal relative imports are fine).
- **Verify** new code with numpy/scipy unit checks + a synthetic `(2,Z,256,256)` run through
  `run_experiment.py` before claiming it works.
- **Do NOT auto-update this file** — the user updates PROJECT_STATE.md only on explicit request.

---

## 15. External references & memory

- **Idea sources** (parent dir, not in repo): `../book-cv05_13_chapter13_Texture.pdf` (Textons,
  Gabor/AM-FM energy operators & DESA, MRF/GRF, granulometries) and
  `../book-cv05_17_chapter17_ActivContourLevelsetVar.pdf` (snakes, curve evolution & level sets,
  geometric/geodesic active contours, morphological PDEs, watershed/eikonal, variational methods).
- **Baseline:** `../my_unet-uncertainty/` — U-Net + its preprocessing/splits/eval (fairness contract §4).
- **Persistent agent memory** (auto-loaded via `MEMORY.md`): keys `project-tradsegmethods`,
  `unet-baseline-data-contract`, `env-python-anaconda`, `user-profile`, `feedback-docs-cadence`.

---

*Last updated: 2026-07-07. Tiers 1–3 implemented and run on real data; Tier 1/2 + Tier-3 gabor
evaluated; adaptive-seeding overhaul done (auto re-run pending); spatial-prior masking pending.*
