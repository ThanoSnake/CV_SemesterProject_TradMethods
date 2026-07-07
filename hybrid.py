"""Anchored, band-limited refinement of a U-Net segmentation by a classical method.

The two hybrid ideas share ONE engine and differ only in how the refinement band is drawn:

  * Hybrid #1 - uncertainty-gated (on the MC-Dropout net): the band is the set of
    high-uncertainty voxels near the predicted boundary (uncertainty = MC entropy /
    mutual_info / fg-prob variance).  This is the "gate": the classical method is invited
    in exactly where the network is unsure.

  * Hybrid #2 - weakened baseline (on the dropout-FREE weak net): that net has no
    uncertainty, so the band is a purely MORPHOLOGICAL ring straddling the predicted
    boundary (dilation XOR erosion).

In BOTH cases the refiner may only RELABEL voxels inside the band; every voxel outside the
band keeps the U-Net's label.  This ANCHORS the hybrid to the network -- the result equals
the U-Net except inside a thin band -- so a classical refiner can fix boundary/leakage
errors without dragging a strong prediction toward its own (worse) fixed point.  The whole
premise (discussed with the user) is that unrestricted "refine to convergence" degrades a
good CNN; band-anchoring is what makes a gain possible.

Everything is 2D per axial slice, matching the rest of the project.  No post-processing is
added on top (the U-Net baseline is scored raw), so any Dice delta is attributable to the
refiner alone.
"""

import numpy as np


# --------------------------------------------------------------------------- morphology
def _disk(radius):
    from skimage.morphology import disk
    return disk(int(radius))


def _dilate2d(mask2d, radius):
    if radius <= 0:
        return np.asarray(mask2d, bool)
    from skimage.morphology import binary_dilation
    return binary_dilation(np.asarray(mask2d, bool), _disk(radius))


def _erode2d(mask2d, radius):
    if radius <= 0:
        return np.asarray(mask2d, bool)
    from skimage.morphology import binary_erosion
    return binary_erosion(np.asarray(mask2d, bool), _disk(radius))


def _boundary2d(mask2d, radius=1):
    """Symmetric boundary ring of a 2D mask (dilate XOR erode)."""
    return _dilate2d(mask2d, radius) & ~_erode2d(mask2d, radius)


# --------------------------------------------------------------------------- bands
def morph_band(seg, radius=6):
    """Hybrid #2 band: a ring of width ~2*radius straddling the predicted boundary,
    per slice. Voxels here get relabelled; everything else keeps the U-Net label."""
    seg = np.asarray(seg, bool)
    band = np.zeros(seg.shape, bool)
    for z in range(seg.shape[0]):
        if seg[z].any():
            band[z] = _dilate2d(seg[z], radius) & ~_erode2d(seg[z], radius)
    return band


def uncertainty_band(seg, uncertainty, quantile=0.80, ring_radius=12, boundary_radius=1):
    """Hybrid #1 band: near-boundary voxels whose uncertainty is in the top (1-quantile)
    fraction *within that near-boundary region*.

    Kept deliberately conservative (anchored to the predicted boundary): the refiner does
    not hallucinate spleen on slices where the net predicted nothing. `uncertainty` is any
    per-voxel map (higher = less certain), e.g. MC mutual information."""
    seg = np.asarray(seg, bool)
    unc = np.asarray(uncertainty, np.float32)
    band = np.zeros(seg.shape, bool)
    for z in range(seg.shape[0]):
        b = _boundary2d(seg[z], boundary_radius)
        if not b.any():
            continue
        near = _dilate2d(b, ring_radius)
        u = unc[z]
        vals = u[near]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        thr = float(np.quantile(vals[finite], quantile))
        band[z] = near & np.isfinite(u) & (u >= thr)
    return band


# --------------------------------------------------------------------------- refiners
def refine_random_walker(image, seg, band, beta=130.0, tol=1e-3):
    """Grady random walker restricted to the band. Confident U-Net fg/bg (outside the band)
    are fixed markers; band voxels are relabelled by intensity-weighted diffusion. Anchoring
    is automatic: seeded pixels are returned as their own marker."""
    from skimage.segmentation import random_walker
    seg = np.asarray(seg, bool)
    band = np.asarray(band, bool)
    out = seg.copy()
    for z in range(seg.shape[0]):
        if not band[z].any():
            continue
        s = seg[z]
        markers = np.zeros(s.shape, np.int32)
        markers[(~s) & (~band[z])] = 1        # confident background
        markers[s & (~band[z])] = 2           # confident foreground
        if not (markers == 1).any() or not (markers == 2).any():
            continue                          # need both seeds; else keep the U-Net slice
        try:
            lab = random_walker(image[z].astype(np.float64), markers, beta=beta, tol=tol, mode="bf")
            out[z] = (lab == 2)
        except Exception:
            pass                              # numerical failure -> keep the U-Net slice
    return out


def refine_level_set(image, seg, band, kind="gac", iters=10, smoothing=1, balloon=1.0,
                     alpha=100.0, sigma=2.0, lambda1=1.0, lambda2=1.0):
    """Morphological level set (GAC edge-based, or Chan-Vese region-based) initialised at the
    U-Net contour and evolved a few iterations, then CLAMPED to the band: only changes inside
    the band are accepted. Bridges to Ch.17 (active contours / morphological curvature flows)."""
    from skimage.segmentation import (morphological_geodesic_active_contour,
                                      inverse_gaussian_gradient, morphological_chan_vese)
    seg = np.asarray(seg, bool)
    band = np.asarray(band, bool)
    out = seg.copy()
    for z in range(seg.shape[0]):
        if not band[z].any() or not seg[z].any():
            continue
        init = seg[z].astype(np.uint8)
        img = image[z].astype(np.float32)
        if kind == "gac":
            g = inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)
            ls = morphological_geodesic_active_contour(
                g, iters, init_level_set=init, smoothing=smoothing,
                threshold="auto", balloon=balloon)
        else:                                 # region-based Chan-Vese (no edge stop)
            ls = morphological_chan_vese(
                img, iters, init_level_set=init, smoothing=smoothing,
                lambda1=lambda1, lambda2=lambda2)
        ls = ls.astype(bool)
        res = seg[z].copy()
        res[band[z]] = ls[band[z]]            # ANCHOR: accept changes only inside the band
        out[z] = res
    return out


# --------------------------------------------------------------------------- dispatch
def refine_volume(image, seg, band, refiner="rw", **params):
    """Dispatch to a refiner by name: 'rw' | 'ls' (=GAC) | 'ls_cv' (Chan-Vese)."""
    if refiner == "rw":
        return refine_random_walker(image, seg, band,
                                    beta=params.get("beta", 130.0),
                                    tol=params.get("tol", 1e-3))
    if refiner in ("ls", "ls_gac", "ls_cv"):
        kind = "cv" if refiner == "ls_cv" or params.get("ls_kind") == "cv" else "gac"
        return refine_level_set(
            image, seg, band, kind=kind,
            iters=params.get("iters", 10), smoothing=params.get("smoothing", 1),
            balloon=params.get("balloon", 1.0), alpha=params.get("alpha", 100.0),
            sigma=params.get("sigma", 2.0), lambda1=params.get("lambda1", 1.0),
            lambda2=params.get("lambda2", 1.0))
    raise ValueError(f"unknown refiner {refiner!r} (use 'rw', 'ls', or 'ls_cv')")
