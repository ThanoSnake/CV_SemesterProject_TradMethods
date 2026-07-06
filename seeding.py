"""Seeding + component selection for the AUTO and ORACLE regimes.

AUTO  : build a probabilistic in-plane location prior + spleen intensity model from
        the TRAINING fold labels (never the test GT). Use it to (a) select the
        candidate component that best matches the prior and (b) place fg/bg markers
        or seed points for the seeded methods. Automatic yet fair (uses training data,
        like the U-Net).
ORACLE: derive markers / seed points / selection from the case's own GT -> the
        per-method upper bound.

The in-plane prior assumes a fixed common in-plane size (256x256), i.e. Track A or a
resized Track B. For the no-resize Track B variant use the oracle regime (or a heuristic).
"""

import numpy as np
from scipy import ndimage as ndi

from . import config, io_utils
from . import postprocess as P


class SpatialPrior:
    """P(spleen | in-plane location) + a Gaussian intensity model, from training."""

    def __init__(self, prior2d, inten_mean, inten_std):
        self.prior2d = np.asarray(prior2d, np.float32)      # (H, W) normalised to [0, 1]
        self.inten_mean = float(inten_mean)
        self.inten_std = float(max(inten_std, 1e-3))

    @classmethod
    def from_training(cls, train_names, preprocessed_dir=None, foreground_label=None,
                      smooth_sigma=4.0):
        fl = config.FOREGROUND_LABEL if foreground_label is None else foreground_label
        acc = None
        isum = isq = 0.0
        icount = 0
        for _, image, label in io_utils.iter_cases(train_names, preprocessed_dir):
            fg = label == fl
            if acc is None:
                acc = np.zeros(fg.shape[1:], np.float64)     # (H, W)
            elif fg.shape[1:] != acc.shape:
                raise ValueError("training cases have different in-plane sizes; the "
                                 "spatial prior needs a fixed size (resized track)")
            acc += fg.sum(axis=0)
            vals = image[fg]
            isum += float(vals.sum()); isq += float((vals ** 2).sum()); icount += int(vals.size)
        if acc is None or icount == 0:
            raise RuntimeError("could not build a spatial prior (no foreground in training)")
        if smooth_sigma:
            acc = ndi.gaussian_filter(acc, smooth_sigma)
        prior2d = acc / acc.max() if acc.max() > 0 else acc
        mean = isum / icount
        std = max(isq / icount - mean ** 2, 1e-6) ** 0.5
        return cls(prior2d, mean, std)

    # -- scoring --------------------------------------------------------------
    def prior_volume(self, shape):
        return np.broadcast_to(self.prior2d, shape)          # (H,W) -> (Z,H,W)

    def intensity_match(self, image):
        return np.exp(-0.5 * ((image - self.inten_mean) / self.inten_std) ** 2)

    def score_volume(self, image, body):
        return self.prior_volume(image.shape) * self.intensity_match(image) * body

    # -- component selection (intensity methods) ------------------------------
    def select(self, mask, image):
        return P.select_component_by_prior(mask, self.prior_volume(image.shape), reduce="sum")

    # -- markers for watershed ------------------------------------------------
    def auto_markers(self, image, body, hi=0.6, lo=0.12, match_thr=0.5, erode=1):
        prior = self.prior_volume(image.shape)
        match = self.intensity_match(image)
        fg = (prior > hi) & (match > match_thr) & body
        if fg.any() and erode:
            er = ndi.binary_erosion(fg, iterations=erode)
            fg = er if er.any() else fg
        bg = (~body) | (body & (prior < lo))
        return fg, bg

    # -- seed points for region growing ---------------------------------------
    def auto_seed_points(self, image, body, k=1):
        score = self.score_volume(image, body)
        if not np.any(score > 0):
            return []
        idx = np.argsort(score, axis=None)[::-1][:k]
        return [tuple(int(c) for c in np.unravel_index(i, score.shape)) for i in idx]


# --------------------------- oracle (GT-derived) -----------------------------
def _fg(label, foreground_label):
    fl = config.FOREGROUND_LABEL if foreground_label is None else foreground_label
    return np.asarray(label) == fl


def oracle_markers(label, foreground_label=None, body=None, fg_erode=2, bg_dilate=6):
    gt = _fg(label, foreground_label)
    fg = ndi.binary_erosion(gt, iterations=fg_erode) if gt.any() else gt
    if not fg.any():
        fg = gt
    bg = ~ndi.binary_dilation(gt, iterations=bg_dilate)
    if body is not None:
        bg = bg | (~body)
    return fg, bg


def oracle_seed_points(label, foreground_label=None, k=1):
    gt = _fg(label, foreground_label)
    if not gt.any():
        return []
    dt = ndi.distance_transform_edt(gt)                       # deepest interior voxels
    idx = np.argsort(dt, axis=None)[::-1][:k]
    return [tuple(int(c) for c in np.unravel_index(i, gt.shape)) for i in idx]
