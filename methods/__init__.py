"""Tier-1 method registry."""

from .base import Segmenter
from .thresholding import OtsuSegmenter, MultiOtsuSegmenter
from .clustering import KMeansSegmenter, GMMSegmenter
from .region_growing import RegionGrowingSegmenter
from .watershed import WatershedSegmenter

REGISTRY = {
    "otsu": OtsuSegmenter,
    "multiotsu": MultiOtsuSegmenter,
    "kmeans": KMeansSegmenter,
    "gmm": GMMSegmenter,
    "region_growing": RegionGrowingSegmenter,
    "watershed": WatershedSegmenter,
}

__all__ = [
    "Segmenter", "REGISTRY",
    "OtsuSegmenter", "MultiOtsuSegmenter", "KMeansSegmenter", "GMMSegmenter",
    "RegionGrowingSegmenter", "WatershedSegmenter",
]
