"""Binning module for scorecard development."""

from cr_score.binning.fine_classing import FineClasser
from cr_score.binning.coarse_classing import CoarseClasser
from cr_score.binning.monotonic_merge import MonotonicMerger

try:
    from cr_score.binning.optbinning_wrapper import OptBinningWrapper, AutoBinner
    __all__ = ["FineClasser", "CoarseClasser", "MonotonicMerger", "OptBinningWrapper", "AutoBinner"]
except ImportError:
    # optbinning not installed
    __all__ = ["FineClasser", "CoarseClasser", "MonotonicMerger"]
