"""Helper functions for normalization and safe operations."""

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(max(b, 1e-6))


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi - lo <= 1e-6: return 0.0
    return float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))


# Public API (without underscore)
safe_div = _safe_div
norm01 = _norm01


