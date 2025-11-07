import numpy as np


Array = np.ndarray
FloatArray = np.ndarray
IntArray = np.ndarray


def float_if_not_none(x: Any) -> Any:
    """Convert to float if not None."""
    return None if x is None else float(x)