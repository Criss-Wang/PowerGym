from typing import List

import numpy as np



def cat_f32(parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, dtype=np.float32) if parts else np.zeros(0, np.float32)

def as_f32(value: float) -> np.float32:
    return np.float32(value)

def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if n > 0:
        v[int(np.clip(idx, 0, n - 1))] = 1.0
    return v