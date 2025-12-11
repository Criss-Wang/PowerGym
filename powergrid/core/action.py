from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
import gymnasium as gym

from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict as SpaceDict

from powergrid.utils.array_utils import cat_f32


@dataclass(slots=True)
class Action:
    """
    - Continuous: `c` in physical units, shape (dim_c,)
    - Multi-discrete: `d`, shape (dim_d,), each d[i] in {0..ncats_i-1}
    - `ncats`: either an int (same categories for all discrete heads)
               or a sequence[int] of length dim_d for per-head categories.
    - `masks`: optional list of boolean arrays, one per head, masks[i].shape==(ncats_i,)
               True=allowed, False=disallowed.

    Use `scale()` / `unscale()` for continuous normalization [-1, 1].
    """

    c: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    d: np.ndarray   = field(default_factory=lambda: np.array([], dtype=np.int32))

    dim_c: int = 0
    dim_d: int = 0

    range: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ncats: Sequence[int] = field(default_factory=list)

    _space: Optional[gym.Space] = None

    @property
    def space(self) -> "gym.Space":
        """Lazily-constructed Gymnasium action space for this Action."""
        if self._space is None:
            self._space = self._to_gym_space()
        return self._space

    def _to_gym_space(self) -> "gym.Space":
        """Construct Gymnasium action space for this Action.
        """
        def _build_continuous_space() -> Box:
            lb, ub = self.range
            return Box(low=lb, high=ub, dtype=np.float32)

        def _build_discrete_space():
            if self.dim_d == 1:
                return Discrete(self.ncats[0])
            return MultiDiscrete(self.ncats)

        if self.dim_c and self.dim_d:
            return SpaceDict({
                "c": _build_continuous_space(),
                "d": _build_discrete_space(),
            })

        if self.dim_c:
            return _build_continuous_space()

        if self.dim_d:
            return _build_discrete_space()

        raise ValueError("Action must have either continuous or discrete components.")

    def _validate_and_prepare(self) -> None:
        # shape init
        if self.dim_c and self.c.size == 0:
            self.c = np.zeros(self.dim_c, dtype=np.float32)
        if self.dim_d and self.d.size == 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        if self.dim_d == 0:
            self.d = np.array([], dtype=np.int32)
        if self.dim_d == 0 and self.ncats != []:
            raise ValueError("ncats must be >=1 when dim_d > 0.")
        if self.dim_d > 0:
            if len(self.ncats) != self.dim_d:
                raise ValueError("len(ncats) must equal dim_d.")

        # range validation
        if self.range is not None:
            lb, ub = self.range
            if lb.shape != ub.shape:
                raise ValueError("range must be a tuple of (lb, ub) with identical shapes.")
            if lb.ndim != 1 or (self.dim_c and lb.shape[0] != self.dim_c):
                raise ValueError("range arrays must be 1D with length == dim_c.")
            if not np.all(lb <= ub):
                raise ValueError("range lower bounds must be <= upper bounds.")

    def set_specs(
        self,
        dim_c: int = 0,
        dim_d: int = 0,
        ncats: Optional[Sequence[int]] = [],
        range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.dim_c, self.dim_d = int(dim_c), int(dim_d)
        if isinstance(ncats, int):
            ncats = [ncats] * self.dim_d
        self.ncats = list(ncats)
        self.c = np.zeros(self.dim_c, dtype=np.float32)
        self.d = np.array([], dtype=np.int32)
        if self.dim_d > 0:
            self.d = np.zeros(self.dim_d, dtype=np.int32)
        if range is None:
            low = np.full(self.dim_c, -np.inf, dtype=np.float32)
            high = np.full(self.dim_c, np.inf, dtype=np.float32)
            self.range = np.asarray([low, high], dtype=np.float32)
        else:
            self.range = np.asarray([
                    np.asarray(range[0], dtype=np.float32),
                    np.asarray(range[1], dtype=np.float32),
            ], dtype=np.float32)
        self._validate_and_prepare()
        self._space = None

    def sample(self, seed: int = None) -> None:
        """
        Sample random action using the underlying Gym space, then populate
        (c, d). If discrete masks are present, any invalid sampled category
        is replaced by a random valid category.

        Continuous part `c` is sampled according to `range`.
        Discrete part `d` is sampled from the Discrete/MultiDiscrete
        space, then post-processed to respect `masks` if provided.
        """
        if seed is not None:
            self.space.seed(seed)

        # Sample from Gym space
        action = self.space.sample()

        # Decode Gym action into (c, d)
        if isinstance(action, dict):
            self.c[...] = action["c"]
            self.d[...] = action["d"]
        else:
            if self.dim_c:
                self.c[...] = action
            if self.dim_d:
                self.d[...] = action
        
        return action

    def reset(
        self,
        action: Any = None,
        *,
        c: Optional[Sequence[float]] = None,
        d: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Reset the action to a neutral value, or to user-provided values.

        Priority:
            1) If `action` is given, delegate to `set_values(action)`.
            2) Else if `c`/`d` are given, delegate to `set_values({"c": c, "d": d})`.
            3) Else:
               - continuous `c` is set to a neutral point based on `range`
                 (midpoint of [lb, ub] where finite, 0.0 otherwise)
               - discrete `d` is set to 0 for all heads.
        """
        if action is not None:
            self.set_values(action)
            return self

        if c is not None or d is not None:
            payload: Dict[str, Any] = {}
            if c is not None:
                payload["c"] = c
            if d is not None:
                payload["d"] = d
            self.set_values(payload)
            return self

        # Neutral reset based on specs
        # -----------------------------
        # Make sure shapes / buffers are initialized
        self._validate_and_prepare()

        # Continuous part: neutral based on range
        if self.dim_c:
            if self.range is not None:
                lb, ub = self.range  # shape (dim_c,)
                neutral = np.zeros_like(lb, dtype=np.float32)
                finite = np.isfinite(lb) & np.isfinite(ub)

                # Midpoint where finite, 0.0 otherwise
                if np.any(finite):
                    neutral[finite] = 0.5 * (lb[finite] + ub[finite])
                # non-finite dimensions stay at 0.0

                self.c[...] = neutral
            else:
                # No range info → default to zeros
                self.c[...] = 0.0

        # Discrete part: neutral = 0 in each head (first category)
        if self.dim_d:
            self.d[...] = 0

        # Ensure everything is within valid ranges/categories
        return self.clip()

    def set_values(
        self,
        action: Any = None,
        *,
        c: Optional[Sequence[float]] = None,
        d: Optional[Sequence[int]] = None,
    ) -> "Action":
        """
        Set this Action from an action value.

        Supported formats:
            - action is a dict with optional keys "c", "d"
            - action is a scalar int for pure discrete (dim_c == 0, dim_d > 0)
            - action is a 1D array-like [c..., d...] of length dim_c + dim_d
            - or via keyword args: c=..., d=... (either or both)

        Unspecified parts (c or d) are left unchanged.
        """
        # Normalize kwargs into an `action` object if needed
        if action is None and (c is not None or d is not None):
            payload: Dict[str, Any] = {}
            if c is not None:
                payload["c"] = c
            if d is not None:
                payload["d"] = d
            action = payload

        if action is None:
            # Nothing to do
            return self.clip()

        # ------------------------------------------------------------
        # Case 1: dict → may contain "c", "d" (either or both)
        if isinstance(action, dict):
            # Continuous part
            if self.dim_c and "c" in action and action["c"] is not None:
                c_arr = np.asarray(action["c"], dtype=np.float32)
                if c_arr.size != self.dim_c:
                    raise ValueError(
                        f"continuous part length {c_arr.size} != dim_c {self.dim_c}"
                    )
                self.c[...] = c_arr

            # Discrete part
            if self.dim_d and "d" in action and action["d"] is not None:
                d_raw = action["d"]
                if np.isscalar(d_raw):
                    d_arr = np.array([int(d_raw)], dtype=np.int32)
                else:
                    d_arr = np.asarray(d_raw, dtype=np.int32)

                if d_arr.size != self.dim_d:
                    raise ValueError(
                        f"discrete part length {d_arr.size} != dim_d {self.dim_d}"
                    )
                self.d[...] = d_arr

            return self.clip()

        # ------------------------------------------------------------
        # Case 2: scalar → pure discrete
        if np.isscalar(action):
            if self.dim_d == 0:
                raise ValueError("set_values: scalar action only valid when dim_d > 0.")
            if self.dim_c != 0:
                raise ValueError(
                    "set_values: scalar action ambiguous when dim_c > 0; "
                    "use dict or d=... instead."
                )
            self.d[...] = int(action)
            return self.clip()

        # ------------------------------------------------------------
        # Case 3: flat vector [c..., d...] or pure c / pure d
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                "set_values expects a 1D array-like for non-dict, non-scalar inputs."
            )

        expected = self.dim_c + self.dim_d
        if arr.size != expected:
            raise ValueError(
                f"Action vector length {arr.size} != expected {expected} "
                f"(dim_c={self.dim_c}, dim_d={self.dim_d})"
            )

        # Continuous slice
        if self.dim_c:
            self.c[...] = arr[: self.dim_c]

        # Discrete slice
        if self.dim_d:
            self.d[...] = arr[self.dim_c :].astype(np.int32)

        return self.clip()

    def clip(self) -> None:
        """Clip `c` to `range` and `d_i` to [0, ncats_i-1] in-place."""
        if self.dim_c:
            lb, ub = self.range
            np.clip(self.c, lb, ub, out=self.c)

        if self.dim_d:
            for i, K in enumerate(self.ncats):
                if self.d[i] < 0:
                    self.d[i] = 0
                elif self.d[i] >= K:
                    self.d[i] = K - 1

    def scale(self) -> np.ndarray:
        """Return normalized [-1, 1] copy of `c`. Zero-span axes → 0."""
        if self.range is None or self.c.size == 0:
            return self.c.astype(np.float32, copy=True)
        lb, ub = self.range
        span = ub - lb
        x = np.zeros_like(self.c, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            x[mask] = 2.0 * (self.c[mask] - lb[mask]) / span[mask] - 1.0
        return x

    def unscale(self, x: Sequence[float]) -> np.ndarray:
        """Set `c` from normalized [-1, 1] vector `x` (physical units). 
           Zero-span axes → lb.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != self.dim_c:
            raise ValueError("normalized vector length must equal dim_c")
        if self.range is None:
            self.c = x.copy()
            return self.c
        lb, ub = self.range
        span = ub - lb
        self.c = np.empty_like(lb, dtype=np.float32)
        mask = span > 0
        if np.any(mask):
            self.c[mask] = lb[mask] + 0.5 * (x[mask] + 1.0) * span[mask]
        if np.any(~mask):
            self.c[~mask] = lb[~mask]
        return self.c

    def vector(self) -> np.ndarray:
        """Flatten to `[c..., d...]` (float32) for logging/export."""
        if self.dim_d:
            parts = [self.c.astype(np.float32), self.d.astype(np.int32)]
            return cat_f32(parts)
        return self.c.astype(np.float32, copy=True)

    def vector(self) -> np.ndarray:
        """Flatten to `[c..., d...]` (float32) for logging/export."""
        if self.dim_d:
            parts = [self.c.astype(np.float32), self.d.astype(np.int32)]
            return cat_f32(parts)
        return self.c.astype(np.float32, copy=True)

    @classmethod
    def from_vector(
        cls,
        vec: Sequence[float],
        dim_c: int,
        dim_d: int,
        ncats: Optional[Sequence[int]] = [],
        range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "Action":
        """Create an Action from a flat vector `[c..., d...]` (d length = dim_d)."""
        vec = np.asarray(vec, dtype=np.float32)
        expected = dim_c + dim_d
        if vec.size != expected:
            raise ValueError(f"vector length {vec.size} != expected {expected}")
        c = vec[:dim_c].astype(np.float32)
        d = vec[dim_c:].astype(np.int32) if dim_d else np.array([], dtype=np.int32)
        return cls(
            c=c, d=d, dim_c=dim_c, dim_d=dim_d, ncats=ncats, range=range, 
        )

    def reset(self) -> None:
        if self.dim_c:
            self.c[...] = 0.0
        if self.dim_d:
            self.d[...] = 0