from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from powergrid.features.base import FeatureProvider
from powergrid.utils.registry import provider
from powergrid.utils.typing import Array


@provider()
@dataclass(slots=True)
class StatusBlock(FeatureProvider):
    in_service: Optional[bool] = None
    out_service: Optional[bool] = None

    state: Optional[str] = None
    states_vocab: Optional[List[str]] = None

    t_in_state_s: Optional[float] = None
    t_to_next_s: Optional[float] = None
    progress_frac: Optional[float] = None  # [0..1]

    emit_state_one_hot: bool = True
    emit_state_index: bool = False

    lock_schema: bool = True

    _export_in_service: bool = field(default=False, init=False, repr=False)
    _export_out_service: bool = field(default=False, init=False, repr=False)
    _export_state_oh: bool = field(default=False, init=False, repr=False)
    _export_state_idx: bool = field(default=False, init=False, repr=False)
    _export_t_in: bool = field(default=False, init=False, repr=False)
    _export_t_to: bool = field(default=False, init=False, repr=False)
    _export_prog: bool = field(default=False, init=False, repr=False)
    _vocab_index: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._validate_()
        self._prime_schema_()
        self.clamp_()

    def _validate_(self) -> None:
        if self.in_service is True and self.out_service is True:
            raise ValueError(
                "StatusBlock: `in_service` and `out_service` cannot both be True."
            )

        if self.in_service is False and self.out_service is False:
            raise ValueError(
                "StatusBlock: `in_service` and `out_service` cannot both be False."
            )

        if self.states_vocab is not None:
            if not isinstance(self.states_vocab, list) or not self.states_vocab:
                raise ValueError("states_vocab must be a non-empty list of strings.")
            if len(set(self.states_vocab)) != len(self.states_vocab):
                raise ValueError("states_vocab contains duplicates.")
            if not all(isinstance(s, str) and s for s in self.states_vocab):
                raise ValueError("states_vocab must contain non-empty strings.")

        if self.state is not None and self.states_vocab is not None:
            if self.state not in self.states_vocab:
                raise ValueError(
                    f"state '{self.state}' not in states_vocab {self.states_vocab}."
                )
    
        for v, nm in ((self.t_in_state_s, "t_in_state_s"),
                      (self.t_to_next_s, "t_to_next_s")):
            if v is not None and float(v) < 0.0:
                raise ValueError(f"{nm} must be >= 0.")

        if self.progress_frac is not None:
            p = float(self.progress_frac)
            if not (0.0 <= p <= 1.0):
                raise ValueError("progress_frac must be in [0,1].")

        # If state is present, enforce exporting at least one representation
        if self.state is not None and self.states_vocab is not None:
            if not (self.emit_state_one_hot or self.emit_state_index):
                raise ValueError(
                    "state is present; enable emit_state_one_hot or emit_state_index."
                )

        if self.emit_state_one_hot is True and self.emit_state_index is True:
            raise ValueError(
                "StatusBlock: " \
                "`emit_state_one_hot` and `emit_state_one_hot` cannot both be True."
            )

        if self.emit_state_one_hot is False and self.emit_state_index is False:
            raise ValueError(
                "StatusBlock: " \
                "`emit_state_one_hot` and `emit_state_one_hot` cannot both be False."
            )

        # cache vocab index for O(1) lookup later
        if self.states_vocab:
            self._vocab_index = {s: i for i, s in enumerate(self.states_vocab)}

    def _prime_schema_(self) -> None:
        """Decide once which slots to export to keep shapes/names stable."""
        if not self.lock_schema:
            return  # dynamic schema allowed (legacy behavior)

        self._export_in_service  = self.in_service  is not None or self._export_in_service
        self._export_out_service = self.out_service is not None or self._export_out_service

        idx_present = (self.state is not None and self.states_vocab is not None)
        self._export_state_oh  = (
            (self.emit_state_one_hot and idx_present) or self._export_state_oh
        )
        self._export_state_idx = (
            (self.emit_state_index and idx_present) or self._export_state_idx
        )

        self._export_t_in = self.t_in_state_s  is not None or self._export_t_in
        self._export_t_to = self.t_to_next_s   is not None or self._export_t_to
        self._export_prog = self.progress_frac is not None or self._export_prog

    def _state_index(self) -> Optional[int]:
        if self.state is None or self.states_vocab is None:
            return None
        # O(1) lookup; validation ensures key exists
        return self._vocab_index[self.state]

    def _one_hot(self, idx: int, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        if 0 <= idx < n:
            out[idx] = 1.0
        return out

    def vector(self) -> Array:
        parts: List[np.ndarray] = []

        # booleans — include if schema locked to include, else only when not None
        if (
            (self.lock_schema and self._export_in_service)
            or (not self.lock_schema and self.in_service is not None)
        ):
            parts.append(np.array([1.0 if self.in_service else 0.0], np.float32))

        if (
            (self.lock_schema and self._export_out_service)
            or (not self.lock_schema and self.out_service is not None)
        ):
            parts.append(np.array([1.0 if self.out_service else 0.0], np.float32))

        # categorical state
        idx = self._state_index()
        if idx is not None:
            n = len(self.states_vocab)
            if (
                (self.lock_schema and self._export_state_oh)
                or (not self.lock_schema and self.emit_state_one_hot)
            ):
                parts.append(self._one_hot(idx, n))
            if (
                (self.lock_schema and self._export_state_idx)
                or (not self.lock_schema and self.emit_state_index)
            ):
                parts.append(np.array([idx], np.float32))
        else:
            # if schema says we must export state (but state is currently missing), 
            # emit zeros
            if (
                self.lock_schema
                and (self._export_state_oh or self._export_state_idx)
                and self.states_vocab
            ):
                if self._export_state_oh:
                    parts.append(np.zeros(len(self.states_vocab), np.float32))
                if self._export_state_idx:
                    parts.append(np.zeros(1, np.float32))

        # timing / progress
        def _maybe(v, flag):
            return (self.lock_schema and flag) or (not self.lock_schema and v is not None)

        if _maybe(self.t_in_state_s, self._export_t_in):
            v = [0.0 if self.t_in_state_s is None else self.t_in_state_s]
            parts.append(np.array(v, np.float32))
        if _maybe(self.t_to_next_s, self._export_t_to):
            v = [0.0 if self.t_to_next_s is None else self.t_to_next_s]
            parts.append(np.array(v, np.float32))
        if _maybe(self.progress_frac, self._export_prog):
            v = [0.0 if self.progress_frac is None else self.progress_frac]
            parts.append(np.array(v, np.float32))

        if not parts:
            return np.zeros(0, np.float32)

        return np.concatenate(parts).astype(np.float32)

    def names(self) -> List[str]:
        n: List[str] = []

        if (
            (self.lock_schema and self._export_in_service)
            or (not self.lock_schema and self.in_service is not None)
        ):
            n.append("in_service")
        if (
            (self.lock_schema and self._export_out_service)
            or (not self.lock_schema and self.out_service is not None)
        ):
            n.append("out_service")

        # state names according to schema
        if (
            (self.lock_schema and (self._export_state_oh or self._export_state_idx))
            or (not self.lock_schema and self._state_index() is not None)
        ):
            if (
                (self.lock_schema and self._export_state_oh)
                or (not self.lock_schema and self.emit_state_one_hot)
            ):
                n += [f"state_{tok}" for tok in (self.states_vocab or [])]
            if (
                (self.lock_schema and self._export_state_idx)
                or (not self.lock_schema and self.emit_state_index)
            ):
                n.append("state_idx")

        if (
            (self.lock_schema and self._export_t_in)
            or (not self.lock_schema and self.t_in_state_s is not None)
        ):
            n.append("t_in_state_s")
        if (
            (self.lock_schema and self._export_t_to)
            or (not self.lock_schema and self.t_to_next_s is not None)
        ):
            n.append("t_to_next_s")
        if (
            (self.lock_schema and self._export_prog)
            or (not self.lock_schema and self.progress_frac is not None)
        ):
            n.append("progress_frac")

        return n

    def clamp_(self) -> None:
        if self.t_in_state_s is not None:
            self.t_in_state_s = max(0.0, self.t_in_state_s)
        if self.t_to_next_s is not None:
            self.t_to_next_s = max(0.0, self.t_to_next_s)
        if self.progress_frac is not None:
            self.progress_frac = np.clip(self.progress_frac, 0.0, 1.0)

    def to_dict(self) -> Dict:
        d = {
            "in_service": self.in_service,
            "out_service": self.out_service,
            "state": self.state,
            "states_vocab": self.states_vocab,
            "t_in_state_s": self.t_in_state_s,
            "t_to_next_s": self.t_to_next_s,
            "progress_frac": self.progress_frac,
            "emit_state_one_hot": self.emit_state_one_hot,
            "emit_state_index": self.emit_state_index,
            "lock_schema": self.lock_schema,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "StatusBlock":
        obj = cls(
            in_service=d.get("in_service"),
            out_service=d.get("out_service"),
            state=d.get("state"),
            states_vocab=d.get("states_vocab"),
            t_in_state_s=d.get("t_in_state_s"),
            t_to_next_s=d.get("t_to_next_s"),
            progress_frac=d.get("progress_frac"),
            emit_state_one_hot=d.get("emit_state_one_hot", True),
            emit_state_index=d.get("emit_state_index", False),
            lock_schema=d.get("lock_schema", True),
        )
        return obj
