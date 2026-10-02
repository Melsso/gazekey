import math

import numpy as np
from numpy.typing import ArrayLike

from . import F64


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t_ns: int | None = None
        self._x_raw: F64 = np.zeros(0)
        self._x_hat: F64 = np.zeros(0)
        self._dx_hat: F64 = np.zeros(0)

    def update(self, t_ns: int, x: ArrayLike) -> F64:
        """Feed one sample; returns the filtered value. The first sample passes through."""
        value = np.asarray(x, dtype=np.float64)
        if self._t_ns is None:
            self._t_ns, self._x_raw, self._x_hat = t_ns, value, value.copy()
            self._dx_hat = np.zeros_like(value)
            return value.copy()
        dt = (t_ns - self._t_ns) * 1e-9
        if dt <= 0:
            return self._x_hat.copy()
        dx = (value - self._x_raw) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx_hat))
        a = _alpha(cutoff, dt)
        self._x_hat = a * value + (1.0 - a) * self._x_hat
        self._x_raw, self._t_ns = value, t_ns
        return self._x_hat.copy()


class GazeFilter:
    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
        hold_s: float = 0.4,
        reset_gap_s: float = 1.0,
    ) -> None:
        self._smoother = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._hold_ns = round(hold_s * 1e9)
        self._reset_ns = round(reset_gap_s * 1e9)
        self._last_valid_ns: int | None = None
        self._last_out: F64 | None = None

    def update(self, t_ns: int, point: ArrayLike | None) -> tuple[F64 | None, bool]:
        if point is None:
            if (
                self._last_out is not None
                and self._last_valid_ns is not None
                and t_ns - self._last_valid_ns <= self._hold_ns
            ):
                return self._last_out.copy(), True
            return None, False
        if self._last_valid_ns is not None and t_ns - self._last_valid_ns > self._reset_ns:
            self._smoother.reset()
        out = self._smoother.update(t_ns, point)
        self._last_valid_ns, self._last_out = t_ns, out
        return out.copy(), False
