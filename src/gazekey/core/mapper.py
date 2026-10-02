from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import F64
from .features import feature_index, valid_mask

IRIS = ("r_h", "l_h", "r_v_corner", "l_v_corner")
POSE = ("yaw", "pitch", "roll", "tx", "ty", "tz")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "iris": IRIS,
    "iris+pose": IRIS + POSE,
    "iris+pose+scale": IRIS + POSE + ("iris_diam",),
}
MAPPER_DEGREES: dict[str, int] = {"ridge": 1, "poly2": 2, "poly3": 3}
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in 10.0 ** np.linspace(-3, 3, 13))
_EPS = 1e-12


def feature_columns(feature_set: str) -> list[int]:
    try:
        names = FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}"
        ) from None
    return [feature_index(n) for n in names]


def usable_mask(features: F64, columns: Sequence[int], open_threshold: float) -> NDArray[np.bool_]:
    out: NDArray[np.bool_] = valid_mask(features, open_threshold) & np.isfinite(
        features[:, list(columns)]
    ).all(axis=1)
    return out


def polynomial_features(x: F64, degree: int) -> F64:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    cols = [
        x[:, list(combo)].prod(axis=1)
        for deg in range(1, degree + 1)
        for combo in combinations_with_replacement(range(x.shape[1]), deg)
    ]
    out: F64 = np.stack(cols, axis=1)
    return out


def _safe_std(a: F64) -> F64:
    std: F64 = a.std(axis=0)
    std[std < _EPS] = 1.0
    return std


def loo_mse(zc: F64, yc: F64, alpha: float) -> float:
    n, p = zc.shape
    a_inv = np.linalg.inv(zc.T @ zc + alpha * np.eye(p))
    hat_diag = 1.0 / n + np.einsum("ij,jk,ik->i", zc, a_inv, zc)
    resid = yc - zc @ (a_inv @ zc.T @ yc)
    loo = resid / np.maximum(1.0 - hat_diag, 1e-9)[:, None]
    return float(np.mean(loo**2))


@dataclass(frozen=True, slots=True, eq=False)
class _Fit:
    x_mean: F64
    x_std: F64
    z_mean: F64
    z_std: F64
    coef: F64
    y_mean: F64
    alpha: float


class RidgeMapper:
    def __init__(
        self,
        degree: int = 1,
        alpha: float | None = None,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if alpha is not None and alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.degree = degree
        self._alpha = alpha
        self._alphas = tuple(alphas)
        self._fit: _Fit | None = None

    @property
    def alpha_(self) -> float:
        if self._fit is None:
            raise RuntimeError("mapper is not fitted")
        return self._fit.alpha

    def _expand(self, x: F64, x_mean: F64, x_std: F64) -> F64:
        return polynomial_features((x - x_mean) / x_std, self.degree)

    def fit(self, x: ArrayLike, y: ArrayLike) -> Self:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if xa.ndim != 2 or ya.shape != (xa.shape[0], 2):
            raise ValueError("x must be (n, d) and y (n, 2)")
        if xa.shape[0] < 2:
            raise ValueError("need at least 2 training samples")
        if not (np.isfinite(xa).all() and np.isfinite(ya).all()):
            raise ValueError("training data must be finite")

        x_mean, x_std = xa.mean(axis=0), _safe_std(xa)
        z = self._expand(xa, x_mean, x_std)
        z_mean, z_std = z.mean(axis=0), _safe_std(z)
        zc = (z - z_mean) / z_std
        y_mean = ya.mean(axis=0)
        yc = ya - y_mean

        alpha = self._alpha
        if alpha is None:
            alpha = min(self._alphas, key=lambda a: loo_mse(zc, yc, a))
        coef = np.linalg.solve(zc.T @ zc + alpha * np.eye(zc.shape[1]), zc.T @ yc)
        self._fit = _Fit(x_mean, x_std, z_mean, z_std, coef, y_mean, alpha)
        return self

    def predict(self, x: ArrayLike) -> F64:
        if self._fit is None:
            raise RuntimeError("mapper is not fitted")
        f = self._fit
        xa = np.asarray(x, dtype=np.float64)
        if xa.ndim != 2 or xa.shape[1] != f.x_mean.shape[0]:
            raise ValueError(f"x must be (n, {f.x_mean.shape[0]})")
        z = (self._expand(xa, f.x_mean, f.x_std) - f.z_mean) / f.z_std
        out: F64 = z @ f.coef + f.y_mean
        return out


def make_mapper(name: str) -> RidgeMapper:
    try:
        return RidgeMapper(degree=MAPPER_DEGREES[name])
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {list(MAPPER_DEGREES)}") from None
