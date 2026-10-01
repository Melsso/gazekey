import numpy as np
import pytest

from gazekey.core import F64
from gazekey.core.features import FEATURE_NAMES
from gazekey.core.mapper import (
    DEFAULT_ALPHAS,
    FEATURE_SETS,
    RidgeMapper,
    feature_columns,
    loo_mse,
    make_mapper,
    polynomial_features,
)


def linear_data(n: int, noise: float = 0.0, seed: int = 0) -> tuple[F64, F64]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    a = np.array([[300.0, 10.0], [-250.0, 5.0], [20.0, 400.0], [15.0, -350.0]])
    y = x @ a + np.array([720.0, 450.0]) + rng.normal(0.0, noise, size=(n, 2))
    return x, y


def test_polynomial_features_layout() -> None:
    x = np.array([[2.0, 3.0]])
    assert polynomial_features(x, 1).tolist() == [[2.0, 3.0]]
    assert polynomial_features(x, 2).tolist() == [[2.0, 3.0, 4.0, 6.0, 9.0]]
    assert polynomial_features(x, 3).shape == (1, 9)
    with pytest.raises(ValueError):
        polynomial_features(x, 0)


def test_feature_sets_reference_real_features() -> None:
    for name, names in FEATURE_SETS.items():
        assert set(names) <= set(FEATURE_NAMES), name
        assert feature_columns(name) == [FEATURE_NAMES.index(n) for n in names]
    assert len(feature_columns("iris")) == 4
    with pytest.raises(ValueError, match="unknown feature set"):
        feature_columns("nope")


def test_recovers_known_linear_mapping_on_a_calibration_sized_set() -> None:
    x, y = linear_data(9)
    mapper = RidgeMapper().fit(x, y)
    x_test, y_test = linear_data(200, seed=5)
    assert np.abs(mapper.predict(x_test) - y_test).max() < 1.0


def test_recovers_linear_mapping_under_noise() -> None:
    x, y = linear_data(300, noise=5.0)
    mapper = RidgeMapper(alpha=1e-3).fit(x, y)
    x_test, y_test = linear_data(300, seed=9)
    assert np.sqrt(np.mean((mapper.predict(x_test) - y_test) ** 2)) < 3.0


def test_polynomial_model_recovers_quadratic_mapping_linear_cannot() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(200, 2))
    y = np.stack([500 * x[:, 0] ** 2 + 100 * x[:, 1], 300 * x[:, 0] * x[:, 1] + 50], axis=1)
    x_test = rng.uniform(-1, 1, size=(200, 2))
    y_test = np.stack(
        [500 * x_test[:, 0] ** 2 + 100 * x_test[:, 1], 300 * x_test[:, 0] * x_test[:, 1] + 50],
        axis=1,
    )
    poly = RidgeMapper(degree=2, alpha=1e-6).fit(x, y)
    lin = RidgeMapper(degree=1, alpha=1e-6).fit(x, y)
    assert np.abs(poly.predict(x_test) - y_test).max() < 1e-2
    assert np.abs(lin.predict(x_test) - y_test).max() > 50.0


def test_closed_form_loo_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    z = rng.normal(size=(9, 5))
    y = rng.normal(size=(9, 2)) * 100.0
    zc, yc = z - z.mean(axis=0), y - y.mean(axis=0)
    for alpha in (1e-3, 0.5, 20.0):
        errors = []
        for i in range(9):
            keep = np.arange(9) != i
            z_tr, y_tr = zc[keep], y[keep]
            zm, ym = z_tr.mean(axis=0), y_tr.mean(axis=0)
            a = (z_tr - zm).T @ (z_tr - zm) + alpha * np.eye(5)
            coef = np.linalg.solve(a, (z_tr - zm).T @ (y_tr - ym))
            errors.append((zc[i] - zm) @ coef + ym - y[i])
        assert loo_mse(zc, yc, alpha) == pytest.approx(np.mean(np.square(errors)), rel=1e-8)


def test_cross_validation_picks_more_shrinkage_for_noisy_overparameterised_fits() -> None:
    x_clean, y_clean = linear_data(9)
    x_noisy, y_noisy = linear_data(9, noise=80.0, seed=2)
    clean = RidgeMapper().fit(x_clean, y_clean).alpha_
    noisy = RidgeMapper(degree=2).fit(x_noisy, y_noisy).alpha_
    assert clean == DEFAULT_ALPHAS[0]
    assert noisy > clean
    assert RidgeMapper(alpha=0.25).fit(x_clean, y_clean).alpha_ == 0.25


def test_constant_feature_column_is_harmless() -> None:
    x, y = linear_data(20)
    x[:, 2] = 7.0
    pred = RidgeMapper().fit(x, y).predict(x)
    assert np.isfinite(pred).all()


def test_fit_is_deterministic() -> None:
    x, y = linear_data(9, noise=3.0)
    a = RidgeMapper().fit(x, y).predict(x)
    b = RidgeMapper().fit(x, y).predict(x)
    assert np.array_equal(a, b)


def test_input_validation() -> None:
    x, y = linear_data(9)
    with pytest.raises(RuntimeError):
        RidgeMapper().predict(x)
    with pytest.raises(RuntimeError):
        _ = RidgeMapper().alpha_
    with pytest.raises(ValueError):
        RidgeMapper().fit(x, y[:, :1])
    with pytest.raises(ValueError):
        RidgeMapper().fit(x[:1], y[:1])
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        RidgeMapper().fit(bad, y)
    mapper = RidgeMapper().fit(x, y)
    with pytest.raises(ValueError):
        mapper.predict(x[:, :3])
    with pytest.raises(ValueError):
        RidgeMapper(degree=0)
    with pytest.raises(ValueError):
        RidgeMapper(alpha=-1.0)


def test_make_mapper() -> None:
    assert make_mapper("ridge").degree == 1
    assert make_mapper("poly2").degree == 2
    assert make_mapper("poly3").degree == 3
    with pytest.raises(ValueError, match="unknown model"):
        make_mapper("svm")
