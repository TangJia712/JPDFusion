from datetime import date

import numpy as np

from jpdfusion.clustering import fixed_window_indices
from jpdfusion.copula import GaussianCopulaModel, WeightedEmpiricalCDF, local_mean_std
from jpdfusion.io import parse_date
from jpdfusion.preprocessing import linear_interpolate_and_edge_fill, observed_constrained_sg
from jpdfusion.residual import compensate_residuals


def test_parse_common_date_formats():
    assert parse_date("HLS_2020-01-09.tif") == date(2020, 1, 9)
    assert parse_date("SAR_20200109_stack.tif") == date(2020, 1, 9)
    assert parse_date("MCD_2020009.tif") == date(2020, 1, 9)


def test_fixed_temporal_window_replicates_edges():
    assert fixed_window_indices(0, 5, 2) == [0, 0, 0, 1, 2]
    assert fixed_window_indices(4, 5, 2) == [2, 3, 4, 4, 4]


def test_linear_interpolation_uses_actual_day_distance_and_edges():
    values = np.array([[np.nan], [2.0], [np.nan], [8.0], [np.nan]], dtype=np.float32)
    valid = np.isfinite(values)
    completed, edge = linear_interpolate_and_edge_fill(values, valid, np.array([0, 2, 4, 8, 10]))
    np.testing.assert_allclose(completed[:, 0], [2.0, 2.0, 4.0, 8.0, 8.0])
    assert edge[:, 0].tolist() == [True, False, False, False, True]


def test_sg_preserves_observed_values():
    observed = np.array([[1.0], [np.nan], [5.0], [np.nan], [9.0]], dtype=np.float32)
    valid = np.isfinite(observed)
    completed, edge = linear_interpolate_and_edge_fill(observed, valid, np.arange(5))
    filtered = observed_constrained_sg(completed, observed, valid, edge, 5, 2, True)
    np.testing.assert_allclose(filtered[valid], observed[valid])


def test_weighted_cdf_and_copula_return_finite_quantiles():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 2))
    y = 0.4 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.2, size=300)
    weights = np.ones(300)
    cdf = WeightedEmpiricalCDF.fit(y, weights)
    assert np.all(np.diff(cdf.probabilities) > 0)
    model = GaussianCopulaModel.fit(y, x, weights, 0.05)
    quantiles = model.conditional_quantiles(x[:10], (0.05, 0.5, 0.95))
    assert np.all(np.isfinite(quantiles))
    assert np.all(quantiles[:, 0] <= quantiles[:, 1])
    assert np.all(quantiles[:, 1] <= quantiles[:, 2])


def test_local_statistics_require_explicit_window():
    mean, std = local_mean_std(np.ones((5, 5), dtype=np.float32), 3)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, 0.0)


def test_residual_compensation_reproduces_valid_hls():
    q50 = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32)
    hls = np.array([[0.11], [np.nan], [0.33], [np.nan], [0.55]], dtype=np.float32)
    _, _, final = compensate_residuals(q50, hls, np.arange(5), 3, 3, True)
    valid = np.isfinite(hls)
    np.testing.assert_allclose(final[valid], hls[valid], atol=1e-6)

