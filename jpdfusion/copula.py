"""Step 4: cluster-wise Gaussian-copula fitting and conditional prediction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import rasterio
from scipy import ndimage as ndi
from scipy.stats import norm

from .config import output_dir
from .io import (
    assert_same_grid,
    date_map,
    list_date_rasters,
    output_profile,
    padded_window,
    read_bands_as_nan,
    tile_windows,
    write_json,
)
from .preprocessing import preprocessed_input_dir

LOGGER = logging.getLogger(__name__)
EPS_U = 1.0e-6
CLUSTER_NODATA = 0


def local_mean_std(array: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware local mean and standard deviation."""
    values = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(values).astype(np.float32)
    zeroed = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    size = (int(window_size), int(window_size))
    area = float(window_size * window_size)
    count = ndi.uniform_filter(valid, size=size, mode="nearest") * area
    total = ndi.uniform_filter(zeroed, size=size, mode="nearest") * area
    total_squared = ndi.uniform_filter(zeroed * zeroed, size=size, mode="nearest") * area
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    second = np.divide(total_squared, count, out=np.full_like(total, np.nan), where=count > 0)
    variance = np.maximum(0.0, second - mean * mean)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def gradient_magnitude(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    filled = np.where(np.isfinite(values), values, 0.0)
    gx = ndi.sobel(filled, axis=1, mode="nearest")
    gy = ndi.sobel(filled, axis=0, mode="nearest")
    gradient = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    gradient[~np.isfinite(values)] = np.nan
    return gradient


def sar_features(vv: np.ndarray, vh: np.ndarray, rvi: np.ndarray, texture_window: int) -> np.ndarray:
    vv_mean, vv_std = local_mean_std(vv, texture_window)
    vh_mean, vh_std = local_mean_std(vh, texture_window)
    return np.stack(
        [
            vv,
            vh,
            rvi,
            vv - vh,
            vv_mean,
            vv_std,
            vh_mean,
            vh_std,
            gradient_magnitude(vv),
            gradient_magnitude(vh),
        ],
        axis=-1,
    ).astype(np.float32)


def mcd43a4_features(red: np.ndarray, nir: np.ndarray, texture_window: int) -> np.ndarray:
    denominator = nir + red
    ndvi = np.divide(nir - red, denominator, out=np.full_like(red, np.nan), where=np.isfinite(denominator) & (denominator != 0))
    red_mean, red_std = local_mean_std(red, texture_window)
    nir_mean, nir_std = local_mean_std(nir, texture_window)
    return np.stack(
        [
            red,
            nir,
            ndvi,
            red_mean,
            red_std,
            nir_mean,
            nir_std,
            gradient_magnitude(red),
            gradient_magnitude(nir),
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class WeightedEmpiricalCDF:
    values_sorted: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, weights: np.ndarray) -> "WeightedEmpiricalCDF":
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values = values[valid]
        weights = weights[valid]
        if values.size == 0:
            raise ValueError("Cannot fit an empirical CDF to empty data")
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        probabilities = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
        return cls(sorted_values, np.clip(probabilities, EPS_U, 1.0 - EPS_U))

    def cdf(self, values: np.ndarray) -> np.ndarray:
        probabilities = np.interp(
            np.asarray(values, dtype=np.float64),
            self.values_sorted,
            self.probabilities,
            left=self.probabilities[0],
            right=self.probabilities[-1],
        )
        return np.clip(probabilities, EPS_U, 1.0 - EPS_U)

    def inverse(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPS_U, 1.0 - EPS_U)
        return np.interp(
            clipped,
            self.probabilities,
            self.values_sorted,
            left=self.values_sorted[0],
            right=self.values_sorted[-1],
        )


@dataclass
class GaussianCopulaModel:
    marginals: list[WeightedEmpiricalCDF]
    beta: np.ndarray
    conditional_sigma: float

    @classmethod
    def fit(
        cls,
        response: np.ndarray,
        predictors: np.ndarray,
        weights: np.ndarray,
        shrinkage_alpha: float,
    ) -> "GaussianCopulaModel":
        response = np.asarray(response, dtype=np.float64)
        predictors = np.asarray(predictors, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(response) & np.isfinite(weights) & (weights > 0) & np.all(np.isfinite(predictors), axis=1)
        response, predictors, weights = response[valid], predictors[valid], weights[valid]
        if response.size == 0:
            raise ValueError("No complete cases for Gaussian copula")

        marginals = [WeightedEmpiricalCDF.fit(response, weights)]
        marginals.extend(WeightedEmpiricalCDF.fit(predictors[:, index], weights) for index in range(predictors.shape[1]))
        latent_y = norm.ppf(marginals[0].cdf(response))
        latent_x = np.column_stack(
            [norm.ppf(marginals[index + 1].cdf(predictors[:, index])) for index in range(predictors.shape[1])]
        )
        latent = np.column_stack([latent_y, latent_x])
        total_weight = weights.sum()
        center = np.sum(latent * weights[:, None], axis=0) / total_weight
        centered = latent - center
        covariance = (centered.T * weights) @ centered / total_weight
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1.0e-12))
        correlation = covariance / np.outer(standard_deviation, standard_deviation)
        correlation = np.clip(correlation, -0.9999, 0.9999)
        np.fill_diagonal(correlation, 1.0)
        alpha = float(shrinkage_alpha)
        sigma = (1.0 - alpha) * correlation + alpha * np.eye(correlation.shape[0])

        yx = sigma[0, 1:]
        xx = sigma[1:, 1:]
        inverse_xx = np.linalg.pinv(xx, hermitian=True)
        beta = yx @ inverse_xx
        conditional_variance = max(1.0e-8, float(1.0 - yx @ inverse_xx @ yx))
        return cls(marginals, beta.astype(np.float64), math.sqrt(conditional_variance))

    def conditional_quantiles(self, predictors: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
        predictors = np.asarray(predictors, dtype=np.float64)
        latent_x = np.column_stack(
            [
                norm.ppf(self.marginals[index + 1].cdf(predictors[:, index]))
                for index in range(predictors.shape[1])
            ]
        )
        conditional_mean = latent_x @ self.beta
        output = []
        for probability in probabilities:
            latent_y = conditional_mean + self.conditional_sigma * norm.ppf(float(probability))
            output.append(self.marginals[0].inverse(norm.cdf(latent_y)))
        return np.column_stack(output).astype(np.float32)

    def conditional_variance(self, predictors: np.ndarray) -> np.ndarray:
        central = self.conditional_quantiles(predictors, (0.16, 0.84))
        standard_deviation = (central[:, 1] - central[:, 0]) / 2.0
        return np.maximum(standard_deviation * standard_deviation, 1.0e-12).astype(np.float32)


def _read_auxiliary_feature_tile(
    path: Path,
    source: str,
    window,
    width: int,
    height: int,
    cfg: Mapping[str, object],
) -> np.ndarray:
    copula_cfg = cfg["copula"]
    texture_window = int(copula_cfg["texture_window"])
    pad = texture_window // 2 + 1
    expanded, row_offset, col_offset = padded_window(window, pad, width, height)
    with rasterio.open(path) as dataset:
        if source == "SAR":
            bands = [int(item) for item in copula_cfg["sar_bands"]]
            values = read_bands_as_nan(dataset, bands, expanded)
            features = sar_features(values[0], values[1], values[2], texture_window)
        else:
            bands = [int(copula_cfg["mcd43a4_red_band"]), int(copula_cfg["mcd43a4_nir_band"])]
            values = read_bands_as_nan(dataset, bands, expanded)
            features = mcd43a4_features(values[0], values[1], texture_window)
    return features[
        row_offset : row_offset + int(window.height),
        col_offset : col_offset + int(window.width),
        :,
    ]


def _read_hls_tile(path: Path, window, cfg: Mapping[str, object]) -> np.ndarray:
    raster_cfg = cfg["raster"]
    low, high = [float(item) for item in raster_cfg["hls_valid_range_raw"]]
    with rasterio.open(path) as dataset:
        raw = read_bands_as_nan(dataset, [1], window)[0]
    raw[(raw < low) | (raw > high)] = np.nan
    return raw * float(raster_cfg["hls_scale"])


def _build_samples(
    cluster_id: int,
    target_index: int,
    radius: int,
    cluster_tile: np.ndarray,
    auxiliary_features: Mapping[int, np.ndarray],
    hls_tiles: Mapping[int, np.ndarray],
    sample_per_cluster: int,
    max_total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(cluster_tile == cluster_id)
    if rows.size == 0:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    if rows.size > sample_per_cluster:
        chosen = rng.choice(rows.size, size=sample_per_cluster, replace=False)
        rows, cols = rows[chosen], cols[chosen]

    first = max(0, target_index - radius)
    last = min(max(auxiliary_features) if auxiliary_features else -1, target_index + radius)
    indices = [item for item in range(first, last + 1) if item in auxiliary_features and item in hls_tiles]
    if not indices:
        return np.empty(0), np.empty((0, 0)), np.empty(0)

    validity: dict[int, np.ndarray] = {}
    response: dict[int, np.ndarray] = {}
    predictors: dict[int, np.ndarray] = {}
    valid_counts = np.zeros(rows.size, dtype=np.int32)
    for index in indices:
        y = hls_tiles[index][rows, cols]
        x = auxiliary_features[index][rows, cols, :]
        valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        validity[index] = valid
        response[index] = y
        predictors[index] = x
        valid_counts += valid.astype(np.int32)

    responses: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    total = 0
    for index in indices:
        valid = validity[index] & (valid_counts > 0)
        if not np.any(valid):
            continue
        remaining = max_total - total
        if remaining <= 0:
            break
        selected = np.flatnonzero(valid)[:remaining]
        responses.append(response[index][selected])
        matrices.append(predictors[index][selected])
        weights.append(1.0 / valid_counts[selected].astype(np.float64))
        total += selected.size
    if not responses:
        return np.empty(0), np.empty((0, 0)), np.empty(0)
    return np.concatenate(responses), np.vstack(matrices), np.concatenate(weights)


def _predict_tile(
    target_index: int,
    target_features: np.ndarray,
    cluster_tile: np.ndarray,
    auxiliary_features: Mapping[int, np.ndarray],
    hls_tiles: Mapping[int, np.ndarray],
    cfg: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    copula_cfg = cfg["copula"]
    cluster_cfg = cfg["clustering"]
    quantiles = [float(item) for item in copula_cfg["quantiles"]]
    height, width, feature_count = target_features.shape
    flat_features = target_features.reshape(-1, feature_count)
    flat_clusters = cluster_tile.reshape(-1)
    output = np.full((4, height * width), np.nan, dtype=np.float32)
    minimum = max(int(copula_cfg["min_samples"]), 10 * (feature_count + 1))

    for cluster_id in range(1, int(cluster_cfg["n_clusters"]) + 1):
        model = None
        for radius in range(int(copula_cfg["h_init"]), int(copula_cfg["h_max"]) + 1):
            response, predictors, weights = _build_samples(
                cluster_id,
                target_index,
                radius,
                cluster_tile,
                auxiliary_features,
                hls_tiles,
                int(copula_cfg["pixel_sample_per_cluster"]),
                int(copula_cfg["max_samples_per_cluster"]),
                rng,
            )
            if response.size < minimum or predictors.shape != (response.size, feature_count):
                continue
            try:
                model = GaussianCopulaModel.fit(
                    response,
                    predictors,
                    weights,
                    float(copula_cfg["shrinkage_alpha"]),
                )
                break
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if model is None:
            continue
        selected = np.flatnonzero(
            (flat_clusters == cluster_id) & np.all(np.isfinite(flat_features), axis=1)
        )
        if selected.size == 0:
            continue
        predicted = model.conditional_quantiles(flat_features[selected], quantiles)
        output[0:3, selected] = predicted.T
        output[3, selected] = model.conditional_variance(flat_features[selected])
    return output.reshape(4, height, width)


def copula_source(cfg: Mapping[str, object], source: str, overwrite: bool = False) -> Path:
    source = source.upper()
    auxiliary_items = list_date_rasters(preprocessed_input_dir(cfg, source))
    auxiliary_dates = [item[0] for item in auxiliary_items]
    auxiliary_paths = [item[1] for item in auxiliary_items]
    clusters = date_map(output_dir(cfg, "02_clusters", source))
    hls = date_map(cfg["paths"]["hls"])
    missing_clusters = [item for item in auxiliary_dates if item not in clusters]
    if missing_clusters:
        raise FileNotFoundError(f"Missing cluster rasters for dates: {missing_clusters[:5]}")

    copula_cfg = cfg["copula"]
    required_aux_bands = max([int(item) for item in copula_cfg["sar_bands"]]) if source == "SAR" else max(int(copula_cfg["mcd43a4_red_band"]), int(copula_cfg["mcd43a4_nir_band"]))
    grid = assert_same_grid(auxiliary_paths, minimum_bands=required_aux_bands)
    assert_same_grid(list(hls.values()), minimum_bands=1)
    assert_same_grid([auxiliary_paths[0], *hls.values()], minimum_bands=1)
    with rasterio.open(auxiliary_paths[0]) as reference:
        reference_profile = reference.profile.copy()
    output = output_dir(cfg, "03_copula", source)
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"{item_date:%Y-%m-%d}.tif" for item_date in auxiliary_dates]
    manifest_path = output / "copula_manifest.json"
    if manifest_path.exists() and all(path.exists() for path in expected) and not overwrite:
        LOGGER.info("Step 4 %s already complete; reusing %s", source, output)
        return output

    profile = output_profile(
        reference_profile,
        4,
        "float32",
        float(cfg["raster"]["output_nodata"]),
        str(cfg["raster"]["compression"]),
    )
    quantile_descriptions = tuple(
        f"conditional_q{int(round(float(probability) * 100)):02d}"
        for probability in copula_cfg["quantiles"]
    )
    date_to_index = {item: index for index, item in enumerate(auxiliary_dates)}
    for target_index, target_date in enumerate(auxiliary_dates):
        LOGGER.info("Step 4 %s %s (%d/%d)", source, target_date, target_index + 1, len(auxiliary_dates))
        output_path = output / f"{target_date:%Y-%m-%d}.tif"
        with rasterio.open(output_path, "w", **profile) as writer:
            descriptions = (*quantile_descriptions, "conditional_variance")
            for band, description in enumerate(descriptions, start=1):
                writer.set_band_description(band, description)
            for tile_number, window in enumerate(
                tile_windows(
                    int(grid["width"]),
                    int(grid["height"]),
                    int(copula_cfg["tile_width"]),
                    int(copula_cfg["tile_height"]),
                )
            ):
                first = max(0, target_index - int(copula_cfg["h_max"]))
                last = min(len(auxiliary_dates) - 1, target_index + int(copula_cfg["h_max"]))
                auxiliary_features = {
                    index: _read_auxiliary_feature_tile(
                        auxiliary_paths[index], source, window, int(grid["width"]), int(grid["height"]), cfg
                    )
                    for index in range(first, last + 1)
                }
                hls_tiles = {
                    index: _read_hls_tile(hls[auxiliary_dates[index]], window, cfg)
                    for index in range(first, last + 1)
                    if auxiliary_dates[index] in hls
                }
                with rasterio.open(clusters[target_date]) as cluster_dataset:
                    cluster_tile = cluster_dataset.read(1, window=window)
                rng = np.random.default_rng(int(copula_cfg["random_state"]) + target_index * 1000003 + tile_number)
                predicted = _predict_tile(
                    target_index,
                    auxiliary_features[target_index],
                    cluster_tile,
                    auxiliary_features,
                    hls_tiles,
                    cfg,
                    rng,
                )
                nodata = float(cfg["raster"]["output_nodata"])
                predicted[~np.isfinite(predicted)] = nodata
                writer.write(predicted.astype(np.float32), window=window)

    write_json(
        manifest_path,
        {
            "source": source,
            "dates": [str(item) for item in auxiliary_dates],
            "hls_observation_dates": [str(item) for item in sorted(set(auxiliary_dates) & set(hls))],
            "model": "cluster-wise Gaussian copula",
            "quantiles": [float(item) for item in copula_cfg["quantiles"]],
            "bands": [*quantile_descriptions, "conditional_variance"],
            "units": "physical HLS reflectance; variance is reflectance squared",
            "texture_window": int(copula_cfg["texture_window"]),
            "temporal_radius_range": [int(copula_cfg["h_init"]), int(copula_cfg["h_max"])],
            "pixel_sample_per_cluster": int(copula_cfg["pixel_sample_per_cluster"]),
        },
    )
    return output


def copula_all(cfg: Mapping[str, object], sources: list[str] | None = None, overwrite: bool = False) -> dict[str, Path]:
    selected = sources or list(cfg["sources"])
    return {source: copula_source(cfg, source, overwrite=overwrite) for source in selected}
