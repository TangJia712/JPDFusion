"""Step 3: temporally contextualized MiniBatchKMeans clustering."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import rasterio
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans

from .config import output_dir
from .io import assert_same_grid, iter_windows, list_date_rasters, output_profile, read_bands_as_nan, write_json
from .preprocessing import preprocessed_input_dir

LOGGER = logging.getLogger(__name__)
CLUSTER_NODATA = np.uint16(0)


def fixed_window_indices(index: int, length: int, half_window: int) -> list[int]:
    """Return a fixed-size temporal neighborhood with edge replication."""
    return [min(max(item, 0), length - 1) for item in range(index - half_window, index + half_window + 1)]


def _feature_bands(cfg: Mapping[str, object], source: str) -> list[int]:
    key = "sar_feature_bands" if source.upper() == "SAR" else "mcd43a4_feature_bands"
    return [int(item) for item in cfg["clustering"][key]]


def _stack_window(datasets: Sequence[rasterio.io.DatasetReader], bands: Sequence[int], window) -> np.ndarray:
    pieces = [read_bands_as_nan(dataset, bands, window) for dataset in datasets]
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def _collect_samples(
    datasets: Sequence[rasterio.io.DatasetReader],
    reference: rasterio.io.DatasetReader,
    bands: Sequence[int],
    sample_size: int,
    block_size: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    samples: list[np.ndarray] = []
    collected = 0
    for window in iter_windows(reference.width, reference.height, block_size):
        stack = _stack_window(datasets, bands, window)
        matrix = stack.reshape(stack.shape[0], -1).T
        matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
        if matrix.size == 0:
            continue
        remaining = sample_size - collected
        if remaining <= 0:
            break
        take = min(matrix.shape[0], remaining, 20000)
        if matrix.shape[0] > take:
            matrix = matrix[rng.choice(matrix.shape[0], size=take, replace=False)]
        samples.append(matrix)
        collected += matrix.shape[0]
    if not samples:
        raise RuntimeError("No complete auxiliary samples are available for clustering")
    return np.vstack(samples)


def _fit_model(
    samples: np.ndarray,
    n_clusters: int,
    batch_size: int,
    max_iter: int,
    random_state: int,
    init_centers: np.ndarray | None,
) -> MiniBatchKMeans:
    if samples.shape[0] < n_clusters:
        raise RuntimeError(f"Only {samples.shape[0]} complete samples for {n_clusters} clusters")
    kwargs = dict(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
        reassignment_ratio=0.01,
    )
    if init_centers is None:
        model = MiniBatchKMeans(n_init=3, **kwargs)
    else:
        model = MiniBatchKMeans(init=init_centers, n_init=1, **kwargs)
    return model.fit(samples)


def _alignment_mapping(previous: np.ndarray | None, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map current zero-based labels onto the previous date's aligned labels."""
    count = current.shape[0]
    if previous is None:
        return np.arange(count, dtype=np.int32), current.copy()
    distances = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
    aligned_ids, current_ids = linear_sum_assignment(distances)
    current_to_aligned = np.empty(count, dtype=np.int32)
    current_to_aligned[current_ids] = aligned_ids
    aligned_centers = np.empty_like(current)
    aligned_centers[aligned_ids] = current[current_ids]
    return current_to_aligned, aligned_centers


def _write_labels(
    datasets: Sequence[rasterio.io.DatasetReader],
    reference: rasterio.io.DatasetReader,
    bands: Sequence[int],
    model: MiniBatchKMeans,
    mapping: np.ndarray,
    output_path: Path,
    block_size: int,
    compression: str,
) -> None:
    profile = output_profile(reference.profile, 1, "uint16", int(CLUSTER_NODATA), compression)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as writer:
        writer.set_band_description(1, "temporally_aligned_cluster_id")
        for window in iter_windows(reference.width, reference.height, block_size):
            stack = _stack_window(datasets, bands, window)
            matrix = stack.reshape(stack.shape[0], -1).T
            valid = np.all(np.isfinite(matrix), axis=1)
            labels = np.full(matrix.shape[0], CLUSTER_NODATA, dtype=np.uint16)
            if np.any(valid):
                current = model.predict(matrix[valid])
                labels[valid] = (mapping[current] + 1).astype(np.uint16)
            writer.write(labels.reshape(int(window.height), int(window.width)), 1, window=window)


def cluster_source(cfg: Mapping[str, object], source: str, overwrite: bool = False) -> Path:
    source = source.upper()
    input_dir = preprocessed_input_dir(cfg, source)
    items = list_date_rasters(input_dir)
    dates = [item[0] for item in items]
    paths = [item[1] for item in items]
    bands = _feature_bands(cfg, source)
    assert_same_grid(paths, minimum_bands=max(bands))
    cluster_cfg = cfg["clustering"]
    raster_cfg = cfg["raster"]
    output = output_dir(cfg, "02_clusters", source)
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / f"{item_date:%Y-%m-%d}.tif" for item_date in dates]
    manifest_path = output / "clustering_manifest.json"
    if manifest_path.exists() and all(path.exists() for path in expected) and not overwrite:
        LOGGER.info("Step 3 %s already complete; reusing %s", source, output)
        return output

    half_window = int(cluster_cfg["half_window"])
    n_clusters = int(cluster_cfg["n_clusters"])
    previous_centers: np.ndarray | None = None
    for target_index, (target_date, _) in enumerate(items):
        indices = fixed_window_indices(target_index, len(items), half_window)
        LOGGER.info("Step 3 %s %s (%d/%d)", source, target_date, target_index + 1, len(items))
        with ExitStack() as stack:
            datasets = [stack.enter_context(rasterio.open(paths[item])) for item in indices]
            reference = datasets[half_window]
            samples = _collect_samples(
                datasets,
                reference,
                bands,
                int(cluster_cfg["sample_size"]),
                int(raster_cfg["block_size"]),
                int(cluster_cfg["random_state"]) + target_index,
            )
            model = _fit_model(
                samples,
                n_clusters,
                int(cluster_cfg["batch_size"]),
                int(cluster_cfg["max_iter"]),
                int(cluster_cfg["random_state"]) + target_index,
                previous_centers,
            )
            mapping, aligned_centers = _alignment_mapping(previous_centers, model.cluster_centers_)
            _write_labels(
                datasets,
                reference,
                bands,
                model,
                mapping,
                output / f"{target_date:%Y-%m-%d}.tif",
                int(raster_cfg["block_size"]),
                str(raster_cfg["compression"]),
            )
            previous_centers = aligned_centers

    write_json(
        manifest_path,
        {
            "source": source,
            "dates": [str(item) for item in dates],
            "n_clusters": n_clusters,
            "feature_bands": bands,
            "temporal_window_length": 2 * half_window + 1,
            "sample_size": int(cluster_cfg["sample_size"]),
            "label_encoding": "1..K; 0 is NoData",
            "temporal_alignment": "Hungarian minimum-distance matching",
        },
    )
    return output


def cluster_all(cfg: Mapping[str, object], sources: list[str] | None = None, overwrite: bool = False) -> dict[str, Path]:
    selected = sources or list(cfg["sources"])
    return {source: cluster_source(cfg, source, overwrite=overwrite) for source in selected}
