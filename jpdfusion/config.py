"""Configuration loading and path resolution for the reproducible pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULTS: dict[str, Any] = {
    "project_root": ".",
    "paths": {
        "hls": "Data/HLS",
        "sar": "Data/SAR",
        "mcd43a4": "Data/MCD43A4",
        "outputs": "Outputs",
    },
    "sources": ["SAR", "MCD43A4"],
    "raster": {
        "hls_scale": 0.0001,
        "hls_valid_range_raw": [0, 10000],
        "output_nodata": -9999.0,
        "compression": "deflate",
        "block_size": 256,
    },
    "preprocessing": {
        "enabled": True,
        "write_linear": True,
        "write_linear_sg": True,
        "sg_window_length": 7,
        "sg_polyorder": 2,
        "preserve_observations": True,
        "edge_fill": "nearest",
    },
    "clustering": {
        "n_clusters": 15,
        "half_window": 2,
        "sample_size": 200000,
        "batch_size": 4096,
        "max_iter": 100,
        "random_state": 42,
        "sar_feature_bands": [1, 2, 3],
        "mcd43a4_feature_bands": [1, 2, 3, 4, 5, 6],
        "input_product": "Linear_SG",
    },
    "copula": {
        "pixel_sample_per_cluster": 3000,
        "texture_window": 5,
        "h_init": 2,
        "h_max": 7,
        "min_samples": 80,
        "max_samples_per_cluster": 120000,
        "shrinkage_alpha": 0.05,
        "quantiles": [0.05, 0.50, 0.95],
        "tile_height": 256,
        "tile_width": 256,
        "random_state": 42,
        "sar_bands": [1, 2, 3],
        "mcd43a4_red_band": 1,
        "mcd43a4_nir_band": 2,
    },
    "residual": {
        "background_median_window": 3,
        "smooth_window": 5,
        "preserve_observed_hls": True,
        "clip_reflectance": [0.0, 1.0],
    },
}


def _merge(base: dict[str, Any], custom: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in custom.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, apply defaults, and resolve every path from project_root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        custom = yaml.safe_load(stream) or {}
    cfg = _merge(DEFAULTS, custom)

    root_value = Path(str(cfg["project_root"])).expanduser()
    if not root_value.is_absolute():
        root_value = config_path.parent / root_value
    root = root_value.resolve()
    cfg["project_root"] = root
    cfg["config_path"] = config_path

    resolved: dict[str, Path] = {}
    for key, value in cfg["paths"].items():
        candidate = Path(str(value)).expanduser()
        resolved[key] = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    cfg["paths"] = resolved

    sources = [str(source).upper() for source in cfg["sources"]]
    invalid = sorted(set(sources) - {"SAR", "MCD43A4"})
    if invalid:
        raise ValueError(f"Unsupported auxiliary source(s): {invalid}")
    cfg["sources"] = sources
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    pre = cfg["preprocessing"]
    sg_window = int(pre["sg_window_length"])
    sg_order = int(pre["sg_polyorder"])
    if sg_window < 3 or sg_window % 2 == 0:
        raise ValueError("preprocessing.sg_window_length must be an odd integer >= 3")
    if sg_order < 0 or sg_order >= sg_window:
        raise ValueError("preprocessing.sg_polyorder must be >= 0 and < sg_window_length")
    if bool(pre["enabled"]) and not (bool(pre["write_linear"]) or bool(pre["write_linear_sg"])):
        raise ValueError("Step 2 is enabled but neither Linear nor Linear_SG output is requested")
    product = str(cfg["clustering"]["input_product"])
    available_products = {
        "Linear": bool(pre["write_linear"]),
        "Linear_SG": bool(pre["write_linear_sg"]),
    }
    if bool(pre["enabled"]) and (product not in available_products or not available_products[product]):
        raise ValueError("clustering.input_product must name an enabled Step 2 output: Linear or Linear_SG")

    clustering = cfg["clustering"]
    if int(clustering["n_clusters"]) < 2:
        raise ValueError("clustering.n_clusters must be >= 2")
    if int(clustering["half_window"]) < 0:
        raise ValueError("clustering.half_window must be >= 0")

    copula = cfg["copula"]
    texture_window = int(copula["texture_window"])
    if texture_window < 1 or texture_window % 2 == 0:
        raise ValueError("copula.texture_window must be a positive odd integer")
    if int(copula["h_init"]) < 0 or int(copula["h_max"]) < int(copula["h_init"]):
        raise ValueError("Require 0 <= copula.h_init <= copula.h_max")
    qs = [float(q) for q in copula["quantiles"]]
    if len(qs) != 3 or any(q <= 0 or q >= 1 for q in qs) or qs != sorted(qs):
        raise ValueError("copula.quantiles must contain three increasing values in (0, 1)")
    if not abs(qs[1] - 0.5) < 1.0e-12:
        raise ValueError("The second copula quantile must be 0.50 because Step 5 reads it as q50")


def source_key(source: str) -> str:
    normalized = source.upper()
    if normalized == "SAR":
        return "sar"
    if normalized == "MCD43A4":
        return "mcd43a4"
    raise ValueError(f"Unsupported auxiliary source: {source}")


def output_dir(cfg: Mapping[str, Any], stage: str, source: str | None = None) -> Path:
    path = Path(cfg["paths"]["outputs"]) / stage
    return path / source.upper() if source else path
