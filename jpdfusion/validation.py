"""Step 1: validate public example-data structure before computation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import rasterio

from .config import output_dir, source_key
from .io import assert_same_grid, list_date_rasters, write_json


def validate_inputs(cfg: Mapping[str, object], sources: list[str] | None = None) -> Path:
    selected = sources or list(cfg["sources"])
    hls_items = list_date_rasters(cfg["paths"]["hls"])
    hls_paths = [item[1] for item in hls_items]
    assert_same_grid(hls_paths, minimum_bands=1)
    for path in hls_paths:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f"The public example requires single-band HLS; {path} has {dataset.count} bands")
            if dataset.nodata is None:
                raise ValueError(f"HLS GeoTIFF must declare NoData metadata: {path}")
    report: dict[str, object] = {
        "project_root": cfg["project_root"],
        "hls": {
            "directory": cfg["paths"]["hls"],
            "file_count": len(hls_items),
            "dates": [str(item[0]) for item in hls_items],
            "single_band_required": True,
        },
        "auxiliary_sources": {},
    }
    reference = hls_paths[0]
    for source in selected:
        source = source.upper()
        items = list_date_rasters(cfg["paths"][source_key(source)])
        paths = [item[1] for item in items]
        if source == "SAR":
            required = max(
                max(int(item) for item in cfg["clustering"]["sar_feature_bands"]),
                max(int(item) for item in cfg["copula"]["sar_bands"]),
            )
        else:
            required = max(
                max(int(item) for item in cfg["clustering"]["mcd43a4_feature_bands"]),
                int(cfg["copula"]["mcd43a4_red_band"]),
                int(cfg["copula"]["mcd43a4_nir_band"]),
            )
        assert_same_grid(paths, minimum_bands=required)
        assert_same_grid([reference, *paths], minimum_bands=1)
        with rasterio.open(paths[0]) as dataset:
            count = dataset.count
        for path in paths:
            with rasterio.open(path) as dataset:
                if dataset.nodata is None:
                    raise ValueError(f"Auxiliary GeoTIFF must declare NoData metadata: {path}")
        report["auxiliary_sources"][source] = {
            "directory": cfg["paths"][source_key(source)],
            "file_count": len(items),
            "band_count": count,
            "required_band_count": required,
            "dates": [str(item[0]) for item in items],
            "dates_with_hls": [str(item[0]) for item in items if item[0] in dict(hls_items)],
        }

    report_path = output_dir(cfg, "00_validation") / "input_report.json"
    write_json(report_path, report)
    return report_path
