# Public example data

Copy the real, temporally regularized example GeoTIFFs here before publishing:

```text
Data/HLS/
Data/SAR/
Data/MCD43A4/
```

The distributed HLS example is single-band. See `../docs/data_format.md` for the complete band, date, grid, scaling, and NoData contract.

## Dataset record — complete before release

- Dataset title: `REPLACE_WITH_DATASET_TITLE`
- DOI: `REPLACE_WITH_DATASET_DOI`
- Spatial region/tile: `REPLACE_WITH_TILE`
- Temporal range: `REPLACE_WITH_START_AND_END_DATES`
- HLS band: `REPLACE_WITH_BAND_NAME`
- License/terms: `REPLACE_WITH_DATA_LICENSE`
- Source-product acknowledgements: `REPLACE_WITH_HLS_SAR_MCD43A4_ACKNOWLEDGEMENTS`

Run `python scripts/create_data_manifest.py` after copying the files. Commit the resulting `data_manifest.csv` so users can verify byte-identical downloads.

The repository's MIT license applies to software only. It does not replace or override the licenses of HLS, SAR, MCD43A4, or derived example data.

