# Data contract

## Common grid

Every HLS, SAR, and MCD43A4 GeoTIFF must have identical:

- raster width and height;
- CRS;
- affine transform and pixel origin;
- spatial resolution and pixel alignment.

`Step_1_validate_inputs.py` stops on the first mismatch. The public code deliberately does not silently resample inputs, because resampling would add an undocumented experiment choice.

## Date parsing

Each filename must contain one of:

- `2020-01-01`;
- `20200101`;
- `2020001` (year and day of year).

One file per collection and date is allowed. Auxiliary collections should be temporally regularized before this repository's Step 2. Actual day distances are used during linear interpolation.

## HLS

- exactly one analysis band is required for the public example;
- default storage: integer surface reflectance multiplied by 10,000;
- default valid raw range: 0–10,000;
- default conversion: `physical_reflectance = raw_value * 0.0001`;
- NoData must be set in GeoTIFF metadata.

Change `raster.hls_scale` and `raster.hls_valid_range_raw` if the data are already physical reflectance or use another encoding.

## SAR

Default order:

| Band | Variable |
|---:|---|
| 1 | VV |
| 2 | VH |
| 3 | RVI |

The Copula predictor vector is VV, VH, RVI, VV−VH, local VV/VH means and standard deviations, and VV/VH Sobel gradient magnitudes. Auxiliary values are not rescaled because the empirical Copula is rank based; nevertheless, use a consistent encoding across dates.

## MCD43A4

The default temporal clustering uses bands 1–6. The Copula reads red and NIR from configurable bands 1 and 2, computes NDVI, and adds local statistics and gradients. If the sample band order differs, change all three YAML fields before validation:

```yaml
clustering:
  mcd43a4_feature_bands: [1, 2, 3, 4, 5, 6]
copula:
  mcd43a4_red_band: 3
  mcd43a4_nir_band: 4
```

## Outputs and units

- preprocessed auxiliary rasters preserve input dtype and NoData encoding;
- cluster rasters are `uint16`, labels `1..K`, NoData `0`;
- Copula rasters are four-band `float32`: q05, q50, q95, conditional variance;
- background, residual, and final fusion rasters are single-band `float32`;
- quantiles/background/final use physical HLS reflectance;
- variance uses reflectance squared.

