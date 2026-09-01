# Algorithm and implementation correspondence

| Public stage | Reference script supplied by the authors | Public implementation |
|---|---|---|
| Auxiliary interpolation + SG | `Step_2_auxiliary_pseudogap_preprocessing.py` | `jpdfusion/preprocessing.py` |
| SAR clustering | `Step_3_pseudogap_sar_clustering_all_scenarios.py` | `jpdfusion/clustering.py` |
| MCD43A4 clustering | `Step_3_pseudogap_mcd43a4_clustering_all_scenarios.py` | `jpdfusion/clustering.py` |
| SAR–HLS Copula | `Step_4_SAR_HLS_copula_para.py` | `jpdfusion/copula.py` |
| MCD43A4–HLS Copula | `Step_4_MODIS_HLS_copula_para.py` | `jpdfusion/copula.py` |
| Temporal residual compensation | `Step_5_Copula_temporal_smooth.py` | `jpdfusion/residual.py` |

## Step 2

For each auxiliary pixel and band, internal missing values are linearly interpolated using actual acquisition-day distances. Leading and trailing gaps use the nearest valid observation. An SG filter is applied along time, after which observed samples and nearest-filled boundary samples are restored. A pixel with no valid auxiliary observation over the entire series remains NoData.

## Step 3

For target time index `t`, a fixed stack from `t-half_window` through `t+half_window` is created. Indices beyond the series boundary replicate the nearest date. Complete vectors are sampled and fitted with MiniBatchKMeans. Current cluster centers are matched to the preceding date's aligned centers by the Hungarian algorithm. Output label `0` is NoData; valid labels are `1..K`.

## Step 4

The response `Y` is single-band physical HLS reflectance. Predictor features `X` are derived from the chosen auxiliary source. Within each target tile and target-date cluster, pixels are sampled and paired with available HLS observations over an adaptive temporal radius. Each spatial sample has total temporal weight one, preventing pixels with more valid dates from dominating the fit.

Weighted empirical marginal CDFs transform `Y` and each feature to latent normal scores. A shrinkage correlation matrix stabilizes the Gaussian Copula. Conditional latent-normal distributions are transformed back through the inverse HLS marginal to obtain q05, q50, and q95. Conditional variance is estimated from the q16–q84 interval.

The implementation receives `texture_window` explicitly. It never computes feature dimensions at module import using an uninitialized global, avoiding the `NoneType > int` SciPy error found in the research script.

## Step 5

The temporal median of Copula q50 provides a slowly varying background. HLS residuals are computed where observations exist, linearly filled in time, and smoothed by a NaN-aware moving mean. Where `preserve_observed_hls` is true, original observed residuals replace smoothed values, so the final result equals HLS at valid observation pixels. Pixels without any residual support use a zero residual and retain the Copula background.

## Holdout evaluation versus reconstruction

The public default reconstructs a time series from every HLS observation supplied. To conduct leave-one-date-out or pseudo-gap evaluation, create a separate HLS input directory in which the held-out date/pixels are masked, point `paths.hls` to it, and rerun Copula plus residual compensation. This keeps the evaluation mask explicit and auditable instead of embedding experimental deletion in the production pipeline.

