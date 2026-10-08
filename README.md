# GMDH Streamflow Modelling

Polynomial-network modelling and sensitivity assessment of monthly hydrological response to climatic and environmental predictors.

## Source code

### `src/gmdh_cluster_sensitivity.py`
Evaluates discharge sensitivity within spatially defined catchment groups. The analysis combines quadratic GMDH polynomial networks with predictor standardization and sample-size controls. Model experiments include the full predictor set, two-variable combinations and leave-one-predictor-out comparisons. Cluster labels are matched to point identifiers, and the workflow produces tabular performance summaries and high-resolution diagnostic plots.

### `src/tiff_line_cleanup.py`
Provides an auxiliary source-code transformation for high-resolution TIFF figure export. The utility updates the `save_plot_as_tiff` function in a target Python script, using an intermediate PNG rendering, RGB conversion, Adobe Deflate compression and output decoding checks. A backup of the original source file is retained.

## Data

Monthly hydrological observations, predictor time series and cluster assignments are read from external Excel datasets. The source code includes configuration variables for file locations and model settings.
