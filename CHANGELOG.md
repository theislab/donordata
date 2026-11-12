# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] (Unreleased)

First release of DonorData as a package of its own, split from cellink.

### Added

- `DonorData` with a donor level `D`, an optional sample level `S` and the cells `C`, where each level can be a
  table, an `AnnData`, a `MuData`, or for donors and samples an `EHRData`.
- Selection at every level with `filter_donors`, `filter_samples`, `filter_cells`, `filter_donors_by_cells` and
  `subsample_cells`, and `coverage` and `complete_cases` for `MuData` cells with assays on different cells.
- `get_df`, `infer_levels`, `push_obs`, `pull_obs`, `push_obsm` and `to_anndata` to move information between
  levels.
- `aggregate`, `pseudobulk`, `composition` and `cell_weights`.
- `nest`, `flatten` and `collapse` to change the levels.
- `train_test_split` and `kfold` over donors, with stratification and groups.
- `donordata.io` with `read_dd`, `read_h5_dd` and `read_zarr_dd`, and the on-disk format 0.3.0.
- `concat`, `validate` and `register_donordata_namespace`.
- `donordata.ml.MILDataset` with bags per donor or per sample.
- Reading of files in the versions 0.1.0 and 0.2.0 written by cellink. Their donor level `G` is read into `D`
  with a warning.

### Fixed

- `sel`, `dd[...]`, `D_var` and `C_var` match a boolean `pandas.Series` to the axis by name, as `filter_donors`,
  `filter_samples` and `filter_cells` already do. Before, they applied it by position, which selected the wrong
  samples or cells when the Series was built from the input tables, because `DonorData` reorders samples and
  cells. A Series whose index does not contain the names of the axis, such as the default index 0, 1, 2, ..., now
  raises a `ValueError` instead of selecting by position, also when the cells have no real names and are
  called `"0"`, `"1"`, .... Boolean arrays and lists are still applied by position. A missing entry (NA) of a
  mask counts as `False`.
- `dd[mask, ...]` with an ellipsis and a boolean array or Series no longer raises an error.
