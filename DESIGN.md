# Designing `DonorData`

This document outlines design considerations and some technical details about the `DonorData` implementation.

## Levels as containers

`DonorData` stores data at several levels: donors, optionally samples, and cells. Every level is a fully functional
`AnnData` or `MuData` object that can be used on its own, and the levels are linked by identifiers instead of being
merged into one matrix. This follows the container approach of `MuData`, which keeps every modality as an `AnnData`.
A donor table plays a similar role to the global `obs` of a `MuData`, one level up. A donor can be any unit that groups cells, for example a patient, a target gene in a CRISPR screen or a compound in a drug screen.

The other core scverse formats fit the same scheme. A `SpatialData`, the scverse format for spatial omics, can hold
the cells: its table of cells becomes `C`, so every operation on cells works on spatial data as well, and its images,
labels, points and shapes stay attached and come back together with the current table as `dd.spatialdata`. An
`EHRData` from ehrdata, an `AnnData` with a time axis, can be the donor or the sample level, for example to hold the
clinical record of every patient next to the cells.

## The sample level

Some cohorts have several samples per donor, for example blood drawn on two days, two tissues or several
stimulation conditions. Without a sample level there are two options, and both lose information. With the patient
as the donor, the samples are pooled and can no longer be compared. With the sample as the donor, patient covariates
are repeated for every sample, and a split can put two samples of the same patient into training and test data.

The optional sample level `S` sits between donors and cells. `D` has one row per donor, `S` one row per sample with
the donor of every sample, and `C` the cells with the sample of every cell. Analyses can use either level, and splits
are always drawn over donors. Without `S`, cells are linked to donors directly and nothing else changes.

## Order of units

Samples are stored grouped by donor in the row order of `D`, and cells grouped by sample, or by donor, in the row
order of the level above. The cells of a unit are therefore one contiguous block of `C`, which makes selecting the
cells of a unit, building bags of cells for multiple instance learning and aggregating cells to units simple slicing
or a single pass over integer codes. The order is restored on construction and whenever a level is replaced, and
`validate` checks it.

## Storage

A `DonorData` is written with one group per level, `D`, `S` and `C`, each with the `AnnData` or `MuData` encoding, so
every level can be read with AnnData alone. A `SpatialData` level is written in its own Zarr format, so it can be
read with spatialdata alone. See the [specification](docs/io/spec.md) for the full layout.
