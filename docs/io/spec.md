# DonorData specification

Building on top of the {doc}`AnnData specification <anndata:fileformat-prose>` and the MuData specification, this
document describes the `DonorData` on-disk format. HDF5 files and Zarr stores share the same layout. The examples
below use an HDF5 file with three levels.

```{image} ../_static/img/donordata_schema.png
:alt: DonorData on-disk layout
:width: 680px
```

```python-console
>>> import h5py
>>> f = h5py.File("cohort.dd.h5")
>>> list(f.keys())
['C', 'D', 'S']
```

```{contents}
:local:
:depth: 3
```

## `.attrs`

The root group identifies the file and names the identifier columns.

```python-console
>>> dict(f.attrs)
{'donor_id': 'patient_id', 'encoding-type': 'donordata', 'encoding-version': '0.3.0', 'sample_id': 'sample_id', 'var_dims_to_sync': []}
```

`sample_id` is only present when the object has a sample level, and `spatialdata_table` only when the cells come from
a SpatialData, where it names the table of cells. `var_dims_to_sync` lists keys of the donor level
`obsm` whose columns follow the variables of `C`. Unstructured annotations of the object are stored as `uns/<key>`
next to the levels.

## `D`, `S` and `C`

Every level is stored in its own group with the AnnData or MuData encoding, so anndata reads it without DonorData.

```python-console
>>> dict(f["C"].attrs)
{'encoding-type': 'anndata', 'encoding-version': '0.1.0'}
>>> import anndata
>>> anndata.io.read_elem(f["S"])
AnnData object with n_obs × n_vars = 4 × 0
    obs: 'patient_id', 'day'
    obsm: 'pseudobulk'
```

`D` has one row per donor and `C` one row per cell. `S`, when present, has one row per sample and a column named
after `donor_id` with the donor of every sample. Without `S`, `C.obs` has a column named after `donor_id`, and with
`S` a column named after `sample_id`.

### Tables

A donor or sample level given as a table is stored as an AnnData without variables, with the table as `obs` and the
matrices aligned to the units in `obsm` and `obsp`. The group carries the attribute `donordata-table`, and the
reader returns the table itself.

```python-console
>>> dict(f["S"].attrs)
{'donordata-table': True, 'encoding-type': 'anndata', 'encoding-version': '0.1.0'}
```

### MuData

A MuData level is stored in the `.h5mu` layout with `encoding-type` `"MuData"`, its modalities below `mod/`, and
`obsmap` and `varmap`. It is written as it is, so the global `obs` and `var` keep their columns.

### EHRData

An EHRData donor or sample level is stored the way ehrdata stores it: an AnnData group, whose layers may have three dimensions,
with the time axis as a dataframe in `D/tem`. A level with a `tem` entry is read back as an EHRData when ehrdata is
installed.

### SpatialData

Cells given as a SpatialData are stored in `C` in the Zarr format of spatialdata, with their images, labels, points,
shapes and tables, so `spatialdata.read_zarr("cohort.dd.zarr/C")` reads them without DonorData. The group carries
the `spatialdata_attrs` of spatialdata, and the root attribute `spatialdata_table` names the table that holds the
cells. SpatialData can only be stored in Zarr stores.

```python-console
>>> import zarr
>>> f = zarr.open("spatial.dd.zarr")
>>> dict(f.attrs)["spatialdata_table"]
'table'
>>> list(f["C"].keys())
['shapes', 'tables']
```

## `obsm_columns`

AnnData writes a dataframe column by column. A pseudobulk profile with one column per gene would therefore become
tens of thousands of arrays, and in a Zarr store as many files. A numeric dataframe with a single dtype in `obsm` of
the donor or sample level is stored as one array, and its column names in `obsm_columns` of the same group.

```python-console
>>> f["S"]["obsm"]["pseudobulk"]
<HDF5 dataset "pseudobulk": shape (4, 4), type "<f8">
>>> f["S"]["obsm_columns"]["pseudobulk"][:]
array([b'CD14', b'CD3E', b'MS4A1', b'NKG7'], dtype=object)
```

DonorData turns the array back into a dataframe indexed by the units. anndata ignores `obsm_columns` and reads the
array.

## Order and links

Samples are stored grouped by donor in the row order of `D`, and cells grouped by sample, or by donor, in the row
order of the level above. The cells of every unit are therefore one contiguous block of `C`. {func}`donordata.validate`
checks these rules on an object in memory and raises {class}`donordata.DonorDataSchemaError` on the first violation.

## Versions

`encoding-version` follows `MAJOR.MINOR.PATCH`. The `encoding-type` is checked strictly, and an unknown
`encoding-version` is read with a warning.

| Version | Changes |
|---|---|
| `0.3.0` | Levels `D`, `S` and `C`; table levels with `donordata-table`; EHRData levels with `tem`; SpatialData cells; `obsm_columns`; MuData levels in Zarr stores |
| `0.2.0` | Donor level `G`, which could also be a dataframe with `obsm` and `obsp` at the root |
| `0.1.0` | Donor level `G` and cells `C`, both AnnData or MuData |

Files of all three versions are read. The donor level `G` of versions 0.1.0 and 0.2.0 is read into `D` with a
warning.
