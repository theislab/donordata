# DonorData specification

Building on top of the {doc}`AnnData specification <anndata:fileformat-prose>` and the MuData specification, this
document describes the `DonorData` on-disk format. HDF5 files and Zarr stores share the same layout, except for
[EHRData levels](#ehrdata). The examples below use an HDF5 file with three levels.

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

`sample_id` is only present when the object has a sample level. `var_dims_to_sync` lists keys of the donor level
`obsm` whose columns follow the variables of `C`. Unstructured annotations of the object are stored as `uns/<key>`
next to the levels.

## `D`, `S` and `C`

Every level is stored in its own group with the AnnData or MuData encoding, so anndata reads it without DonorData.
An EHRData level of a Zarr store keeps its AnnData one group deeper, see [EHRData](#ehrdata).

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

An EHRData donor or sample level is written with {func}`ehrdata.io.write_elem` and read with
{func}`ehrdata.io.read_elem`, so the group has ehrdata's own layout and ehrdata reads it without DonorData. The
layout differs between the formats:

- In an HDF5 file, the group has the `.h5ed` layout: the AnnData slots at the top level of the group, the time axis
  as a dataframe in `tem`, and the attributes `ehrdata-encoding-type` `"ehrdata"` and `ehrdata-encoding-version`
  next to anndata's own. anndata reads the group as an AnnData.
- In a Zarr store, the group has the layout of {func}`ehrdata.io.write_zarr`: the AnnData in the subgroup
  `anndata`, the time axis in `tem`, and the attributes `encoding-type` `"ehrdata"` and `ehrdata-encoding-version`.
  anndata reads the subgroup `anndata`.

```python-console
>>> dict(f["D"].attrs)
{'ehrdata-encoding-type': 'ehrdata', 'ehrdata-encoding-version': '0.3.0', 'encoding-type': 'anndata', 'encoding-version': '0.1.0'}
>>> import ehrdata
>>> edata = ehrdata.io.read_elem(f["D"])
```

Three dimensional `X` and layers, sparse ones included, `object` arrays holding text, and columns that anndata cannot
write, such as datetimes, are stored the way ehrdata stores them. anndata cannot read a group that holds a sparse three dimensional layer.
DonorData reads an EHRData level into memory, also from a Zarr store, with the defaults of
{func}`ehrdata.io.read_elem`.

## `obsm_columns`

AnnData writes a dataframe column by column. A pseudobulk profile with one column per gene would therefore become
tens of thousands of arrays, and in a Zarr store as many files. A numeric dataframe with a single dtype in `obsm` of
the donor or sample level is stored as one array, and its column names in `obsm_columns` next to `obsm`. For
an EHRData level of a Zarr store that is `anndata/obsm_columns`.

```python-console
>>> f["S"]["obsm"]["pseudobulk"]
<HDF5 dataset "pseudobulk": shape (4, 4), type "<f8">
>>> f["S"]["obsm_columns"]["pseudobulk"][:]
array([b'CD14', b'CD3E', b'MS4A1', b'NKG7'], dtype=object)
```

DonorData turns the array back into a dataframe indexed by the units. anndata and ehrdata ignore `obsm_columns`
and read the array.

## Order and links

Samples are stored grouped by donor in the row order of `D`, and cells grouped by sample, or by donor, in the row
order of the level above. The cells of every unit are therefore one contiguous block of `C`. {func}`donordata.validate`
checks these rules on an object in memory and raises {class}`donordata.DonorDataSchemaError` on the first violation.

## Versions

`encoding-version` follows `MAJOR.MINOR.PATCH`. The `encoding-type` is checked strictly, and an unknown
`encoding-version` is read with a warning.

| Version | Changes |
|---|---|
| `0.3.0` | Levels `D`, `S` and `C`; table levels with `donordata-table`; EHRData levels in ehrdata's layout; `obsm_columns`; MuData levels in Zarr stores |
| `0.2.0` | Donor level `G`, which could also be a dataframe with `obsm` and `obsp` at the root |
| `0.1.0` | Donor level `G` and cells `C`, both AnnData or MuData |

Files of all three versions are read. The donor level `G` of versions 0.1.0 and 0.2.0 is read into `D` with a
warning.
