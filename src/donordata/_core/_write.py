from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata import AnnData
from anndata.io import write_elem
from mudata import MuData

from donordata._core._utils import shallow_with_obs
from donordata._core.schema import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION


def _write_dense_array_zarr_chunked(path: str, X, chunks: tuple[int, int] | None) -> None:
    """Write one dense array to the Zarr array at ``path`` with explicit chunks."""
    if isinstance(X, da.Array):
        X = X.rechunk(chunks) if chunks is not None else X
        da.to_zarr(X, path, overwrite=True)
    else:
        array_chunks = chunks if chunks is not None else tuple(min(dim, 4096) for dim in X.shape)
        z = zarr.open_array(path, mode="w", shape=X.shape, chunks=array_chunks, dtype=X.dtype)
        z[:] = np.asarray(X)
    arr = zarr.open(path, mode="r+")
    arr.attrs["encoding-type"] = "array"
    arr.attrs["encoding-version"] = "0.2.0"


def _write_anndata_zarr_dense_x_chunked(
    f: zarr.Group, key: str, adata: AnnData, zarr_path: str, chunks: tuple[int, int] | None = None
) -> None:
    """Write an AnnData to a Zarr group with ``X`` and dense layers chunked explicitly.

    ``anndata.io.write_elem`` chooses chunks for dense arrays without regard to how the data is laid
    out, which makes writing and reading genome scale matrices far slower than necessary. Everything
    except the dense arrays is written first, with an empty sparse placeholder for ``X``, and each
    dense array is then written on its own with ``chunks``. Without ``chunks`` a dask array keeps its
    own chunks and a plain array is chunked with at most 4096 entries per axis.
    """
    X = adata.X
    layers = {k: v for k, v in adata.layers.items() if k is not None}
    dense_layers = {k: v for k, v in layers.items() if hasattr(v, "ndim") and not sp.issparse(v)}
    sparse_layers = {k: v for k, v in layers.items() if k not in dense_layers}

    placeholder = sp.csr_matrix((adata.n_obs, adata.n_vars), dtype=X.dtype)
    shell = AnnData(
        X=placeholder,
        obs=adata.obs,
        var=adata.var,
        obsm=dict(adata.obsm),
        varm=dict(adata.varm),
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
        layers=dict(sparse_layers),
        uns=dict(adata.uns),
    )
    write_elem(f, key, shell)
    del f[key]["X"]
    _write_dense_array_zarr_chunked(f"{zarr_path}/{key}/X", X, chunks)

    if dense_layers and "layers" not in f[key]:
        f[key].create_group("layers")
    for name, layer in dense_layers.items():
        _write_dense_array_zarr_chunked(f"{zarr_path}/{key}/layers/{name}", layer, chunks)


def write_mudata(f, key: str, mdata: MuData) -> None:
    """Write a MuData into a new group ``key`` of an open HDF5 file or Zarr group.

    Follows the h5mu layout, so the group can be read by mudata, but works for both backends and
    writes the object as it is, without the ``update`` that older mudata versions run while writing.
    """
    group = f.create_group(key)
    write_elem(group, "obs", mdata.obs)
    write_elem(group, "var", mdata.var)
    for slot in ("obsm", "varm", "obsp", "varp"):
        write_elem(group, slot, dict(getattr(mdata, slot)))
    for slot in ("obsmap", "varmap"):
        write_elem(group, slot, {name: np.asarray(value) for name, value in getattr(mdata, slot).items()})
    write_elem(group, "uns", dict(mdata.uns))
    mods = group.create_group("mod")
    for name, modality in mdata.mod.items():
        write_elem(mods, name, modality)
    mods.attrs["mod-order"] = list(mdata.mod)
    group.attrs["encoding-type"] = "MuData"
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["axis"] = int(mdata.axis)


def is_ehrdata(obj) -> bool:
    return any(cls.__name__ == "EHRData" for cls in type(obj).__mro__)


def pack_obsm(obsm) -> tuple[dict, dict[str, np.ndarray]]:
    """Entries of ``obsm`` to write, with numeric DataFrames as arrays, and the column names of those.

    AnnData writes a DataFrame column by column. Pseudobulk profiles have one column per gene, which
    would create tens of thousands of arrays and, in a Zarr store, as many files.
    """
    entries, columns = dict(obsm), {}
    for key, value in entries.items():
        if not isinstance(value, pd.DataFrame) or value.shape[1] == 0 or value.dtypes.nunique() != 1:
            continue
        if pd.api.types.is_numeric_dtype(value.dtypes.iloc[0]):
            entries[key] = value.to_numpy()
            columns[key] = np.asarray(value.columns.astype(str), dtype=str)
    return entries, columns


EHRDATA_X_KEY = "_ed_ondisk_X"
EHRDATA_LAYER_PREFIX = "_ed_ondisk_layers_"


def _dense(array):
    return array.todense() if hasattr(array, "todense") and not sp.issparse(array) else np.asarray(array)


def write_ehrdata(f, key: str, edata, obsm: dict | None = None) -> None:
    """Write an EHRData as ehrdata does, an AnnData group ``key`` with the time axis in ``key/tem``.

    AnnData only stores two dimensional ``X`` and layers, so three dimensional ones are moved to
    ``obsm`` under the keys that ehrdata reserves for them, and the group can be read by anndata.
    """
    obsm = dict(edata.obsm) if obsm is None else dict(obsm)
    X = edata.X
    if getattr(X, "ndim", 2) == 3:
        obsm[EHRDATA_X_KEY] = _dense(X)
        X = None
    layers = {}
    for name, layer in edata.layers.items():
        if name is None:
            continue
        if getattr(layer, "ndim", 2) == 3:
            obsm[f"{EHRDATA_LAYER_PREFIX}{name}"] = _dense(layer)
        else:
            layers[name] = layer
    base = AnnData(
        X=X,
        obs=edata.obs,
        var=edata.var,
        uns=dict(edata.uns),
        obsm=obsm,
        varm=dict(edata.varm),
        obsp=dict(edata.obsp),
        varp=dict(edata.varp),
        layers=layers,
    )
    write_elem(f, key, base)
    write_elem(f[key], "tem", edata.tem)


def _write_level(f, key: str, container, is_table: bool, is_zarr: bool, zarr_path, chunks) -> None:
    if isinstance(container, MuData):
        write_mudata(f, key, container)
        return
    obsm, columns = pack_obsm(container.obsm) if key != "C" else (dict(container.obsm), {})
    if is_table:
        write_elem(f, key, AnnData(obs=container.obs, obsm=obsm, obsp=dict(container.obsp)))
        f[key].attrs["donordata-table"] = True
    elif is_ehrdata(container):
        write_ehrdata(f, key, container, obsm=obsm)
    else:
        if columns:
            container = shallow_with_obs(container, container.obs, obsm=obsm)
        if is_zarr and hasattr(container.X, "ndim") and not sp.issparse(container.X):
            _write_anndata_zarr_dense_x_chunked(f, key, container, zarr_path, chunks=chunks)
        else:
            write_elem(f, key, container)
    if columns:
        group = f[key].require_group("obsm_columns")
        for name, names in columns.items():
            write_elem(group, name, names)


def write_group(dd, f, zarr_path: str | None = None, x_chunks=None) -> None:
    """Write a DonorData into an open HDF5 file or Zarr group."""
    is_zarr = isinstance(f, zarr.Group)

    def chunks_for(key):
        if isinstance(x_chunks, dict):
            return x_chunks.get(key)
        return x_chunks

    _write_level(f, "D", dd._D, dd._D_is_table, is_zarr, zarr_path, chunks_for("D"))
    if dd._S is not None:
        _write_level(f, "S", dd._S, dd._S_is_table, is_zarr, zarr_path, chunks_for("S"))
    if dd._spatial is not None:
        dd.spatialdata.write(f"{zarr_path}/C", update_sdata_path=False)
        f.attrs["spatialdata_table"] = dd._spatial[1]
    else:
        _write_level(f, "C", dd._C, False, is_zarr, zarr_path, chunks_for("C"))

    f.attrs["encoding-type"] = DONORDATA_ENCODING_TYPE
    f.attrs["encoding-version"] = DONORDATA_ENCODING_VERSION
    f.attrs["donor_id"] = dd.donor_id
    if dd.sample_id is not None:
        f.attrs["sample_id"] = dd.sample_id
    f.attrs["var_dims_to_sync"] = list(dd._var_dims_to_sync)

    if dd.uns:
        uns = f.require_group("uns")
        for key, value in dd.uns.items():
            try:
                write_elem(uns, key, value)
            except (TypeError, NotImplementedError):
                uns.create_dataset(key, data=value)
