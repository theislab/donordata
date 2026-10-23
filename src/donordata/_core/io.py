from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import anndata as ad
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata._types import StorageType
from anndata.compat import _read_attr
from anndata.experimental import read_dispatched
from anndata.io import read_elem, sparse_dataset
from mudata import MuData

from donordata._core._utils import build_mudata
from donordata._core._write import EHRDATA_LAYER_PREFIX, EHRDATA_X_KEY
from donordata._core.donordata import DonorData, _copy_matrices
from donordata._core.schema import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION, READABLE_ENCODING_VERSIONS


def _is_matrix_elem(elem_name: str) -> bool:
    return elem_name.endswith("/X") or elem_name.rsplit("/", 1)[0].endswith("/layers")


_ANNDATA_SLOTS = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "raw")


def _read_columns(group, columns) -> pd.DataFrame:
    """Read the index and ``columns`` of a DataFrame stored by anndata."""
    missing = [c for c in columns if c not in group]
    if missing:
        raise KeyError(f"Columns {missing} not found in {group.name}.")
    index_key = group.attrs["_index"]
    frame = pd.DataFrame({c: read_elem(group[c]) for c in columns}, index=pd.Index(read_elem(group[index_key])))
    if index_key != "_index":
        frame.index.name = index_key
    return frame


def _lazy_zarr_callback(func, elem_name: str, elem, iospec, var_columns=None):
    """``read_dispatched`` callback that keeps ``X`` and layers on disk, dense ones as dask arrays."""
    if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
        callback = partial(_lazy_zarr_callback, var_columns=var_columns)
        return ad.AnnData(**{k: read_dispatched(v, callback) for k, v in dict(elem).items() if k in _ANNDATA_SLOTS})
    if var_columns is not None and elem_name.endswith("/var") and iospec.encoding_type == "dataframe":
        return _read_columns(elem, var_columns)
    if _is_matrix_elem(elem_name) and iospec.encoding_type == "array":
        return da.from_zarr(elem)
    if _is_matrix_elem(elem_name) and iospec.encoding_type in ("csr_matrix", "csc_matrix"):
        return sparse_dataset(elem)
    if _is_matrix_elem(elem_name):
        return read_elem(elem)
    return func(elem)


def _read_anndata(group, lazy: bool, var_columns=None):
    if not lazy:
        return read_elem(group)
    return read_dispatched(group, callback=partial(_lazy_zarr_callback, var_columns=var_columns))


def _read_mudata(group: StorageType, lazy: bool = False) -> MuData:
    """Read a MuData stored in a group of a larger HDF5 file or Zarr store."""
    mods_group = group["mod"]
    order = [str(m) for m in _read_attr(mods_group.attrs, "mod-order")] if "mod-order" in mods_group.attrs else []
    names = [m for m in order if m in mods_group] + [m for m in mods_group.keys() if m not in order]
    mods = {name: _read_anndata(mods_group[name], lazy) for name in names}
    slots = ("obs", "var", "obsm", "varm", "obsp", "varp", "uns", "obsmap", "varmap")
    kwargs = {slot: read_elem(group[slot]) for slot in slots if slot in group}
    axis = int(group.attrs.get("axis", 0))
    return build_mudata(lambda: MuData(mods, axis=axis, **kwargs))


def _read_side(group, lazy: bool, var_columns=None):
    encoding = group.attrs.get("encoding-type")
    if encoding == "MuData":
        return _read_mudata(group, lazy)
    if encoding == "anndata":
        return _read_anndata(group, lazy, var_columns)
    if encoding == "dataframe":
        return read_elem(group)
    raise ValueError(f"Unknown encoding type {encoding!r} for {group.name}")


def _read_ehrdata(group, lazy: bool):
    try:
        from ehrdata import EHRData
    except ImportError as e:
        raise ImportError(f"{group.name} holds an EHRData, install ehrdata to read it.") from e
    adata = _read_anndata(group, lazy)
    obsm = dict(adata.obsm)
    X = obsm.pop(EHRDATA_X_KEY, adata.X)
    layers = {name: layer for name, layer in adata.layers.items() if name is not None}
    for name in [k for k in obsm if k.startswith(EHRDATA_LAYER_PREFIX)]:
        layers[name.removeprefix(EHRDATA_LAYER_PREFIX)] = obsm.pop(name)
    return EHRData(
        X=X,
        obs=adata.obs,
        var=adata.var,
        uns=dict(adata.uns),
        obsm=obsm,
        varm=dict(adata.varm),
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
        layers=layers,
        tem=read_elem(group["tem"]),
    )


def _unpack_obsm(group, container) -> None:
    """Turn the arrays that were written for numeric DataFrames in ``obsm`` back into DataFrames."""
    if "obsm_columns" not in group:
        return
    for entry in group["obsm_columns"].keys():
        names = np.asarray(read_elem(group["obsm_columns"][entry])).astype(object)
        columns = pd.Index([n.decode() if isinstance(n, bytes) else str(n) for n in names])
        values = np.asarray(container.obsm[entry])
        container.obsm[entry] = pd.DataFrame(values, index=container.obs_names, columns=columns)


def _read_level(f, key: str, lazy: bool, var_columns=None):
    """Return the object to hand to DonorData for a level, and the AnnData holding a table level."""
    group = f[key]
    if group.attrs.get("donordata-table", False):
        shell = read_elem(group)
        _unpack_obsm(group, shell)
        return shell.obs, shell
    container = _read_ehrdata(group, lazy) if "tem" in group else _read_side(group, lazy, var_columns)
    if isinstance(container, AnnData):
        _unpack_obsm(group, container)
    return container, None


def _read_cells(f, path: str | None, lazy: bool):
    if "spatialdata_attrs" in f["C"].attrs:
        try:
            from spatialdata import read_zarr
        except ImportError as e:
            raise ImportError("The cells of this file are a SpatialData, install spatialdata to read them.") from e
        return read_zarr(f"{path}/C")
    return _read_side(f["C"], lazy)


def _read_dd(f, lazy: bool = False, path: str | None = None, var_columns=None) -> DonorData:
    encoding_type = f.attrs.get("encoding-type")
    if encoding_type is not None and encoding_type != DONORDATA_ENCODING_TYPE:
        raise ValueError(
            f"Expected encoding-type={DONORDATA_ENCODING_TYPE!r} at the top level of a DonorData file, "
            f"found {encoding_type!r}. This file was not written by DonorData."
        )
    encoding_version = f.attrs.get("encoding-version")
    if encoding_version is not None and encoding_version not in READABLE_ENCODING_VERSIONS:
        warnings.warn(
            f"DonorData file has encoding-version={encoding_version!r}, this version of DonorData writes "
            f"{DONORDATA_ENCODING_VERSION!r}. Reading anyway, but the result may be incomplete.",
            stacklevel=3,
        )

    if "D" not in f and "G" in f:
        warnings.warn(
            f"This file stores the donor level in G, as DonorData files of version {encoding_version or '0.1.0'} "
            "did. It is read into D, and writing the object again stores it in the current format.",
            UserWarning,
            stacklevel=3,
        )
    D, D_shell = _read_level(f, "D" if "D" in f else "G", lazy, var_columns)
    S, S_shell = _read_level(f, "S", lazy) if "S" in f else (None, None)
    C = _read_cells(f, path, lazy)
    if D_shell is not None:
        obsm, obsp = dict(D_shell.obsm), dict(D_shell.obsp)
    else:
        obsm = read_elem(f["obsm"]) if "obsm" in f else None
        obsp = read_elem(f["obsp"]) if "obsp" in f else None
    uns = {key: read_elem(f["uns"][key]) for key in f["uns"].keys()} if "uns" in f else {}
    sample_id = f.attrs.get("sample_id")

    dd = DonorData(
        D=D,
        S=S,
        C=C,
        donor_id=f.attrs.get("donor_id", "donor"),
        sample_id=None if sample_id is None else str(sample_id),
        var_dims_to_sync=list(f.attrs.get("var_dims_to_sync", [])),
        uns=uns,
        obsm=obsm,
        obsp=obsp,
        table_name=f.attrs.get("spatialdata_table"),
    )
    if S_shell is not None:
        _copy_matrices(S_shell, dd._S)
    return dd


def read_h5_dd(path: str | Path) -> DonorData:
    """Read a DonorData from an HDF5 file.

    Parameters
    ----------
    path
        File written by :meth:`DonorData.write_h5_dd`.

    Returns
    -------
    DonorData
    """
    with h5py.File(str(path), "r") as f:
        return _read_dd(f)


def read_zarr_dd(path: str | Path, var_columns: list[str] | None = None) -> DonorData:
    """Read a DonorData from a Zarr store.

    ``X`` and layers stay on disk, dense ones as dask arrays and sparse ones as backed sparse matrices,
    so selecting a subset, for example a single cell type, only reads that subset, and
    :meth:`DonorData.aggregate` reads them block by block.

    Parameters
    ----------
    path
        Store written by :meth:`DonorData.write_zarr_dd`.
    var_columns
        Columns of ``D.var`` to read, for example ``["chrom", "pos"]`` for genotypes. All by default.

    Returns
    -------
    DonorData
    """
    return _read_dd(zarr.open(str(path), mode="r"), lazy=True, path=str(path), var_columns=var_columns)


def read_dd(path: str | Path, fmt: str | None = None) -> DonorData:
    """Read a DonorData from HDF5 or Zarr, choosing the format from the extension.

    Parameters
    ----------
    path
        File or store ending in ``.h5`` or ``.zarr``.
    fmt
        ``"h5"`` or ``"zarr"`` to override the detection.

    Returns
    -------
    DonorData

    Examples
    --------
    >>> dd = read_dd("donor_data.dd.h5")
    >>> dd = read_dd("donor_data.dd.zarr")
    """
    path = str(path)
    if fmt is None:
        if path.endswith(".h5"):
            fmt = "h5"
        elif path.endswith(".zarr"):
            fmt = "zarr"
        else:
            raise ValueError("Cannot detect format from file extension. Provide `fmt` as 'h5' or 'zarr'.")
    if fmt == "h5":
        return read_h5_dd(path)
    if fmt == "zarr":
        return read_zarr_dd(path)
    raise ValueError("Unknown format: use 'h5' or 'zarr'.")
