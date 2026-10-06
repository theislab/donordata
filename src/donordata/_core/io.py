from __future__ import annotations

import warnings
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
from anndata.io import read_elem
from mudata import MuData

from donordata._core._utils import build_mudata
from donordata._core.donordata import DonorData, _copy_matrices
from donordata._core.schema import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION, READABLE_ENCODING_VERSIONS


def _is_matrix_elem(elem_name: str) -> bool:
    return elem_name.endswith("/X") or elem_name.rsplit("/", 1)[0].endswith("/layers")


_ANNDATA_SLOTS = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns", "raw")


def _lazy_zarr_callback(func, elem_name: str, elem, iospec):
    """``read_dispatched`` callback that keeps dense ``X`` and layers as dask arrays."""
    if iospec.encoding_type == "anndata" or elem_name.endswith("/"):
        return ad.AnnData(
            **{k: read_dispatched(v, _lazy_zarr_callback) for k, v in dict(elem).items() if k in _ANNDATA_SLOTS}
        )
    if _is_matrix_elem(elem_name) and iospec.encoding_type == "array":
        return da.from_zarr(elem)
    if _is_matrix_elem(elem_name):
        return read_elem(elem)
    return func(elem)


def _read_anndata(group, lazy: bool):
    return read_dispatched(group, callback=_lazy_zarr_callback) if lazy else read_elem(group)


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


def _read_side(group, lazy: bool):
    encoding = group.attrs.get("encoding-type")
    if encoding == "MuData":
        return _read_mudata(group, lazy)
    if encoding == "anndata":
        return _read_anndata(group, lazy)
    if encoding == "dataframe":
        return read_elem(group)
    raise ValueError(f"Unknown encoding type {encoding!r} for {group.name}")


def _read_ehrdata(group, lazy: bool):
    try:
        from ehrdata import EHRData
    except ImportError as e:
        raise ImportError(f"{group.name} holds an EHRData, install ehrdata to read it.") from e
    return EHRData.from_adata(_read_anndata(group, lazy), tem=read_elem(group["tem"]))


def _unpack_obsm(group, container) -> None:
    """Turn the arrays that were written for numeric DataFrames in ``obsm`` back into DataFrames."""
    if "obsm_columns" not in group:
        return
    for entry in group["obsm_columns"].keys():
        names = np.asarray(read_elem(group["obsm_columns"][entry])).astype(object)
        columns = pd.Index([n.decode() if isinstance(n, bytes) else str(n) for n in names])
        values = np.asarray(container.obsm[entry])
        container.obsm[entry] = pd.DataFrame(values, index=container.obs_names, columns=columns)


def _read_level(f, key: str, lazy: bool):
    """Return the object to hand to DonorData for a level, and the AnnData holding a table level."""
    group = f[key]
    if group.attrs.get("donordata-table", False):
        shell = read_elem(group)
        _unpack_obsm(group, shell)
        return shell.obs, shell
    container = _read_ehrdata(group, lazy) if "tem" in group else _read_side(group, lazy)
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


def _read_dd(f, lazy: bool = False, path: str | None = None) -> DonorData:
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
    D, D_shell = _read_level(f, "D" if "D" in f else "G", lazy)
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


def read_zarr_dd(path: str | Path) -> DonorData:
    """Read a DonorData from a Zarr store.

    Dense ``X`` and dense layers are read lazily as dask arrays, so selecting a subset, for example a
    single cell type, only loads that subset once it is computed.

    Parameters
    ----------
    path
        Store written by :meth:`DonorData.write_zarr_dd`.

    Returns
    -------
    DonorData
    """
    return _read_dd(zarr.open(str(path), mode="r"), lazy=True, path=str(path))


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
