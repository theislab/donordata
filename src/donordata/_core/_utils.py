from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

AGGREGATIONS = ("sum", "mean", "var", "count_nonzero", "median", "min", "max")


def _keep(self, *args, **kwargs) -> None:
    return None


@contextmanager
def _without_update() -> Iterator[None]:
    original = MuData.update
    MuData.update = _keep
    try:
        yield
    finally:
        MuData.update = original


def build_mudata(make: Callable[[], MuData]) -> MuData:
    """Build a MuData with ``make`` exactly as given, without the update mudata runs on construction.

    That update copies modality columns into the global ``obs`` and ``var`` unless pulling is switched
    off, and in mudata 0.3 it fails for modalities that share variable names or name their index
    differently from the global table. Copies and objects read from disk already carry their global
    tables and the maps to the modalities, so nothing has to be recomputed.
    """
    with _without_update():
        return make()


def copy_container(container: AnnData | MuData) -> AnnData | MuData:
    """Copy an AnnData or MuData, keeping the global tables of a MuData as they are."""
    if isinstance(container, MuData):
        return build_mudata(container.copy)
    return container.copy()


def string_labels(values: pd.Series) -> pd.Categorical | np.ndarray:
    """Return identifiers as strings, keeping a categorical encoding when the input has one."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return values.cat.rename_categories(values.cat.categories.astype(str)).array
    return values.astype(str).to_numpy()


def codes_for(values: pd.Series, categories: pd.Index) -> np.ndarray:
    """Position of every value in ``categories``, or ``-1`` when it is not one of them."""
    return np.asarray(pd.Categorical(string_labels(values), categories=categories).codes, dtype=np.int64)


def first_positions(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Row of the first member of every group, or ``-1`` for empty groups."""
    first = np.full(n_groups, -1, dtype=np.int64)
    valid = np.flatnonzero(codes >= 0)
    groups, index = np.unique(codes[valid], return_index=True)
    first[groups] = valid[index]
    return first


def is_constant_within(values: pd.Series, codes: np.ndarray, first: np.ndarray) -> bool:
    """Whether ``values`` takes a single value, or is missing throughout, inside every group."""
    try:
        array = pd.factorize(values)[0]
    except TypeError:
        return False
    member = codes >= 0
    return bool((array[member] == array[first[codes[member]]]).all())


def constant_columns(obs: pd.DataFrame, codes: np.ndarray, first: np.ndarray, exclude=()) -> list[str]:
    """Columns of ``obs`` that are constant within every group."""
    return [c for c in obs.columns if c not in exclude and is_constant_within(obs[c], codes, first)]


def resolve_mask(key, frame: pd.DataFrame, what: str) -> np.ndarray:
    """Turn a selection given as query, callable, labels or boolean mask into a boolean mask over rows."""
    if isinstance(key, str):
        result = frame.eval(key)
    elif callable(key):
        result = key(frame)
    else:
        result = key
    if isinstance(result, pd.Series) and pd.api.types.is_bool_dtype(result.dtype):
        result = result.to_numpy(dtype=bool, na_value=False)
    array = np.asarray(result)
    if array.dtype != bool:
        if array.ndim != 1:
            raise ValueError(f"Cannot select {what} with an array of shape {array.shape}.")
        return frame.index.isin(pd.Index(array).astype(str))
    if array.shape != (len(frame),):
        raise ValueError(f"Boolean mask over {what} has length {array.size}, expected {len(frame)}.")
    return array


def is_full_slice(index) -> bool:
    return isinstance(index, slice) and index == slice(None)


def _is_dask(X) -> bool:
    return type(X).__module__.startswith("dask")


def _in_memory(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, np.ndarray) or sp.issparse(X) or _is_dask(X):
        return X
    if hasattr(X, "to_memory"):
        return X.to_memory()
    return np.asarray(X[...])


def _dense(X) -> np.ndarray:
    if _is_dask(X):
        X = X.compute()
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def group_sum(X, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``X`` within every group. Rows with code ``-1`` are ignored."""
    X = _in_memory(X)
    if _is_dask(X):
        total = np.zeros((n_groups, X.shape[1]))
        start = 0
        for size in X.chunks[0]:
            stop = start + size
            total += group_sum(X[start:stop].compute(), codes[start:stop], n_groups)
            start = stop
        return total
    rows = np.flatnonzero(codes >= 0)
    indicator = sp.csr_matrix((np.ones(rows.size), (codes[rows], rows)), shape=(n_groups, X.shape[0]))
    if not sp.issparse(X):
        X = np.asarray(X, dtype=np.float64)
    result = indicator @ X
    return result.toarray() if sp.issparse(result) else np.asarray(result, dtype=np.float64)


def _transform(X, func: Callable):
    X = _in_memory(X)
    if sp.issparse(X):
        return func(X.astype(np.float64))
    if _is_dask(X):
        return func(X.astype(np.float64))
    return func(np.asarray(X, dtype=np.float64))


def aggregate_rows(X, codes: np.ndarray, n_groups: int, func: str = "mean") -> np.ndarray:
    """Aggregate the rows of ``X`` within groups given by integer ``codes``.

    Parameters
    ----------
    X
        Dense, sparse or dask matrix with one row per element of ``codes``.
    codes
        Group of every row, ``-1`` for rows to ignore.
    n_groups
        Number of groups. Groups without rows are filled with ``NaN``.
    func
        One of ``"sum"``, ``"mean"``, ``"var"`` (with one degree of freedom), ``"count_nonzero"``,
        ``"median"``, ``"min"`` or ``"max"``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_groups, X.shape[1])``.
    """
    if func not in AGGREGATIONS:
        raise ValueError(f"func must be one of {AGGREGATIONS}, got {func!r}.")
    codes = np.asarray(codes, dtype=np.int64)
    counts = np.bincount(codes[codes >= 0], minlength=n_groups).astype(np.float64)[:, None]

    with np.errstate(invalid="ignore", divide="ignore"):
        if func == "sum":
            result = group_sum(X, codes, n_groups)
        elif func == "count_nonzero":
            result = group_sum(_transform(X, lambda A: (A != 0).astype(np.float64)), codes, n_groups)
        elif func in ("mean", "var"):
            mean = group_sum(X, codes, n_groups) / counts
            if func == "mean":
                result = mean
            else:
                squares = group_sum(_transform(X, lambda A: A.multiply(A) if sp.issparse(A) else A**2), codes, n_groups)
                result = np.clip((squares - counts * mean**2) / (counts - 1), 0, None)
        else:
            result = np.full((n_groups, X.shape[1]), np.nan)
            reduce = {"median": np.median, "min": np.min, "max": np.max}[func]
            X = _in_memory(X)
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            for group in np.unique(sorted_codes[sorted_codes >= 0]):
                lo, hi = np.searchsorted(sorted_codes, [group, group + 1])
                result[group] = reduce(_dense(X[np.sort(order[lo:hi])]), axis=0)

    result[counts[:, 0] == 0] = np.nan
    return result


def modality(C: AnnData | MuData, mod: str | None) -> tuple[AnnData, np.ndarray | None]:
    """The AnnData holding the matrices of ``C`` and the rows of ``C`` it covers.

    The returned AnnData has its rows in the order of ``C.obs``. The second value is ``None`` when it
    covers every cell, and otherwise the positions in ``C`` of the cells measured in ``mod``, which
    happens for MuData objects that pair assays measured on different cells.
    """
    if isinstance(C, AnnData):
        if mod is not None:
            raise ValueError("mod can only be used when C is a MuData.")
        return C, None
    if mod is None:
        raise ValueError(f"C is a MuData, pass mod= to choose one of {list(C.mod)}.")
    positions = np.asarray(C.obsmap[mod]).ravel().astype(np.int64) - 1
    adata = C.mod[mod]
    present = positions >= 0
    if present.all():
        if positions.size == adata.n_obs and np.array_equal(positions, np.arange(adata.n_obs)):
            return adata, None
        return adata[positions], None
    rows = np.flatnonzero(present)
    return adata[positions[rows]], rows


def cell_matrix(C: AnnData | MuData, layer: str | None = None, obsm: str | None = None, mod: str | None = None):
    """Matrix of ``C`` selected by ``layer`` or ``obsm``, defaulting to ``X``, and the cells it covers."""
    if obsm is not None and isinstance(C, MuData) and obsm in C.obsm:
        return C.obsm[obsm], None
    adata, rows = modality(C, mod)
    if obsm is not None:
        return adata.obsm[obsm], rows
    if layer is not None and layer != "X":
        return adata.layers[layer], rows
    return adata.X, rows


def column_of(matrix, position: int) -> np.ndarray:
    """One column of a dense, sparse or dask matrix as a dense vector."""
    column = matrix[:, position]
    if _is_dask(column):
        column = column.compute()
    if sp.issparse(column):
        column = column.toarray()
    return np.asarray(column).ravel()


def broadcast(frame: pd.DataFrame, codes: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Repeat the rows of ``frame`` according to ``codes``, keeping column dtypes."""
    return pd.DataFrame({c: frame[c].array.take(codes) for c in frame.columns}, index=index)


def collect(frame: pd.DataFrame, codes: np.ndarray, n_groups: int, index: pd.Index, func, what: str) -> pd.DataFrame:
    """Combine the rows of ``frame`` within groups given by ``codes``.

    Columns that are constant within every group keep their value and dtype. Other numeric columns
    are aggregated with ``func``, and other non numeric columns raise an error.
    """
    first = first_positions(codes, n_groups)
    columns = {}
    for column in frame.columns:
        values = frame[column]
        if is_constant_within(values, codes, first):
            columns[column] = values.array.take(first, allow_fill=True)
        elif pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            member = codes >= 0
            grouped = values[member].groupby(codes[member]).agg(func)
            columns[column] = grouped.reindex(np.arange(n_groups)).to_numpy()
        else:
            raise ValueError(f"'{column}' varies within {what}, use composition for categorical cell annotations.")
    return pd.DataFrame(columns, index=index)


def shallow_with_obs(adata: AnnData | MuData, obs: pd.DataFrame, obsm: dict | None = None) -> AnnData | MuData:
    """Copy of ``adata`` with a new ``obs``, and for an AnnData optionally ``obsm``, sharing all matrices."""
    if isinstance(adata, MuData):
        new = copy_container(adata)
        new.obs = obs
        return new
    new = AnnData(
        X=adata.X,
        obs=obs,
        var=adata.var,
        uns=dict(adata.uns),
        obsm=dict(adata.obsm) if obsm is None else obsm,
        varm=dict(adata.varm),
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
        layers=dict(adata.layers),
    )
    if adata.raw is not None:
        new.raw = adata.raw.to_adata()
    return new
