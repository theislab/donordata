from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from donordata._core.donordata import DonorData


def _stack_rows(values: list):
    if all(isinstance(v, pd.DataFrame) for v in values):
        return pd.concat(values, axis=0, join="inner")
    if any(sp.issparse(v) for v in values):
        return sp.vstack([sp.csr_matrix(v) for v in values])
    return np.concatenate([np.asarray(v) for v in values], axis=0)


def _concat_level(containers: list, is_table: bool, key: str, label, keys, join):
    """Concatenate one level of several inputs, returning the object and any table matrices."""
    if is_table:
        table = pd.concat([c.obs for c in containers], axis=0, join=join)
        table.index.name = key
        if label is not None:
            table[label] = pd.Categorical(np.repeat(keys, [c.n_obs for c in containers]), categories=keys)
        obsm = {}
        for name in sorted(set.intersection(*(set(c.obsm.keys()) for c in containers))):
            values = [c.obsm[name] for c in containers]
            if len({v.shape[1] for v in values}) == 1 or all(isinstance(v, pd.DataFrame) for v in values):
                obsm[name] = _stack_rows(values)
        return table, obsm
    adata = ad.concat(containers, join=join, merge="same", label=label, keys=keys if label else None)
    adata.obs.index.name = key
    return adata, None


def concat(
    dds: Sequence[DonorData] | Mapping[str, DonorData],
    *,
    label: str | None = None,
    join: Literal["inner", "outer"] = "inner",
    index_unique: str | None = None,
) -> DonorData:
    """Concatenate cohorts along the donor axis.

    Donor and sample identifiers must be unique across the inputs. Cells are concatenated with
    :func:`anndata.concat`. Donor- and sample-aligned matrices present in every input with matching
    widths are stacked. Pairwise matrices are dropped because they are undefined across cohorts.

    Parameters
    ----------
    dds
        DonorData objects with the same identifiers and the same kind of ``D`` and ``S``, either all
        tables or all AnnData. A mapping uses its keys as cohort labels.
    label
        Donor table column that records the cohort of every donor. Requires a mapping, or uses the
        position of each input otherwise.
    join
        How to combine the variables of ``C`` and of AnnData levels.
    index_unique
        Separator used to make cell names unique by appending the cohort label, see
        :func:`anndata.concat`.

    Returns
    -------
    DonorData
    """
    if isinstance(dds, Mapping):
        keys, dds = list(dds.keys()), list(dds.values())
    else:
        dds = list(dds)
        keys = [str(i) for i in range(len(dds))]
    if not dds:
        raise ValueError("Nothing to concatenate.")
    if any(dd._spatial is not None for dd in dds):
        raise NotImplementedError(
            "concat does not combine the spatial elements of SpatialData cells. Concatenate the SpatialData "
            "objects with spatialdata.concatenate and build a new DonorData from the result."
        )
    donor_id, sample_id = dds[0].donor_id, dds[0].sample_id
    if any(dd.donor_id != donor_id or dd.sample_id != sample_id for dd in dds):
        raise ValueError("All inputs must share the same donor_id and sample_id.")
    kinds = {dd.D_type for dd in dds}
    if len(kinds) != 1 or kinds == {"mudata"}:
        raise ValueError(f"Cannot concatenate D of kinds {sorted(kinds)}, use all tables or all AnnData.")
    if sample_id is not None and len({dd._S_is_table for dd in dds}) != 1:
        raise ValueError("Cannot concatenate sample levels of different kinds, use all tables or all AnnData.")
    if any(not isinstance(dd.C, AnnData) for dd in dds):
        raise ValueError("concat requires C to be an AnnData in every input.")

    for level in ("donor", "sample") if sample_id is not None else ("donor",):
        names = pd.Index(np.concatenate([dd.levels[level].obs_names.to_numpy() for dd in dds]))
        if not names.is_unique:
            duplicated = names[names.duplicated()].unique()[:8].tolist()
            raise ValueError(f"{level.capitalize()} identifiers are not unique across inputs: {duplicated}")

    D, obsm = _concat_level([dd._D for dd in dds], kinds == {"table"}, donor_id, label, keys, join)
    S, sample_obsm = None, None
    if sample_id is not None:
        S, sample_obsm = _concat_level([dd._S for dd in dds], dds[0]._S_is_table, sample_id, None, keys, join)
    C = ad.concat(
        [dd.C for dd in dds],
        join=join,
        merge="same",
        keys=keys if index_unique else None,
        index_unique=index_unique,
    )
    uns = {}
    for dd in reversed(dds):
        uns.update(dd.uns)
    synced = [k for k in dds[0]._var_dims_to_sync if all(k in dd._var_dims_to_sync for dd in dds)]
    merged = DonorData(
        D=D, S=S, C=C, donor_id=donor_id, sample_id=sample_id, obsm=obsm, uns=uns, var_dims_to_sync=synced
    )
    if sample_obsm:
        positions = S.index.get_indexer(merged._S.obs_names)
        for name, value in sample_obsm.items():
            merged._S.obsm[name] = value.iloc[positions] if isinstance(value, pd.DataFrame) else value[positions]
    return merged
