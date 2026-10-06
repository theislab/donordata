from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from mudata import MuData

from donordata._core import _utils

_PREFIXES = {"D": "donor", "S": "sample", "C": "cell"}
_RANK = {"donor": 0, "sample": 1, "cell": 2}


def _matrix_frame(matrix, key: str, index: pd.Index) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.set_axis(index)
    array = np.asarray(_utils._dense(matrix))
    array = array.reshape(array.shape[0], -1)
    return pd.DataFrame(array, index=index, columns=[f"{key}_{i}" for i in range(array.shape[1])])


def _variable(container, key: str, layer: str | None) -> pd.Series | None:
    if container is None or isinstance(container, MuData) or container.n_vars == 0 or key not in container.var_names:
        return None
    matrix = container.X if layer in (None, "X") else container.layers[layer]
    position = container.var_names.get_loc(key)
    return pd.Series(_utils.column_of(matrix, position), index=container.obs_names, name=key)


def _find(dd, key: str, level: str, layer: str | None, mod: str | None):
    """Locate ``key`` and return its level, its values and the cells they cover at the cell level."""
    name, levels = key, [level] + [lv for lv in ("donor", "sample", "cell") if lv != level]
    if ":" in key:
        prefix, rest = key.split(":", 1)
        if prefix in _PREFIXES:
            name, levels = rest, [_PREFIXES[prefix]]
        elif isinstance(dd.C, MuData) and prefix in dd.C.mod:
            name, levels, mod = rest, ["cell"], prefix
    for candidate in levels:
        if candidate not in dd.level_names:
            continue
        container = dd.levels[candidate]
        if candidate != "cell" or not isinstance(container, MuData) or mod is None:
            if name in container.obs.columns:
                return candidate, container.obs[[name]], None
            if name in container.obsm:
                return candidate, _matrix_frame(container.obsm[name], name, container.obs_names), None
        if candidate == "cell":
            if isinstance(container, MuData):
                if mod is None:
                    owners = [m for m in container.mod if name in container.mod[m].var_names]
                    if len(owners) > 1:
                        raise KeyError(f"'{name}' is a variable of several modalities, use 'mod:{name}'.")
                    mod = owners[0] if owners else None
                if mod is not None:
                    adata, rows = _utils.modality(container, mod)
                    values = _variable(adata, name, layer)
                    if values is not None:
                        return candidate, values.to_frame(), rows
            else:
                values = _variable(container, name, layer)
                if values is not None:
                    return candidate, values.to_frame(), None
        elif dd.levels[candidate].n_vars:
            values = _variable(container, name, None)
            if values is not None:
                return candidate, values.to_frame(), None
    raise KeyError(f"'{key}' was not found at any level.")


def get_df(dd, keys: str | Sequence[str], *, level: str, func, layer: str | None, mod: str | None) -> pd.DataFrame:
    """Collect variables from any level of ``dd`` into a table at ``level``."""
    dd._check_level(level, allow_cell=True)
    keys = [keys] if isinstance(keys, str) else list(keys)
    index = dd.levels[level].obs_names
    parts = []
    for key in keys:
        source, frame, rows = _find(dd, key, level, layer, mod)
        if frame.shape[1] == 1:
            frame = frame.set_axis([key], axis=1)
        if source == level:
            moved = frame
        elif _RANK[source] < _RANK[level]:
            if source == "donor" and level == "sample":
                codes = dd.sample_donor_codes
            else:
                codes = dd._codes(source)
            moved = _utils.broadcast(frame, codes, index)
        else:
            if source == "sample":
                codes = dd.sample_donor_codes
            else:
                codes = dd._codes(level)
                if rows is not None:
                    codes = codes[rows]
            moved = _utils.collect(frame, codes, len(index), index, func, f"{level}s")
        if rows is not None and level == "cell":
            full = pd.DataFrame(index=index, columns=moved.columns, dtype=float)
            full.iloc[rows] = moved.to_numpy()
            moved = full
        parts.append(moved.set_axis(index))
    return pd.concat(parts, axis=1)
