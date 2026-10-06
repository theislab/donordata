from __future__ import annotations

import numpy as np

from donordata._core.donordata import DonorData


def _isin(values, wanted) -> np.ndarray:
    wanted = [wanted] if isinstance(wanted, str) or np.isscalar(wanted) else list(np.asarray(wanted).ravel())
    return np.asarray(values.isin(wanted))


def _apply(dd: DonorData, selected: DonorData, inplace: bool, copy: bool) -> DonorData | None:
    if inplace:
        dd._finish(selected, inplace=True)
        return None
    return selected.copy() if copy else selected


def cell_level_obs_filter(
    dd: DonorData,
    cell_level_key: str = None,
    cell_level_values=None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """Keep cells whose ``C.obs[cell_level_key]`` is one of ``cell_level_values``.

    Donors that lose all their cells are dropped.

    Parameters
    ----------
    dd
        Data to filter.
    cell_level_key
        Column of ``C.obs``.
    cell_level_values
        A single value or several values to keep.
    inplace
        Modify ``dd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    DonorData or None

    Examples
    --------
    >>> donordata.pp.cell_level_obs_filter(dd, "celltype", "CD8", inplace=False)
    """
    selected = dd.sel(C_obs=_isin(dd.C.obs[cell_level_key], cell_level_values))
    return _apply(dd, selected, inplace, copy)


def donor_level_obs_filter(
    dd: DonorData,
    donor_level_key: str = None,
    donor_level_values=None,
    *,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """Keep donors whose donor table column ``donor_level_key`` is one of ``donor_level_values``.

    Parameters
    ----------
    dd
        Data to filter.
    donor_level_key
        Column of the donor table.
    donor_level_values
        A single value or several values to keep.
    inplace
        Modify ``dd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    DonorData or None

    Examples
    --------
    >>> donordata.pp.donor_level_obs_filter(dd, "disease_state", "COVID-19", inplace=False)
    """
    selected = dd.sel(D_obs=_isin(dd.obs[donor_level_key], donor_level_values))
    return _apply(dd, selected, inplace, copy)


def donor_level_var_filter(
    dd: DonorData,
    *,
    donor_level_key: str = None,
    donor_level_values=None,
    inplace: bool = True,
    copy: bool = True,
) -> DonorData | None:
    """Keep donor variables whose ``D.var[donor_level_key]`` is one of ``donor_level_values``.

    Only available when ``D`` is an AnnData or MuData.

    Parameters
    ----------
    dd
        Data to filter.
    donor_level_key
        Column of ``D.var``.
    donor_level_values
        A single value or several values to keep.
    inplace
        Modify ``dd`` instead of returning a filtered object.
    copy
        Return a copy instead of views when ``inplace`` is not set.

    Returns
    -------
    DonorData or None

    Examples
    --------
    >>> donordata.pp.donor_level_var_filter(dd, donor_level_key="chrom", donor_level_values="2", inplace=False)
    """
    if dd.D_type == "table":
        raise ValueError("D is a donor table and has no variables to filter.")
    selected = dd.sel(D_var=_isin(dd.D.var[donor_level_key], donor_level_values))
    return _apply(dd, selected, inplace, copy)
