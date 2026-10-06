from __future__ import annotations

import numpy as np
import pandas as pd

DONORDATA_ENCODING_TYPE = "donordata"
DONORDATA_ENCODING_VERSION = "0.3.0"
READABLE_ENCODING_VERSIONS = ("0.1.0", "0.2.0", "0.3.0")


class DonorDataSchemaError(ValueError):
    """Raised by :func:`validate` when a DonorData breaks one of its structural invariants."""


def _check_grouping(child_ids: pd.Series, parents: pd.Index, child: str, parent: str) -> None:
    ids = child_ids.astype(str)
    unknown = sorted(set(ids) - set(parents))
    if unknown:
        raise DonorDataSchemaError(f"{child} refer to {parent} that are not present: {unknown[:8]}")
    codes = pd.Categorical(ids, categories=parents).codes
    empty = parents[np.bincount(codes, minlength=len(parents)) == 0].tolist()
    if empty:
        raise DonorDataSchemaError(f"{parent.capitalize()} without any {child}: {empty[:8]}")
    if np.any(np.diff(codes) < 0):
        raise DonorDataSchemaError(
            f"{child.capitalize()} are not grouped by {parent[:-1]} in the order of the {parent}."
        )


def validate(dd) -> bool:
    """Check that the levels of a DonorData are consistent.

    Verifies that donor and sample identifiers are unique, that every cell belongs to a known sample
    or donor and every sample to a known donor, that no donor or sample is empty, and that cells and
    samples are grouped in the row order of the level above. When ``C.obs`` also names donors next to
    samples, both must agree. These are the invariants that construction and every public setter
    establish, so a failure means the object was modified through private attributes or in place.

    Parameters
    ----------
    dd
        The DonorData to check.

    Returns
    -------
    bool
        ``True`` when every check passes.

    Raises
    ------
    DonorDataSchemaError
        Naming the first check that failed.
    """
    donors = dd.obs_names
    if not donors.is_unique:
        duplicated = donors[donors.duplicated()].unique().tolist()
        raise DonorDataSchemaError(f"Donor identifiers in D are not unique: {duplicated[:8]}")
    if dd.S is None:
        _check_grouping(dd.C.obs[dd.donor_id], donors, "cells", "donors")
        return True

    samples = dd.levels["sample"].obs_names
    if not samples.is_unique:
        duplicated = samples[samples.duplicated()].unique().tolist()
        raise DonorDataSchemaError(f"Sample identifiers in S are not unique: {duplicated[:8]}")
    _check_grouping(dd.levels["sample"].obs[dd.donor_id], donors, "samples", "donors")
    _check_grouping(dd.C.obs[dd.sample_id], samples, "cells", "samples")
    if dd.donor_id in dd.C.obs.columns:
        sample_donors = dd.levels["sample"].obs[dd.donor_id].astype(str).to_numpy()
        expected = sample_donors[pd.Categorical(dd.C.obs[dd.sample_id].astype(str), categories=samples).codes]
        if not np.array_equal(dd.C.obs[dd.donor_id].astype(str).to_numpy(), expected):
            raise DonorDataSchemaError("C.obs names donors that disagree with the donors of the samples.")
    return True
