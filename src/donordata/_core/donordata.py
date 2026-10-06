from __future__ import annotations

import copy as _copy
import logging
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from mudata import MuData

from donordata._core import _spatial, _utils
from donordata._core._repr import render
from donordata._core._write import write_group
from donordata._core.fields import DAnn

logger = logging.getLogger(__name__)

Selection = str | Callable | Sequence | np.ndarray | pd.Series | pd.Index
Level = Literal["donor", "sample", "cell"]
_RANK = {"donor": 0, "sample": 1, "cell": 2}


class DonorData:
    """A multiscale data object.

    :class:`~donordata.DonorData` stores single-cell data :attr:`C` together with data on the donors
    the cells come from, :attr:`D`, and optionally on the samples in between, :attr:`S`. ``D`` holds
    one row per donor and can be a DataFrame of donor covariates, an :class:`~anndata.AnnData` such as
    genotypes, bulk omics or clinical measurements, an EHRData with a time axis, or a
    :class:`~mudata.MuData`. ``S`` holds one row per sample, such as a time point, tissue or replicate,
    and its ``donor_id`` column names the donor of every sample. ``C`` holds the cells as an AnnData or
    MuData, with the sample of every cell in ``C.obs[sample_id]``, or its donor in
    ``C.obs[donor_id]`` when there is no sample level. A SpatialData can be passed as ``C``: its table of
    cells becomes ``C``, and its images, labels, points and shapes stay available in
    :attr:`spatialdata`.

    On construction, and whenever a level is replaced, only units connected across all levels are
    kept: donors with samples, samples with cells and cells with a known sample or donor. Samples are
    grouped by donor in the row order of ``D`` and cells by sample, or by donor, in the row order of
    the level above. Matrices aligned to donors or samples, such as embeddings, compositions,
    pseudobulk profiles or distances, live in ``obsm`` and ``obsp`` of their level and are subset
    with it.

    A donor can be any unit that groups cells, for example a patient, a target gene in a CRISPR
    screen or a compound in a drug screen.

    Parameters
    ----------
    D
        Donor-level data with one row per donor. Donor identifiers are taken from the ``donor_id``
        column when there is one and from the index otherwise. Optional when ``S`` is given, in which
        case it is a table of the donors named in ``S[donor_id]`` without further columns.
    C
        Cell-level data as an AnnData, MuData or SpatialData. ``C.obs`` holds the sample of every
        cell in ``sample_id`` when ``S`` is given, and the donor of every cell in ``donor_id``
        otherwise. For a SpatialData these columns are read from the table of cells.
    donor_id
        Name of the donor identifier.
    S
        Optional sample-level data with one row per sample and the donor of every sample in a
        ``donor_id`` column. Sample identifiers are taken from the ``sample_id`` column when there is
        one and from the index otherwise.
    sample_id
        Name of the sample identifier. Required with ``S``.
    var_dims_to_sync
        Keys of donor-level ``obsm`` whose columns are kept in sync with ``C.var_names``.
    uns
        Unstructured annotations of the object.
    obsm
        Donor-aligned matrices to add to :attr:`obsm`, with rows in the row order of ``D``.
    obsp
        Donor-by-donor matrices to add to :attr:`obsp`, with rows and columns in the row order of ``D``.
    table_name
        Table of a SpatialData ``C`` that holds the cells. Defaults to its only table, or to the
        table named ``"table"``.

    Examples
    --------
    >>> dd = DonorData(D=patients, C=adata, donor_id="patient")
    >>> dd = DonorData(S=samples, C=adata, donor_id="patient", sample_id="sample")
    >>> dd = DonorData.from_anndata(adata, donor_id="patient", sample_id="sample")
    >>> dd.filter_donors("age > 60").filter_donors_by_cells(20, groupby="cell_type")
    >>> dd.get_df(["age", "cell_type", "CD14"], level="cell")
    """

    _spatial = None

    def __init__(
        self,
        *,
        D: pd.DataFrame | AnnData | MuData | None = None,
        C: AnnData | MuData,
        donor_id: str = DAnn.donor,
        S: pd.DataFrame | AnnData | None = None,
        sample_id: str | None = None,
        var_dims_to_sync: list[str] | None = None,
        uns: dict | None = None,
        obsm: dict | None = None,
        obsp: dict | None = None,
        table_name: str | None = None,
    ):
        if _spatial.is_spatialdata(C):
            name, table = _spatial.table_of(C, table_name)
            self._spatial = (C, name)
            C = table
        if (S is None) != (sample_id is None):
            raise TypeError("S and sample_id must be given together.")
        if D is None and S is None:
            raise TypeError("Pass the donor-level data as D, or a sample table S that names the donor of every sample.")
        if not isinstance(C, AnnData | MuData):
            raise TypeError(f"C must be an AnnData, MuData or SpatialData object, got {type(C).__name__}.")
        self.donor_id = donor_id
        self.sample_id = sample_id
        cell_key = sample_id if S is not None else donor_id
        if cell_key not in C.obs.columns:
            raise ValueError(f"'{cell_key}' not found in C.obs")
        if S is not None:
            S, S_is_table = self._prepare_level(S, sample_id)
            if donor_id not in S.obs.columns:
                raise ValueError(f"'{donor_id}' not found in the sample table, it must name the donor of every sample.")
        else:
            S_is_table = False
        if D is None:
            names = pd.unique(pd.Series(_utils.string_labels(S.obs[donor_id])).dropna())
            D = pd.DataFrame(index=pd.Index(np.asarray(names, dtype=str), name=donor_id))
        D, D_is_table = self._prepare_level(D, donor_id)
        for key, value in (obsm or {}).items():
            D.obsm[key] = value
        for key, value in (obsp or {}).items():
            D.obsp[key] = value
        self._var_dims_to_sync = [] if var_dims_to_sync is None else list(var_dims_to_sync)
        self.uns = {} if uns is None else uns
        self._match(D, S, C, D_is_table, S_is_table)

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData | MuData,
        donor_id: str = DAnn.donor,
        *,
        sample_id: str | None = None,
        columns: Sequence[str] | None = None,
        keep_in_cells: bool = False,
        table_name: str | None = None,
    ) -> DonorData:
        """Split a cell-level object into tables for donors, optionally samples, and the cells.

        Columns of ``adata.obs`` that take a single value within every donor are moved into a donor
        table, which becomes ``D``. With ``sample_id``, columns that are constant within samples but
        not within donors go to a sample table ``S``. This turns metadata that is repeated on every
        cell into one row per unit, so it can be filtered, edited and stored once.

        Parameters
        ----------
        adata
            Cell-level data with the donor of every cell in ``adata.obs[donor_id]``. For a
            SpatialData the columns are read from its table of cells.
        donor_id
            Column of ``adata.obs`` with donor identifiers.
        sample_id
            Column of ``adata.obs`` with sample identifiers. Every sample must belong to one donor.
        columns
            Columns to move. By default every column that is constant within donors, or within
            samples. Columns given explicitly must be constant within samples, or within donors
            without ``sample_id``.
        keep_in_cells
            Keep the moved columns in ``C.obs`` as well. Without it ``C`` is a shallow copy of
            ``adata`` with the columns removed, sharing all matrices with ``adata``.
        table_name
            Table of a SpatialData ``adata`` that holds the cells.

        Returns
        -------
        DonorData
            With units sorted by their identifier, or in category order for categorical columns.
        """
        if _spatial.is_spatialdata(adata):
            name, table = _spatial.table_of(adata, table_name)
            dd = cls.from_anndata(table, donor_id, sample_id=sample_id, columns=columns, keep_in_cells=keep_in_cells)
            dd._spatial = (adata, name)
            return dd
        obs = adata.obs
        for key in (donor_id, sample_id):
            if key is not None and key not in obs.columns:
                raise ValueError(f"'{key}' not found in adata.obs")
        donor_codes, donors = pd.factorize(pd.Series(_utils.string_labels(obs[donor_id])), sort=True)
        donor_first = _utils.first_positions(donor_codes, len(donors))
        ids = ({donor_id} if sample_id is None else {donor_id, sample_id}) | _spatial.linking_columns(adata)
        if sample_id is None:
            unit_codes, unit_first = donor_codes, donor_first
        else:
            unit_codes, samples = pd.factorize(pd.Series(_utils.string_labels(obs[sample_id])), sort=True)
            unit_first = _utils.first_positions(unit_codes, len(samples))
            if not _utils.is_constant_within(obs[donor_id], unit_codes, unit_first):
                raise ValueError(f"Some samples in '{sample_id}' contain cells of several donors.")

        if columns is None:
            candidates = [c for c in obs.columns if c not in ids]
            unit_columns = [c for c in candidates if _utils.is_constant_within(obs[c], unit_codes, unit_first)]
        else:
            unit_columns = [c for c in columns if c not in ids]
            varying = [c for c in unit_columns if not _utils.is_constant_within(obs[c], unit_codes, unit_first)]
            if varying:
                level = "samples" if sample_id is not None else "donors"
                raise ValueError(f"Columns {varying} vary within {level} and cannot be moved out of the cells.")

        if sample_id is None:
            donor_columns, sample_columns = unit_columns, []
        else:
            donor_columns = [c for c in unit_columns if _utils.is_constant_within(obs[c], donor_codes, donor_first)]
            sample_columns = [c for c in unit_columns if c not in donor_columns]

        table = obs.iloc[donor_first][list(donor_columns)].copy()
        table.index = pd.Index(np.asarray(donors, dtype=str), name=donor_id)
        moved = list(donor_columns) + list(sample_columns)
        cells = adata if keep_in_cells or not moved else _utils.shallow_with_obs(adata, obs.drop(columns=moved))
        if sample_id is None:
            return cls(D=table, C=cells, donor_id=donor_id)
        sample_table = obs.iloc[unit_first][[donor_id, *sample_columns]].copy()
        sample_table[donor_id] = np.asarray(_utils.string_labels(sample_table[donor_id]), dtype=str)
        sample_table.index = pd.Index(np.asarray(samples, dtype=str), name=sample_id)
        return cls(D=table, S=sample_table, C=cells, donor_id=donor_id, sample_id=sample_id)

    def _prepare_level(self, value, key: str) -> tuple[AnnData | MuData, bool]:
        if isinstance(value, pd.DataFrame):
            table = value
            if key in table.columns and table.index.name != key:
                table = table.set_index(key, drop=False)
            elif table.index.name is None:
                table = table.rename_axis(key)
            elif table.index.name != key:
                raise ValueError(f"'{key}' must be a column of the table or the name of its index")
            table = table.copy()
            table.index = table.index.astype(str)
            return AnnData(obs=table), True

        if not isinstance(value, AnnData | MuData):
            raise TypeError(f"Expected a DataFrame, AnnData or MuData object, got {type(value).__name__}.")
        if isinstance(value, MuData) and value.obs.index.name is None:
            for mod in value.mod.values():
                if mod.obs.index.name == key:
                    value.obs.index.name = key
                    break
        if key in value.obs.columns and value.obs.index.name != key:
            value.obs = value.obs.set_index(key, drop=False)
        elif value.obs.index.name is None:
            value.obs.index.name = key
        elif value.obs.index.name != key:
            raise ValueError(f"'{key}' must be in obs or set as its index")
        return value, False

    def _match(self, D, S, C, D_is_table: bool, S_is_table: bool) -> None:
        if not D.obs_names.is_unique:
            raise ValueError("Donor identifiers in D must be unique.")
        if S is None:
            cell_codes = _utils.codes_for(C.obs[self.donor_id], D.obs_names)
            present = np.bincount(cell_codes[cell_codes >= 0], minlength=D.n_obs) > 0
            if not present.any():
                self._raise_empty(D, None, C)
            if not present.all():
                D = D[present]
                new_position = np.cumsum(present) - 1
                cell_codes = np.where(cell_codes >= 0, new_position[np.maximum(cell_codes, 0)], -1)
        else:
            if not S.obs_names.is_unique:
                raise ValueError("Sample identifiers in S must be unique.")
            sample_donors = _utils.codes_for(S.obs[self.donor_id], D.obs_names)
            cell_codes = _utils.codes_for(C.obs[self.sample_id], S.obs_names)
            linked = cell_codes >= 0
            linked[linked] = sample_donors[cell_codes[linked]] >= 0
            cell_codes = np.where(linked, cell_codes, -1)
            sample_present = (np.bincount(cell_codes[linked], minlength=S.n_obs) > 0) & (sample_donors >= 0)
            donor_present = np.bincount(sample_donors[sample_present], minlength=D.n_obs) > 0
            if not donor_present.any():
                self._raise_empty(D, S, C)
            if not donor_present.all():
                D = D[donor_present]
                new_position = np.cumsum(donor_present) - 1
                sample_donors = np.where(sample_donors >= 0, new_position[np.maximum(sample_donors, 0)], -1)
            samples = np.flatnonzero(sample_present)
            sample_order = samples[np.argsort(sample_donors[samples], kind="stable")]
            if sample_order.size != S.n_obs or not np.array_equal(sample_order, np.arange(S.n_obs)):
                S = S[sample_order]
            new_position = np.full(sample_present.size, -1)
            new_position[sample_order] = np.arange(sample_order.size)
            cell_codes = np.where(cell_codes >= 0, new_position[np.maximum(cell_codes, 0)], -1)
            if S_is_table and S.is_view:
                S = S.copy()
        if D_is_table and D.is_view:
            D = D.copy()

        cells = np.flatnonzero(cell_codes >= 0)
        order = cells[np.argsort(cell_codes[cells], kind="stable")]
        if order.size != C.n_obs or not np.array_equal(order, np.arange(C.n_obs)):
            C = C[order]

        self._D, self._S, self._C = D, S, C
        self._D_is_table, self._S_is_table = D_is_table, S_is_table
        if S is not None and self.donor_id in C.obs.columns:
            expected = D.obs_names[_utils.codes_for(S.obs[self.donor_id], D.obs_names)]
            stated = pd.Index(_utils.string_labels(C.obs[self.donor_id]), dtype=str)
            if not np.array_equal(stated.to_numpy(), expected.to_numpy()[self.sample_codes]):
                raise ValueError(f"C.obs['{self.donor_id}'] disagrees with the donor of the sample of some cells.")

    def _raise_empty(self, D, S, C) -> None:
        if D.n_obs == 0 or C.n_obs == 0 or (S is not None and S.n_obs == 0):
            raise ValueError("The selection leaves no donors with cells.")
        if self.sample_id is None:
            raise ValueError(
                f"No overlapping donors between D and C, make sure C.obs['{self.donor_id}'] holds the donor identifiers of D."
            )
        raise ValueError(
            f"No overlapping donors across the levels, make sure C.obs['{self.sample_id}'] holds the sample identifiers "
            f"of S and S['{self.donor_id}'] the donor identifiers of D."
        )

    def update(self) -> None:
        """Synchronize the levels again after editing identifier columns in place.

        Construction and the setters keep the levels aligned. Changes made directly to
        ``C.obs[donor_id]``, ``C.obs[sample_id]`` or the donor column of the sample table bypass them,
        and this method applies the same matching once more.
        """
        self._match(self._D, self._S, self._C, self._D_is_table, self._S_is_table)

    def _derive(self, D, S, C, var_dims_to_sync=None) -> DonorData:
        new = DonorData.__new__(DonorData)
        new.donor_id = self.donor_id
        new.sample_id = self.sample_id
        new._var_dims_to_sync = list(self._var_dims_to_sync if var_dims_to_sync is None else var_dims_to_sync)
        new.uns = dict(self.uns)
        new._spatial = self._spatial
        new._match(D, S, C, self._D_is_table, self._S_is_table)
        return new

    def _finish(self, new: DonorData, inplace: bool) -> DonorData | None:
        if not inplace:
            return new
        self._D, self._S, self._C = new._D, new._S, new._C
        self._D_is_table, self._S_is_table = new._D_is_table, new._S_is_table
        self._var_dims_to_sync = new._var_dims_to_sync
        return None

    def _actualize(self, level: Level) -> None:
        attribute = {"donor": "_D", "sample": "_S", "cell": "_C"}[level]
        container = getattr(self, attribute)
        if container.is_view:
            setattr(self, attribute, _utils.copy_container(container))

    def _actualize_D(self) -> None:
        self._actualize("donor")

    def _actualize_C(self) -> None:
        self._actualize("cell")

    def copy(self) -> DonorData:
        """Return a deep copy."""
        new = DonorData.__new__(DonorData)
        new.donor_id = self.donor_id
        new.sample_id = self.sample_id
        new._var_dims_to_sync = list(self._var_dims_to_sync)
        new._D_is_table, new._S_is_table = self._D_is_table, self._S_is_table
        new._D = _utils.copy_container(self._D)
        new._S = None if self._S is None else self._S.copy()
        new._C = _utils.copy_container(self._C)
        new.uns = _copy.deepcopy(self.uns)
        new._spatial = self._spatial
        return new

    @property
    def D(self) -> pd.DataFrame | AnnData | MuData:
        """Donor-level data. A DataFrame when ``D`` was given as a donor table."""
        return self._D.obs if self._D_is_table else self._D

    @D.setter
    def D(self, value: pd.DataFrame | AnnData | MuData) -> None:
        D, is_table = self._prepare_level(value, self.donor_id)
        if is_table:
            _carry_over_matrices(self._D, D)
        self._match(D, self._S, self._C, is_table, self._S_is_table)

    @property
    def S(self) -> pd.DataFrame | AnnData | None:
        """Sample-level data, a DataFrame when given as a table, or ``None`` without a sample level."""
        if self._S is None:
            return None
        return self._S.obs if self._S_is_table else self._S

    @S.setter
    def S(self, value: pd.DataFrame | AnnData) -> None:
        if self.sample_id is None:
            raise ValueError("This DonorData has no sample level, use nest() to add one.")
        S, is_table = self._prepare_level(value, self.sample_id)
        if self.donor_id not in S.obs.columns:
            raise ValueError(f"'{self.donor_id}' not found in the sample table.")
        if is_table:
            _carry_over_matrices(self._S, S)
        self._match(self._D, S, self._C, self._D_is_table, is_table)

    @property
    def C(self) -> AnnData | MuData:
        """Cell-level data, grouped by sample or donor in the row order of the level above."""
        return self._C

    @C.setter
    def C(self, value: AnnData | MuData) -> None:
        if _spatial.is_spatialdata(value):
            name, table = _spatial.table_of(value, None)
            self._spatial = (value, name)
            value = table
        if not isinstance(value, AnnData | MuData):
            raise TypeError(f"C must be an AnnData or MuData object, got {type(value).__name__}.")
        key = self.sample_id or self.donor_id
        if key not in value.obs.columns:
            raise ValueError(f"'{key}' not found in C.obs")
        self._match(self._D, self._S, value, self._D_is_table, self._S_is_table)

    @property
    def spatialdata(self):
        """The SpatialData the cells come from, with the current ``C`` as its table of cells.

        ``None`` when ``C`` was not given as a SpatialData. Images, labels, points and shapes are
        shared with the SpatialData that was passed, so selecting cells changes the table but keeps
        the spatial elements as they are.
        """
        if self._spatial is None:
            return None
        sdata, name = self._spatial
        self._actualize("cell")
        return _spatial.with_table(sdata, name, self._C)

    @property
    def D_type(self) -> Literal["table", "anndata", "mudata"]:
        """Kind of donor-level data held in ``D``."""
        if self._D_is_table:
            return "table"
        return "mudata" if isinstance(self._D, MuData) else "anndata"

    @property
    def level_names(self) -> tuple[str, ...]:
        """Levels of this object from the coarsest to the cells."""
        return ("donor", "sample", "cell") if self._S is not None else ("donor", "cell")

    @property
    def levels(self) -> dict[str, AnnData | MuData]:
        """Every level as an AnnData-like object with ``obs``, ``obsm`` and ``obsp``.

        Table levels are shown through the AnnData that holds them, so donor- and sample-aligned
        matrices of any level are reached in the same way, for example
        ``dd.levels["sample"].obsm["X_pca"]``. Tools that expect a sample-level AnnData can take
        ``dd.levels["donor"]`` directly.
        """
        result = {"donor": self._D}
        if self._S is not None:
            result["sample"] = self._S
        result["cell"] = self._C
        return result

    def _check_level(self, level: str, allow_cell: bool = False) -> None:
        allowed = [name for name in self.level_names if allow_cell or name != "cell"]
        if level not in allowed:
            raise ValueError(f"level must be one of {allowed}, got {level!r}.")

    def _codes(self, level: str) -> np.ndarray:
        self._check_level(level)
        return self.donor_codes if level == "donor" else self.sample_codes

    def _table(self, level: str) -> pd.DataFrame:
        """Table of a level, with the donor columns added to the sample table."""
        self._check_level(level)
        if level == "donor":
            return self._D.obs
        extra = [c for c in self._D.obs.columns if c not in self._S.obs.columns]
        if not extra:
            return self._S.obs
        donor_columns = _utils.broadcast(self._D.obs[extra], self.sample_donor_codes, self._S.obs_names)
        return pd.concat([self._S.obs, donor_columns], axis=1)

    @property
    def obs(self) -> pd.DataFrame:
        """Donor table, one row per donor. Equal to ``D`` for a table and to ``D.obs`` otherwise."""
        return self._D.obs

    @property
    def obs_names(self) -> pd.Index:
        """Donor identifiers in their current order."""
        return self._D.obs_names

    @property
    def obsm(self):
        """Donor-aligned matrices such as embeddings or pseudobulk profiles."""
        return self._D.obsm

    @obsm.setter
    def obsm(self, value) -> None:
        self._actualize("donor")
        self._D.obsm = value

    @property
    def obsp(self):
        """Donor-by-donor matrices such as distances, similarities or kinship."""
        return self._D.obsp

    @obsp.setter
    def obsp(self, value) -> None:
        self._actualize("donor")
        self._D.obsp = value

    @property
    def n_donors(self) -> int:
        return self._D.n_obs

    @property
    def n_samples(self) -> int | None:
        """Number of samples, or ``None`` without a sample level."""
        return None if self._S is None else self._S.n_obs

    @property
    def n_cells(self) -> int:
        return self._C.n_obs

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(n_donors, n_donor_vars, n_cells, n_cell_vars)``. A donor table has no donor variables."""
        return *self._D.shape[:2], *self._C.shape

    @property
    def sample_codes(self) -> np.ndarray:
        """Position of the sample of every cell in the sample table."""
        if self._S is None:
            raise ValueError("This DonorData has no sample level.")
        return _utils.codes_for(self._C.obs[self.sample_id], self._S.obs_names)

    @property
    def sample_donor_codes(self) -> np.ndarray:
        """Position of the donor of every sample in :attr:`obs_names`."""
        if self._S is None:
            raise ValueError("This DonorData has no sample level.")
        return _utils.codes_for(self._S.obs[self.donor_id], self.obs_names)

    @property
    def donor_codes(self) -> np.ndarray:
        """Position in :attr:`obs_names` of the donor of every cell."""
        if self._S is None:
            return _utils.codes_for(self._C.obs[self.donor_id], self.obs_names)
        return self.sample_donor_codes[self.sample_codes]

    @property
    def cell_offsets(self) -> np.ndarray:
        """Boundaries of the contiguous block of cells of every donor in ``C``.

        The cells of donor ``i`` are ``C[cell_offsets[i]:cell_offsets[i + 1]]``.
        """
        return self._offsets("donor")

    def _offsets(self, level: str) -> np.ndarray:
        n = self.n_donors if level == "donor" else self.n_samples
        return np.concatenate([[0], np.cumsum(np.bincount(self._codes(level), minlength=n))])

    @property
    def n_cells_per_donor(self) -> pd.Series:
        """Number of cells of every donor."""
        return pd.Series(np.diff(self._offsets("donor")), index=self.obs_names, name="n_cells")

    @property
    def n_cells_per_sample(self) -> pd.Series:
        """Number of cells of every sample."""
        return pd.Series(np.diff(self._offsets("sample")), index=self._S.obs_names, name="n_cells")

    @property
    def n_samples_per_donor(self) -> pd.Series:
        """Number of samples of every donor."""
        counts = np.bincount(self.sample_donor_codes, minlength=self.n_donors)
        return pd.Series(counts, index=self.obs_names, name="n_samples")

    def cells_of(self, name: str, *, level: Level = "donor") -> AnnData | MuData:
        """Cells of a single donor or sample as a view of ``C``."""
        names = self.levels[level].obs_names
        position = names.get_loc(str(name))
        offsets = self._offsets(level)
        return self._C[offsets[position] : offsets[position + 1]]

    def _sync_var_dims(self, D, C):
        kept = []
        for key in self._var_dims_to_sync:
            if key not in D.obsm:
                continue
            kept.append(key)
            frame = D.obsm[key]
            if isinstance(frame, pd.DataFrame) and not frame.columns.equals(C.var_names):
                if D.is_view:
                    D = _utils.copy_container(D)
                D.obsm[key] = D.obsm[key].reindex(columns=C.var_names)
        return D, kept

    def _subset(
        self, D_obs=slice(None), D_var=slice(None), C_obs=slice(None), C_var=slice(None), S_obs=slice(None)
    ) -> DonorData:
        D, S, C = self._D, self._S, self._C
        if not _utils.is_full_slice(D_obs):
            D = D[D_obs]
        if not _utils.is_full_slice(D_var):
            if self._D_is_table:
                raise IndexError("D is a donor table and has no variables to select.")
            D = D[:, D_var]
        if not _utils.is_full_slice(S_obs):
            if S is None:
                raise IndexError("This DonorData has no sample level.")
            S = S[S_obs]
        if not _utils.is_full_slice(C_obs):
            C = C[C_obs]
        if not _utils.is_full_slice(C_var):
            C = C[:, C_var]
        D, kept = self._sync_var_dims(D, C)
        return self._derive(D, S, C, kept)

    def sel(
        self,
        *,
        D_obs=slice(None),
        D_var=slice(None),
        C_obs=slice(None),
        C_var=slice(None),
        S_obs=slice(None),
    ) -> DonorData:
        """Select along the axes of every level by keyword.

        Parameters
        ----------
        D_obs
            Donors to keep.
        D_var
            Donor variables to keep. Not available for a donor table.
        C_obs
            Cells to keep.
        C_var
            Cell variables to keep.
        S_obs
            Samples to keep.

        Returns
        -------
        DonorData
            Units that lose their connection to the other levels are dropped.
        """
        return self._subset(D_obs, D_var, C_obs, C_var, S_obs)

    def __getitem__(self, key) -> DonorData:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 4:
            raise IndexError("DonorData only supports 4 dimensions")
        key = self._expand_ellipsis(key)
        key = tuple((index,) if isinstance(index, str) else index for index in key)
        key = key + (slice(None),) * (4 - len(key))
        return self._subset(*key)

    def _expand_ellipsis(self, index_tuple):
        n_ellipsis = sum(1 for index in index_tuple if index is Ellipsis)
        if n_ellipsis == 0:
            return index_tuple
        if n_ellipsis > 1:
            raise IndexError("DonorData only supports a single ellipsis ('...').")
        position = index_tuple.index(Ellipsis)
        n_full = 4 - (len(index_tuple) - 1)
        return index_tuple[:position] + (slice(None),) * n_full + index_tuple[position + 1 :]

    def filter_donors(self, key: Selection, *, inplace: bool = False) -> DonorData | None:
        """Keep a subset of donors together with their samples and cells.

        Parameters
        ----------
        key
            A query on the donor table such as ``"age > 60 and sex == 'F'"``, a callable that takes
            the donor table and returns a boolean mask, a boolean mask over donors, or donor
            identifiers.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self.obs, "donors")
        return self._finish(self._subset(D_obs=mask), inplace)

    def filter_samples(self, key: Selection, *, inplace: bool = False) -> DonorData | None:
        """Keep a subset of samples together with their cells. Donors without samples are dropped.

        Parameters
        ----------
        key
            A query on the sample table, which also sees donor columns, such as
            ``"timepoint == 'day 7' and age > 60"``, a callable that takes this table and returns a
            boolean mask, a boolean mask over samples, or sample identifiers.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self._table("sample"), "samples")
        return self._finish(self._subset(S_obs=mask), inplace)

    def filter_cells(self, key: Selection, *, inplace: bool = False) -> DonorData | None:
        """Keep a subset of cells. Samples and donors that lose all their cells are dropped.

        Parameters
        ----------
        key
            A query on ``C.obs`` such as ``"cell_type == 'B cell'"``, a callable that takes ``C.obs``
            and returns a boolean mask, a boolean mask over cells, or cell names.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        mask = _utils.resolve_mask(key, self._C.obs, "cells")
        return self._finish(self._subset(C_obs=mask), inplace)

    def _subset_level(self, level: str, mask) -> DonorData:
        return self._subset(D_obs=mask) if level == "donor" else self._subset(S_obs=mask)

    def filter_donors_by_cells(
        self,
        min_cells: int,
        *,
        groupby: str | None = None,
        groups: Sequence | None = None,
        level: Level = "donor",
        inplace: bool = False,
    ) -> DonorData | None:
        """Keep donors, or samples, with enough cells, optionally in every one of several cell groups.

        Pseudobulk and composition analyses need each unit to contribute enough cells of every cell
        type that is compared. With ``groupby`` a unit is kept only if it has at least ``min_cells``
        cells in each of ``groups``.

        Parameters
        ----------
        min_cells
            Minimum number of cells.
        groupby
            Column of ``C.obs`` defining cell groups, for example cell types.
        groups
            Groups that must reach ``min_cells``. Defaults to every group of ``groupby``.
        level
            ``"donor"`` or ``"sample"``.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        if groupby is None:
            keep = np.diff(self._offsets(level)) >= min_cells
        else:
            counts = self._group_counts(groupby, level)
            groups = list(counts.columns) if groups is None else list(groups)
            missing = [g for g in groups if g not in counts.columns]
            if missing:
                raise KeyError(f"Groups {missing} not found in C.obs['{groupby}'].")
            keep = (counts[groups] >= min_cells).all(axis=1).to_numpy()
        return self._finish(self._subset_level(level, keep), inplace)

    def subsample_cells(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        groupby: str | None = None,
        level: Level = "donor",
        random_state: int | None = 0,
        inplace: bool = False,
    ) -> DonorData | None:
        """Randomly subsample cells within every donor or sample.

        Caps how much any single unit contributes, for example to stop donors with many cells from
        dominating a model, while keeping every unit.

        Parameters
        ----------
        n
            Maximum number of cells to keep per unit, or per unit and group with ``groupby``.
            Smaller units keep all their cells.
        fraction
            Fraction of cells to keep per unit instead of a fixed number. At least one cell is kept.
        groupby
            Column of ``C.obs`` to stratify by within units, for example cell types.
        level
            ``"donor"`` or ``"sample"``.
        random_state
            Seed of the random generator.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The subsampled object, or ``None`` when ``inplace`` is set.
        """
        if (n is None) == (fraction is None):
            raise ValueError("Provide exactly one of n or fraction.")
        strata = self._codes(level)
        if groupby is not None:
            group_codes, groups = pd.factorize(self._C.obs[groupby], use_na_sentinel=False)
            strata = strata * len(groups) + group_codes
        strata_codes, strata = pd.factorize(strata)
        sizes = np.bincount(strata_codes)
        if n is not None:
            limits = np.minimum(sizes, n)
        else:
            limits = np.maximum(np.round(sizes * fraction), 1).astype(int)
        rng = np.random.default_rng(random_state)
        order = np.lexsort((rng.random(strata_codes.size), strata_codes))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size) - starts[strata_codes[order]]
        keep = rank < limits[strata_codes]
        return self._finish(self._subset(C_obs=keep), inplace)

    def complete_cases(
        self, mods: Sequence[str] | None = None, *, min_cells: int = 1, level: Level = "donor", inplace: bool = False
    ) -> DonorData | None:
        """Keep donors, or samples, that have cells in every modality of a MuData ``C``.

        Parameters
        ----------
        mods
            Modalities that must be present. Defaults to all of them.
        min_cells
            Minimum number of cells per modality.
        level
            ``"donor"`` or ``"sample"``.
        inplace
            Modify this object instead of returning a new one.

        Returns
        -------
        DonorData or None
            The filtered object, or ``None`` when ``inplace`` is set.
        """
        coverage = self.coverage(level=level)
        mods = list(coverage.columns) if mods is None else list(mods)
        keep = (coverage[mods] >= min_cells).all(axis=1).to_numpy()
        return self._finish(self._subset_level(level, keep), inplace)

    def coverage(self, *, level: Level = "donor") -> pd.DataFrame:
        """Number of cells of every donor, or sample, measured in each modality of ``C``.

        A MuData can pair assays measured on different cells of the same donors, for example blood
        scRNA-seq with tissue scATAC-seq. This table shows which units have which assays.

        Parameters
        ----------
        level
            ``"donor"`` or ``"sample"``.

        Returns
        -------
        pandas.DataFrame
            Units by modalities, or a single column ``"C"`` when ``C`` is an AnnData.
        """
        codes = self._codes(level)
        names = self.levels[level].obs_names
        if isinstance(self._C, AnnData):
            return pd.DataFrame({"C": np.bincount(codes, minlength=len(names))}, index=names)
        counts = {}
        for name in self._C.mod:
            present = np.asarray(self._C.obsmap[name]).ravel() > 0
            counts[name] = np.bincount(codes[present], minlength=len(names))
        return pd.DataFrame(counts, index=names)

    def _group_counts(self, groupby: str, level: str = "donor") -> pd.DataFrame:
        codes = self._codes(level)
        names = self.levels[level].obs_names
        if groupby in self._C.obsm:
            membership = self._C.obsm[groupby]
            labels = membership.columns if isinstance(membership, pd.DataFrame) else pd.RangeIndex(membership.shape[1])
            counts = _utils.group_sum(membership, codes, len(names))
            return pd.DataFrame(counts, index=names, columns=pd.Index(labels.astype(str), name=groupby))
        groups = self._C.obs[groupby]
        categories = _observed_categories(groups)
        group_codes = pd.Categorical(groups, categories=categories).codes
        member = group_codes >= 0
        flat = codes[member] * len(categories) + group_codes[member]
        counts = np.bincount(flat, minlength=len(names) * len(categories)).reshape(len(names), len(categories))
        return pd.DataFrame(counts, index=names, columns=pd.Index(categories, name=groupby))

    def composition(
        self,
        groupby: str,
        *,
        normalize: bool = True,
        clr: bool = False,
        pseudocount: float = 0.5,
        level: Level = "donor",
        key_added: str | None = None,
    ) -> pd.DataFrame:
        """Cell group composition of every donor or sample.

        Parameters
        ----------
        groupby
            Column of ``C.obs`` with cell groups, for example cell types, or a key of ``C.obsm`` with
            a cells by groups membership matrix. A membership matrix allows overlapping groups, such
            as the neighbourhoods of Milo, and yields the units by neighbourhoods count matrix.
        normalize
            Return fractions that sum to one per unit instead of counts.
        clr
            Return the centered log-ratio of the counts plus ``pseudocount``, the usual transform for
            comparing compositions.
        pseudocount
            Added to all counts before the centered log-ratio.
        level
            ``"donor"`` or ``"sample"``.
        key_added
            Key of the level's ``obsm`` to store the result under. Defaults to
            ``"{groupby}_clr"``, ``"{groupby}_fractions"`` or ``"{groupby}_counts"``.

        Returns
        -------
        pandas.DataFrame
            Units by groups. The same frame is stored in ``obsm`` of the level as a snapshot that
            follows selections but is not recomputed when cells are filtered.
        """
        counts = self._group_counts(groupby, level)
        if clr:
            logs = np.log(counts.to_numpy(dtype=np.float64) + pseudocount)
            result = pd.DataFrame(logs - logs.mean(axis=1, keepdims=True), index=counts.index, columns=counts.columns)
            suffix = "clr"
        elif normalize:
            result = counts.astype(np.float64).div(counts.sum(axis=1), axis=0)
            suffix = "fractions"
        else:
            result = counts.astype(np.int64)
            suffix = "counts"
        result.columns = result.columns.astype(str)
        self._actualize(level)
        self.levels[level].obsm[key_added or f"{groupby}_{suffix}"] = result
        return result

    def infer_levels(self) -> pd.DataFrame:
        """Find the level every annotation naturally belongs to.

        A column of ``C.obs`` that is constant within every donor is donor-level information that is
        repeated on cells, and one that is constant within samples is sample-level. The same holds
        for columns of the sample table.

        Returns
        -------
        pandas.DataFrame
            One row per column with the table it is stored in and its ``level``.
        """
        rows = []
        donor_codes = self.donor_codes
        donor_first = _utils.first_positions(donor_codes, self.n_donors)
        if self._S is not None:
            sample_codes = self.sample_codes
            sample_first = _utils.first_positions(sample_codes, self.n_samples)
            sample_donors = self.sample_donor_codes
            sample_donor_first = _utils.first_positions(sample_donors, self.n_donors)
            for column in self._S.obs.columns:
                if column in (self.donor_id, self.sample_id):
                    continue
                constant = _utils.is_constant_within(self._S.obs[column], sample_donors, sample_donor_first)
                rows.append((column, "sample", "donor" if constant else "sample"))
        for column in self._C.obs.columns:
            if column in (self.donor_id, self.sample_id):
                continue
            values = self._C.obs[column]
            if _utils.is_constant_within(values, donor_codes, donor_first):
                level = "donor"
            elif self._S is not None and _utils.is_constant_within(values, sample_codes, sample_first):
                level = "sample"
            else:
                level = "cell"
            rows.append((column, "cell", level))
        return pd.DataFrame(rows, columns=["column", "table", "level"]).set_index("column")

    def get_df(
        self,
        keys: str | Sequence[str],
        *,
        level: Level = "cell",
        func: str | Callable = "mean",
        layer: str | None = None,
        mod: str | None = None,
    ) -> pd.DataFrame:
        """Collect variables from any level into one table at the requested level.

        Keys can name donor or sample columns, cell annotations, genes or other variables of ``C``,
        variables of an AnnData ``D``, or ``obsm`` keys of any level, which expand into one column per
        dimension. Values from coarser levels are repeated for their members, values from finer
        levels are aggregated with ``func`` when numeric and kept when constant within the unit. A
        prefix such as ``"D:age"``, ``"S:timepoint"`` or ``"C:n_genes"`` selects the level when a name
        exists at several levels, and ``"rna:CD14"`` selects a modality of a MuData ``C``.

        Parameters
        ----------
        keys
            Names of the variables to collect.
        level
            ``"cell"``, ``"sample"`` or ``"donor"``.
        func
            Aggregation of numeric values moved from finer levels.
        layer
            Layer of ``C`` to read genes from instead of ``X``.
        mod
            Modality of a MuData ``C`` to read genes from.

        Returns
        -------
        pandas.DataFrame
            Indexed by the units of ``level``.

        Examples
        --------
        >>> dd.get_df(["age", "severity", "cell_type"], level="cell")
        >>> dd.get_df(["age", "CD14", "X_scVI"], level="donor")
        """
        from donordata._core._getter import get_df

        return get_df(self, keys, level=level, func=func, layer=layer, mod=mod)

    def push_obs(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        level: Level = "donor",
        to_level: Level = "cell",
        overwrite: bool = False,
    ) -> None:
        """Broadcast columns of a level table to the members of every unit.

        Parameters
        ----------
        columns
            Columns of the table of ``level``. Defaults to all of them except the identifiers.
        level
            Level the columns come from, ``"donor"`` or ``"sample"``.
        to_level
            Finer level to write them to, ``"cell"`` or ``"sample"``.
        overwrite
            Replace columns that already exist at ``to_level``.
        """
        self._check_level(level)
        self._check_level(to_level, allow_cell=True)
        if _RANK[to_level] <= _RANK[level]:
            raise ValueError("to_level must be finer than level.")
        source = self.levels[level].obs
        columns = self._level_columns(source, columns)
        target = self.levels[to_level].obs
        clashes = [c for c in columns if c in target.columns]
        if clashes and not overwrite:
            raise ValueError(f"Columns {clashes} already exist at the {to_level} level, pass overwrite=True.")
        codes = self.sample_donor_codes if to_level == "sample" else self._codes(level)
        self._actualize(to_level)
        target = self.levels[to_level].obs
        for column in columns:
            target[column] = source[column].array.take(codes)

    def pull_obs(
        self,
        columns: str | Sequence[str] | Mapping[str, str] | None = None,
        *,
        func: str | Callable | None = None,
        level: Level = "donor",
        from_level: Level = "cell",
        drop: bool = False,
        overwrite: bool = False,
    ) -> list[str]:
        """Collect columns of a finer level into the table of a coarser one.

        Parameters
        ----------
        columns
            Columns to collect, or a mapping from their names to the names they get in the table of
            ``level``. By default every column that is constant within the units and not yet in the
            target table.
        func
            Aggregation for columns that vary within units, for example ``"mean"`` or ``"median"``.
            Without it such columns raise an error.
        level
            Level to collect into, ``"donor"`` or ``"sample"``.
        from_level
            Finer level to collect from, ``"cell"`` or ``"sample"``.
        drop
            Remove the collected columns from the finer level.
        overwrite
            Replace columns that already exist in the target table.

        Returns
        -------
        list of str
            The names of the collected columns in the target table.

        Examples
        --------
        >>> dd.pull_obs({"n_genes": "median_n_genes", "pct_mito": "median_pct_mito"}, func="median")
        """
        self._check_level(level)
        self._check_level(from_level, allow_cell=True)
        if _RANK[from_level] <= _RANK[level]:
            raise ValueError("from_level must be finer than level.")
        source = self.levels[from_level].obs
        target = self.levels[level].obs
        codes = self.sample_donor_codes if from_level == "sample" else self._codes(level)
        n_units = len(target)
        first = _utils.first_positions(codes, n_units)
        if columns is None:
            exclude = {self.donor_id, self.sample_id} | (set() if overwrite else set(target.columns))
            columns = _utils.constant_columns(source, codes, first, exclude=exclude)
        if isinstance(columns, Mapping):
            names = dict(columns)
        else:
            names = {c: c for c in ([columns] if isinstance(columns, str) else columns)}
        clashes = [c for c in names.values() if c in target.columns]
        if clashes and not overwrite:
            raise ValueError(f"Columns {clashes} already exist in the {level} table, pass overwrite=True.")

        values = {}
        for column, name in names.items():
            if _utils.is_constant_within(source[column], codes, first):
                collected = source[column].iloc[first]
                collected.index = target.index
            elif func is None:
                raise ValueError(f"'{column}' varies within {level}s, pass func= to aggregate it.")
            else:
                collected = source[column].groupby(codes, observed=True).agg(func)
                collected = collected.reindex(np.arange(n_units))
                collected.index = target.index
            values[name] = collected

        self._actualize(level)
        target = self.levels[level].obs
        for name, collected in values.items():
            target[name] = collected
        if drop and names:
            self._actualize(from_level)
            container = self.levels[from_level]
            container.obs = container.obs.drop(columns=list(names))
        return list(values)

    def push_obsm(self, key: str, *, level: Level = "donor", key_added: str | None = None) -> None:
        """Broadcast a donor- or sample-level matrix to every cell in ``C.obsm``.

        Parameters
        ----------
        key
            Key of the level's ``obsm``, for example a patient embedding.
        level
            ``"donor"`` or ``"sample"``.
        key_added
            Key in ``C.obsm``. Defaults to ``key``.
        """
        matrix = self.levels[level].obsm[key]
        codes = self._codes(level)
        if isinstance(matrix, pd.DataFrame):
            broadcast = matrix.iloc[codes]
            broadcast.index = self._C.obs_names
        else:
            broadcast = matrix[codes]
        self._actualize("cell")
        self._C.obsm[key_added or key] = broadcast

    def to_anndata(
        self, columns: str | Sequence[str] | None = None, *, obsm: Sequence[str] = (), level: Level = "donor"
    ) -> AnnData | MuData:
        """Cell-level object with donor and sample annotations attached to every cell.

        Useful to hand the data to tools that only understand a single AnnData, for example to color
        cells by patient covariates. Matrices are shared with ``C``, not copied.

        Parameters
        ----------
        columns
            Columns of the donor table, and of the sample table when there is one, to attach.
            Defaults to all of them except identifiers already present in ``C.obs``.
        obsm
            Keys of the ``obsm`` of ``level`` to attach to ``obsm`` of the result.
        level
            Level whose ``obsm`` the keys in ``obsm`` refer to.

        Returns
        -------
        AnnData or MuData
            Of the same type as ``C``.
        """
        table = self._table("sample" if self._S is not None else "donor")
        codes = self.sample_codes if self._S is not None else self.donor_codes
        if columns is None:
            columns = [c for c in table.columns if c not in self._C.obs.columns]
        else:
            columns = self._level_columns(table, columns)
        clashes = [c for c in columns if c in self._C.obs.columns]
        if clashes:
            raise ValueError(f"Columns {clashes} exist both at a unit level and in C.obs.")
        obs = self._C.obs.assign(**{c: table[c].array.take(codes) for c in columns})
        result = _utils.shallow_with_obs(self._C, obs)
        unit_codes = self._codes(level)
        for key in obsm:
            matrix = self.levels[level].obsm[key]
            if isinstance(matrix, pd.DataFrame):
                matrix = matrix.iloc[unit_codes].set_axis(self._C.obs_names)
            else:
                matrix = matrix[unit_codes]
            result.obsm[key] = matrix
        return result

    def _level_columns(self, table: pd.DataFrame, columns) -> list[str]:
        if columns is None:
            return [c for c in table.columns if c not in (self.donor_id, self.sample_id)]
        columns = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise KeyError(f"Columns {missing} not found.")
        return columns

    def cell_weights(self, *, level: Level = "donor", by: str | None = None) -> pd.Series:
        """Weights that let every donor, or sample, contribute equally to a cell-level model.

        Each cell gets the inverse number of cells of its unit, so donors with many cells do not
        dominate a loss or a fit. With ``by`` the classes of a unit-level column, such as disease
        status, are balanced as well. Weights are scaled to a mean of one.

        Parameters
        ----------
        level
            ``"donor"`` or ``"sample"``.
        by
            Column of the unit table whose classes get equal total weight.

        Returns
        -------
        pandas.Series
            One weight per cell.
        """
        codes = self._codes(level)
        sizes = np.bincount(codes)
        weights = 1.0 / sizes[codes]
        if by is not None:
            classes = pd.factorize(self._table(level)[by], use_na_sentinel=False)[0]
            units_per_class = np.bincount(classes)
            weights = weights / units_per_class[classes[codes]]
        weights *= weights.size / weights.sum()
        return pd.Series(weights, index=self._C.obs_names, name="weight")

    def aggregate(
        self,
        *,
        key_added: None | str = None,
        layer: None | str = None,
        obs: None | str | list[str] = None,
        obsm: None | str = None,
        filter_key: None | str = None,
        filter_value=None,
        add_to_obs: bool = False,
        func: str | Callable = "mean",
        sync_var: bool = False,
        verbose: bool = False,
        mod: str | None = None,
        level: Level = "donor",
    ) -> None:
        """Aggregate cell-level data to donors or samples.

        Aggregates ``C.X`` unless one of ``layer``, ``obsm`` or ``obs`` is given. Matrices are stored
        in ``obsm`` of the level, columns of ``C.obs`` in ``obsm`` or, with ``add_to_obs``, in the
        level table.

        Parameters
        ----------
        key_added
            Key of the result. Defaults to ``"{slot}_{func}"``, followed by ``filter_value`` when
            filtering.
        layer
            Layer of ``C`` to aggregate.
        obs
            Column or columns of ``C.obs`` to aggregate.
        obsm
            Key of ``C.obsm`` to aggregate.
        filter_key
            Column of ``C.obs`` used to restrict the cells, for example a cell type column.
        filter_value
            Value of ``filter_key`` that cells must have.
        add_to_obs
            Store aggregated ``obs`` columns in the level table instead of ``obsm``.
        func
            Aggregation for matrices, one of ``"sum"``, ``"mean"``, ``"var"``, ``"count_nonzero"``,
            ``"median"``, ``"min"`` or ``"max"``. For ``obs`` any pandas aggregation is accepted.
        sync_var
            Keep the columns of the result in sync with ``C.var_names`` when genes are selected.
        verbose
            Log what was aggregated.
        mod
            Modality to aggregate when ``C`` is a MuData. Modalities measured on a part of the cells
            are aggregated over those cells.
        level
            ``"donor"`` or ``"sample"``.
        """
        if filter_key is not None and filter_value is None:
            raise ValueError("filter_value must be provided if filter_key is provided")
        if sum(x is not None for x in (layer, obsm, obs)) > 1:
            raise ValueError("Only one of layer, obsm or obs can be provided")

        codes = self._codes(level)
        container = self.levels[level]
        n_units, names = container.n_obs, container.obs_names
        if filter_key is not None:
            codes = np.where((self._C.obs[filter_key] == filter_value).to_numpy(), codes, -1)

        if obs is not None:
            slot = obs if isinstance(obs, str) else "_".join(obs)
        else:
            slot = layer or obsm or "X"
        if key_added is None:
            key_added = f"{slot}_{func if isinstance(func, str) else func.__name__}"
            if filter_value is not None:
                key_added += f"_{filter_value}"

        if obs is not None:
            columns = [obs] if isinstance(obs, str) else list(obs)
            member = codes >= 0
            result = self._C.obs.loc[member, columns].groupby(codes[member], observed=True).agg(func)
            result = result.reindex(np.arange(n_units))
            result.index = names
            self._actualize(level)
            container = self.levels[level]
            if add_to_obs:
                for column in columns:
                    container.obs[column] = result[column]
            else:
                container.obsm[key_added] = result
        else:
            if not isinstance(func, str):
                raise TypeError("func must be a string when aggregating matrices.")
            matrix, rows = _utils.cell_matrix(self._C, layer=layer, obsm=obsm, mod=mod)
            values = _utils.aggregate_rows(matrix, codes if rows is None else codes[rows], n_units, func)
            if obsm is not None:
                result = (
                    pd.DataFrame(values, index=names, columns=matrix.columns)
                    if isinstance(matrix, pd.DataFrame)
                    else values
                )
            else:
                var_names = _utils.modality(self._C, mod)[0].var_names
                result = pd.DataFrame(values, index=names, columns=var_names)
                if sync_var and level == "donor" and key_added not in self._var_dims_to_sync:
                    self._var_dims_to_sync.append(key_added)
            self._actualize(level)
            self.levels[level].obsm[key_added] = result

        if verbose:
            n_observed = np.unique(codes[codes >= 0]).size
            logger.info("Aggregated %s to %s, observed for %d %ss.", slot, key_added, n_observed, level)

    def pseudobulk(
        self,
        *,
        layer: str | None = None,
        groupby: str | None = None,
        func: str = "sum",
        min_cells: int = 1,
        mod: str | None = None,
        level: Level = "donor",
    ) -> AnnData:
        """Aggregate cells into one profile per donor or sample, or per unit and cell group.

        The result carries the unit table in ``obs``, with donor columns added for samples, and is
        ready for differential testing with covariates of both levels in the design. ``n_cells`` and
        ``total_counts`` record how many cells and counts every profile is built from. Without
        ``groupby`` it also carries ``obsm``, ``obsp`` and :attr:`uns` of the level, so it is a
        complete unit-level AnnData that can be assigned to ``D`` or used with tools that expect one
        AnnData per cohort.

        Parameters
        ----------
        layer
            Layer of ``C`` to aggregate, ``X`` by default. Raw counts suit differential testing.
        groupby
            Column of ``C.obs``, for example cell types, giving one profile per unit and group.
        func
            Aggregation, see :meth:`aggregate`.
        min_cells
            Profiles built from fewer cells are dropped.
        mod
            Modality to aggregate when ``C`` is a MuData.
        level
            ``"donor"`` or ``"sample"``.

        Returns
        -------
        anndata.AnnData
            Profiles by genes, with ``obs`` holding the unit table, ``groupby``, ``n_cells`` and
            ``total_counts``.
        """
        adata, rows = _utils.modality(self._C, mod)
        matrix, _ = _utils.cell_matrix(self._C, layer=layer, mod=mod)
        container = self.levels[level]
        names = container.obs_names
        unit_codes = self._codes(level)
        if rows is not None:
            unit_codes = unit_codes[rows]
        if groupby is None:
            codes, n_groups, categories = unit_codes, len(names), None
        else:
            groups = self._C.obs[groupby]
            if rows is not None:
                groups = groups.iloc[rows]
            categories = _observed_categories(groups)
            group_codes = pd.Categorical(groups, categories=categories).codes
            codes = np.where(group_codes >= 0, unit_codes * len(categories) + group_codes, -1)
            n_groups = len(names) * len(categories)

        counts = np.bincount(codes[codes >= 0], minlength=n_groups)
        keep = np.flatnonzero(counts >= max(min_cells, 1))
        values = _utils.aggregate_rows(matrix, codes, n_groups, func)[keep]
        sums = values if func == "sum" else _utils.aggregate_rows(matrix, codes, n_groups, "sum")[keep]
        totals = sums.sum(axis=1)

        unit_positions = keep if groupby is None else keep // len(categories)
        obs = self._table(level).iloc[unit_positions].copy()
        key = self.donor_id if level == "donor" else self.sample_id
        if key not in obs.columns:
            obs.insert(0, key, obs.index.to_numpy())
        if groupby is None:
            obs.index = names[unit_positions]
        else:
            group_values = categories[keep % len(categories)]
            obs[groupby] = pd.Categorical(group_values, categories=categories)
            obs.index = [f"{d}_{g}" for d, g in zip(names[unit_positions], group_values, strict=True)]
        obs.index.name = None
        obs["n_cells"] = counts[keep]
        obs["total_counts"] = totals
        if groupby is not None:
            return AnnData(X=values, obs=obs, var=adata.var.copy())
        units = container[keep]
        return AnnData(
            X=values,
            obs=obs,
            var=adata.var.copy(),
            obsm=dict(units.obsm),
            obsp=dict(units.obsp),
            uns=_copy.deepcopy(self.uns),
        )

    def collapse(
        self,
        by: str,
        *,
        func: str | dict | None = None,
        obsm_func: str | None = "mean",
        count_key: str | None = None,
    ) -> DonorData:
        """Merge donors into coarser units given by a donor table column.

        Moves one level up, for example from guides to their target genes, from wells to drugs or
        from donors to sites. Cells keep their original donor in ``C.obs`` and gain the new unit in
        ``C.obs[by]``. A sample level is kept, with every sample assigned to its new unit. Use
        :meth:`nest` instead to keep the merged donors as a sample level.

        Parameters
        ----------
        by
            Donor table column naming the new unit of every donor.
        func
            Aggregation for donor table columns that vary within a unit, either one pandas
            aggregation for all numeric columns or a dict per column. Without it such columns are
            dropped.
        obsm_func
            Aggregation of :attr:`obsm` matrices over the donors of a unit, see :meth:`aggregate`.
            ``None`` drops them. Donor-by-donor matrices in :attr:`obsp` are always dropped.
        count_key
            Column with the number of merged donors. Defaults to ``"n_{donor_id}"``.

        Returns
        -------
        DonorData
            With a donor table indexed by the values of ``by``.
        """
        unit_codes, table = self._units_of(by, func=func, count_key=count_key)
        obsm = {}
        if obsm_func is not None:
            for key, matrix in self.obsm.items():
                values = _utils.aggregate_rows(matrix, unit_codes, len(table), obsm_func)
                if isinstance(matrix, pd.DataFrame):
                    values = pd.DataFrame(values, index=table.index, columns=matrix.columns)
                obsm[key] = values

        cell_units = pd.Categorical(table.index[unit_codes[self.donor_codes]], categories=table.index)
        cells = _utils.shallow_with_obs(self._C, self._C.obs.assign(**{by: cell_units}))
        synced = [k for k in self._var_dims_to_sync if k in obsm]
        if self._S is None:
            new = DonorData(D=table, C=cells, donor_id=by, obsm=obsm, uns=dict(self.uns), var_dims_to_sync=synced)
            new._spatial = self._spatial
            return new
        samples = self._S.copy()
        samples.obs[by] = np.asarray(table.index[unit_codes[self.sample_donor_codes]], dtype=str)
        new = DonorData(
            D=table,
            S=samples.obs if self._S_is_table else samples,
            C=cells,
            donor_id=by,
            sample_id=self.sample_id,
            obsm=obsm,
            uns=dict(self.uns),
            var_dims_to_sync=synced,
        )
        if self._S_is_table:
            _copy_matrices(samples, new._S)
        new._spatial = self._spatial
        return new

    def _units_of(self, by: str, *, func=None, count_key: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if by not in self.obs.columns:
            raise KeyError(f"'{by}' not found in the donor table.")
        units = self.obs[by]
        if units.isna().any():
            raise ValueError(f"'{by}' has missing values, every donor needs a unit.")
        unit_codes, unit_names = pd.factorize(pd.Series(_utils.string_labels(units)), sort=True)
        unit_names = pd.Index(np.asarray(unit_names, dtype=str), name=by)
        first = _utils.first_positions(unit_codes, len(unit_names))
        constant = _utils.constant_columns(self.obs, unit_codes, first, exclude=(by, self.donor_id))
        table = self.obs.iloc[first][constant].copy()
        table.index = unit_names
        if func is not None:
            varying = [c for c in self.obs.columns if c not in constant and c not in (by, self.donor_id)]
            if isinstance(func, dict):
                plan = {c: f for c, f in func.items() if c in varying}
            else:
                plan = {c: func for c in varying if pd.api.types.is_numeric_dtype(self.obs[c])}
            if plan:
                aggregated = self.obs[list(plan)].groupby(unit_codes).agg(plan)
                aggregated.index = unit_names
                table = table.join(aggregated)
        table[count_key or f"n_{self.donor_id}"] = np.bincount(unit_codes, minlength=len(unit_names))
        return unit_codes, table

    def nest(self, by: str, *, keep_in_samples: bool = False) -> DonorData:
        """Turn the donors into samples of coarser donors given by a donor table column.

        Unlike :meth:`collapse` nothing is merged away: the current donor table, with its matrices,
        becomes the sample table and the values of ``by`` become the new donors. Columns that are
        constant within the new donors move to their table. This is how a cohort built from samples,
        for example a patient on a collection day, gains its patient level.

        Parameters
        ----------
        by
            Donor table column naming the new donor of every current donor.
        keep_in_samples
            Keep the moved columns in the sample table as well.

        Returns
        -------
        DonorData
            With three levels, where ``sample_id`` is the current ``donor_id`` and ``donor_id`` is
            ``by``.
        """
        if self._S is not None:
            raise ValueError("This DonorData already has a sample level.")
        unit_codes, table = self._units_of(by)
        table = table.drop(columns=[f"n_{self.donor_id}"])
        samples = _utils.copy_container(self._D)
        samples.obs[by] = np.asarray(table.index[unit_codes], dtype=str)
        if not keep_in_samples:
            samples.obs = samples.obs.drop(columns=[c for c in table.columns if c in samples.obs.columns])
        sample_side = samples.obs if self._D_is_table else samples
        new = DonorData(D=table, S=sample_side, C=self._C, donor_id=by, sample_id=self.donor_id, uns=dict(self.uns))
        if self._D_is_table:
            _copy_matrices(samples, new._S)
        new._spatial = self._spatial
        return new

    def flatten(self) -> DonorData:
        """Drop the donor level and make the samples the units.

        Donor columns and donor-aligned matrices are repeated for every sample of a donor, where they
        do not clash with sample columns or matrices. Donor-by-donor matrices are dropped.

        Returns
        -------
        DonorData
            With two levels, where ``donor_id`` is the current ``sample_id``.
        """
        if self._S is None:
            raise ValueError("This DonorData has no sample level.")
        codes = self.sample_donor_codes
        samples = self._S.copy()
        extra = [c for c in self._D.obs.columns if c not in samples.obs.columns]
        if extra:
            donor_columns = _utils.broadcast(self._D.obs[extra], codes, samples.obs_names)
            samples.obs = pd.concat([samples.obs, donor_columns], axis=1)
        for key, matrix in self._D.obsm.items():
            if key in samples.obsm:
                continue
            if isinstance(matrix, pd.DataFrame):
                samples.obsm[key] = matrix.iloc[codes].set_axis(samples.obs_names)
            else:
                samples.obsm[key] = matrix[codes]
        side = samples.obs if self._S_is_table else samples
        new = DonorData(D=side, C=self._C, donor_id=self.sample_id, uns=dict(self.uns))
        if self._S_is_table:
            _copy_matrices(samples, new._D)
        new._spatial = self._spatial
        return new

    def _strata(self, stratify) -> np.ndarray:
        if stratify is None:
            return np.zeros(self.n_donors, dtype=np.int64)
        values = self.obs[stratify] if isinstance(stratify, str) else pd.Series(np.asarray(stratify))
        if len(values) != self.n_donors:
            raise ValueError("stratify must have one value per donor.")
        return pd.factorize(values, use_na_sentinel=False)[0]

    def _split_units(self, stratify, groups) -> tuple[np.ndarray, np.ndarray]:
        strata = self._strata(stratify)
        if groups is None:
            return np.arange(self.n_donors), strata
        group_codes = pd.factorize(self.obs[groups], use_na_sentinel=False)[0]
        group_strata = np.array([np.bincount(strata[group_codes == g]).argmax() for g in range(group_codes.max() + 1)])
        return group_codes, group_strata

    def train_test_split(
        self,
        test_size: float | int = 0.2,
        *,
        stratify: str | Sequence | None = None,
        groups: str | None = None,
        random_state: int | None = 0,
    ) -> tuple[DonorData, DonorData]:
        """Split donors into a training and a test set.

        All samples and cells of a donor land on the same side, so models are evaluated on unseen
        donors. With ``groups`` whole groups of donors, such as sites, studies or families, stay
        together.

        Parameters
        ----------
        test_size
            Fraction of donors, or of groups with ``groups``, held out for testing, or their number
            when an integer.
        stratify
            Donor table column, or one value per donor, whose proportions are kept in both sets. With
            ``groups`` each group counts with its most frequent value.
        groups
            Donor table column whose groups are never split.
        random_state
            Seed of the random generator.

        Returns
        -------
        tuple of DonorData
            Training and test set.
        """
        unit_of_donor, strata = self._split_units(stratify, groups)
        n_units = strata.size
        n_test = int(test_size) if isinstance(test_size, int | np.integer) else int(round(test_size * n_units))
        if not 0 < n_test < n_units:
            raise ValueError(f"test_size gives {n_test} test units out of {n_units}.")
        sizes = np.bincount(strata)
        quota = sizes * n_test / n_units
        n_per_stratum = np.floor(quota).astype(int)
        remainder = n_test - n_per_stratum.sum()
        n_per_stratum[np.argsort(-(quota - n_per_stratum), kind="stable")[:remainder]] += 1

        rng = np.random.default_rng(random_state)
        test_units = np.zeros(n_units, dtype=bool)
        for stratum, n_take in enumerate(n_per_stratum):
            members = np.flatnonzero(strata == stratum)
            test_units[rng.choice(members, size=n_take, replace=False)] = True
        test = test_units[unit_of_donor]
        return self._subset(D_obs=~test), self._subset(D_obs=test)

    def kfold(
        self,
        n_splits: int = 5,
        *,
        stratify: str | Sequence | None = None,
        groups: str | None = None,
        random_state: int | None = 0,
    ) -> Iterator[tuple[DonorData, DonorData]]:
        """Iterate over cross validation folds of donors.

        Parameters
        ----------
        n_splits
            Number of folds. Every donor is in the test set of exactly one fold.
        stratify
            Donor table column, or one value per donor, balanced across folds.
        groups
            Donor table column whose groups, such as sites or studies, are never split across folds.
        random_state
            Seed of the random generator.

        Yields
        ------
        tuple of DonorData
            Training and test set of every fold.
        """
        unit_of_donor, strata = self._split_units(stratify, groups)
        if not 2 <= n_splits <= strata.size:
            raise ValueError(f"n_splits must be between 2 and the number of units ({strata.size}).")
        rng = np.random.default_rng(random_state)
        folds = np.empty(strata.size, dtype=np.int64)
        offset = 0
        for stratum in range(strata.max() + 1):
            members = rng.permutation(np.flatnonzero(strata == stratum))
            folds[members] = (np.arange(members.size) + offset) % n_splits
            offset += members.size
        donor_folds = folds[unit_of_donor]
        for fold in range(n_splits):
            yield self._subset(D_obs=donor_folds != fold), self._subset(D_obs=donor_folds == fold)

    def write_h5_dd(self, path: str) -> None:
        """Write to an HDF5 file.

        Parameters
        ----------
        path
            Destination, conventionally ending in ``.dd.h5``.
        """
        if self._spatial is not None:
            raise ValueError("Cells from a SpatialData can only be written to a Zarr store, use write_zarr_dd.")
        with h5py.File(str(path), "w") as f:
            write_group(self, f)

    def write_zarr_dd(self, path: str, x_chunks: tuple[int, int] | dict[str, tuple[int, int]] | None = None) -> None:
        """Write to a Zarr store.

        Parameters
        ----------
        path
            Destination, conventionally ending in ``.dd.zarr``.
        x_chunks
            Chunk shape for a dense ``X`` and dense layers, for example ``(4096, 4096)``, or a dict
            ``{"D": ..., "C": ...}`` to set the levels independently. Without it a dask array keeps
            its own chunks and a plain array is chunked with at most 4096 entries per axis.
        """
        f = zarr.open(str(path), mode="w")
        write_group(self, f, zarr_path=str(path), x_chunks=x_chunks)

    def write_dd(self, path: str, fmt: Literal["h5", "zarr"] | None = None) -> None:
        """Write to disk, choosing the format from the extension unless ``fmt`` is given.

        Parameters
        ----------
        path
            Destination ending in ``.h5`` or ``.zarr``. Without an extension, ``.dd.h5`` or ``.dd.zarr``
            is appended according to ``fmt``.
        fmt
            ``"h5"`` or ``"zarr"``.
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
            self.write_h5_dd(path if path.endswith(".h5") else path + ".dd.h5")
        elif fmt == "zarr":
            self.write_zarr_dd(path if path.endswith(".zarr") else path + ".dd.zarr")
        else:
            raise ValueError("Unknown format: use 'h5' or 'zarr'.")

    def __repr__(self) -> str:
        return render(self)

    def _repr_html_(self) -> str:
        return render(self, html=True)

    def __str__(self) -> str:
        n_donors, n_donor_vars, n_cells, n_cell_vars = self.shape
        samples = "" if self._S is None else f"n_S_obs={self.n_samples:,}, "
        return (
            f"{self.__class__.__name__}"
            f"(n_D_obs={n_donors:,}, n_D_vars={n_donor_vars:,}, {samples}"
            f"n_C_obs={n_cells:,}, n_C_vars={n_cell_vars:,})"
        )


def _observed_categories(groups: pd.Series) -> pd.Index:
    if isinstance(groups.dtype, pd.CategoricalDtype):
        present = np.bincount(groups.cat.codes[groups.cat.codes >= 0], minlength=len(groups.cat.categories)) > 0
        return groups.cat.categories[present]
    return pd.Index(sorted(pd.unique(groups.dropna())))


def _copy_matrices(source: AnnData, target: AnnData) -> None:
    """Copy ``obsm`` and ``obsp`` from ``source`` to ``target``, matching rows by name."""
    if not len(source.obsm) and not len(source.obsp):
        return
    aligned = source[target.obs_names]
    for key in aligned.obsm:
        target.obsm[key] = aligned.obsm[key]
    for key in aligned.obsp:
        target.obsp[key] = aligned.obsp[key]


def _carry_over_matrices(previous: AnnData | MuData | None, new: AnnData) -> None:
    if previous is None:
        return
    keys_m = [k for k in previous.obsm if k not in new.obsm]
    keys_p = [k for k in previous.obsp if k not in new.obsp]
    if not keys_m and not keys_p:
        return
    if not new.obs_names.isin(previous.obs_names).all():
        warnings.warn(f"New units in the table, dropping aligned matrices {keys_m + keys_p}.", stacklevel=3)
        return
    aligned = previous[new.obs_names]
    for key in keys_m:
        new.obsm[key] = aligned.obsm[key]
    for key in keys_p:
        new.obsp[key] = aligned.obsp[key]
