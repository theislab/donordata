from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from anndata.utils import asarray
from mudata import MuData
from torch.utils.data import Dataset

from donordata._core._utils import modality
from donordata._core.donordata import DonorData
from donordata._core.fields import CAnn, DAnn

DONOR_FIELDS = ("donor_x", "donor_y", "donor_batch", "donor_cat_covs", "donor_cont_covs", "donor_indices")
CELL_FIELDS = ("cell_x", "cell_y", "cell_batch", "cell_cat_covs", "cell_cont_covs", "cell_indices", "cell_mask")


def get_array(array: Any, mask: np.ndarray | None = None) -> np.ndarray | None:
    """Convert an array-like object, optionally restricted to ``mask``, into a NumPy array.

    Parameters
    ----------
    array
        Dense, sparse or dask array, or ``None``.
    mask
        Boolean mask or integer positions of the rows to keep.

    Returns
    -------
    numpy.ndarray or None
    """
    if array is None:
        return None
    return asarray(array[mask] if mask is not None else array)


def _encode(values) -> tuple[np.ndarray, list | None]:
    series = pd.Series(np.asarray(values) if not isinstance(values, pd.Series) else values)
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float32), None
    categorical = pd.Categorical(series)
    return categorical.codes.astype(np.float32), list(categorical.categories)


def _column(table: pd.DataFrame, key: str, container=None):
    if key in table.columns:
        return table[key]
    if isinstance(container, MuData):
        for mod in container.mod.values():
            if key in mod.obs.columns:
                return mod.obs[key]
    return None


def _rows(matrix):
    return matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix


def _resolve(adata: AnnData, key: str | None):
    if key is None or key == "X":
        return adata.X
    if key in adata.layers:
        return adata.layers[key]
    if key in adata.obsm:
        return _rows(adata.obsm[key])
    return None


class MILDataset(Dataset):
    """Torch dataset that yields one bag of cells per donor of a :class:`~donordata.DonorData`.

    Every item holds the features and labels of a donor together with the features of all of its
    cells, the setting of multiple instance learning where one label describes a whole bag. Cells of
    a donor are contiguous in ``dd.C``, so bags are read as slices. With ``n_cells`` every bag is
    subsampled or zero padded to a fixed size and ``cell_mask`` marks the real cells, which lets
    :func:`mil_collate_fn` stack bags into dense tensors.

    Non numeric fields are encoded as integer codes, and the class of every code is listed in
    :attr:`encodings`. Fields that are not found are returned as ``None``, except for keys that were
    passed explicitly, which raise a ``KeyError``.

    Parameters
    ----------
    dd
        Donor and cell data.
    donor_layer
        Donor features. ``None`` uses ``D.X`` when ``D`` has a matrix. Other keys are looked up in
        ``D.layers`` and then in ``dd.obsm``.
    donor_labels_key
        Donor table column with the bag label.
    donor_batch_key
        Donor table column with the donor batch.
    donor_cat_covs_key
        Donor table column with a categorical donor covariate.
    donor_cont_covs_key
        Donor table column or :attr:`~donordata.DonorData.obsm` key with continuous donor covariates.
    donor_indices_key
        Donor table column with donor indices.
    cell_layer
        Cell features. ``None`` uses ``C.X``. Other keys are looked up in ``C.layers`` and then in
        ``C.obsm``.
    cell_labels_key
        Column of ``C.obs`` with cell labels.
    cell_batch_key
        Column of ``C.obs`` with the cell batch.
    cell_cat_covs_key
        Column of ``C.obs`` with a categorical cell covariate.
    cell_cont_covs_key
        Column of ``C.obs`` or key of ``C.obsm`` with continuous cell covariates.
    cell_indices_key
        Column of ``C.obs`` with cell indices.
    split_donors
        Donor identifiers to include.
    split_indices
        Positions of the donors to include, as an alternative to ``split_donors``.
    n_cells
        Fixed bag size. Larger bags are subsampled without replacement, smaller ones are zero padded.
    random_state
        Seed of the generator used for subsampling.
    level
        ``"donor"`` for one bag per donor, or ``"sample"`` for one bag per sample. Unit fields are
        then read from the sample table, which also sees donor columns.
    """

    def __init__(
        self,
        dd: DonorData,
        donor_layer: str | None = None,
        donor_labels_key: str | None = None,
        donor_batch_key: str | None = None,
        donor_cat_covs_key: str | None = None,
        donor_cont_covs_key: str | None = None,
        donor_indices_key: str | None = None,
        cell_layer: str | None = None,
        cell_labels_key: str | None = None,
        cell_batch_key: str | None = None,
        cell_cat_covs_key: str | None = None,
        cell_cont_covs_key: str | None = None,
        cell_indices_key: str | None = None,
        split_donors: list | None = None,
        split_indices: list[int] | None = None,
        n_cells: int | None = None,
        random_state: int | None = None,
        level: str = "donor",
    ):
        if split_donors is not None and split_indices is not None:
            raise ValueError("Both split_donors and split_indices cannot be provided at the same time.")
        dd._check_level(level)

        self.dd = dd
        self.level = level
        self.n_cells = n_cells
        container = dd.levels[level]
        names = container.obs_names
        self.donor_names = names.to_numpy()
        self._offsets = dd._offsets(level)
        self._rng = np.random.default_rng(random_state)
        self.encodings: dict[str, list] = {}

        if split_donors is not None:
            self._positions = names.get_indexer(pd.Index(split_donors).astype(str))
            if (self._positions < 0).any():
                missing = np.asarray(split_donors)[self._positions < 0].tolist()
                raise KeyError(f"Units {missing[:8]} are not in the DonorData.")
        elif split_indices is not None:
            self._positions = np.asarray(split_indices, dtype=np.int64)
        else:
            self._positions = np.arange(len(names))
        self.selected_ids = self.donor_names[self._positions]

        donor_keys = {
            "donor_y": (donor_labels_key, DAnn.DONOR_LABELS_KEY),
            "donor_batch": (donor_batch_key, DAnn.DONOR_BATCH_KEY),
            "donor_cat_covs": (donor_cat_covs_key, DAnn.DONOR_CAT_COVS_KEY),
            "donor_cont_covs": (donor_cont_covs_key, DAnn.DONOR_CONT_COVS_KEY),
            "donor_indices": (donor_indices_key, DAnn.DONOR_INDICES_KEY),
        }
        cell_keys = {
            "cell_y": (cell_labels_key, CAnn.CELL_LABELS_KEY),
            "cell_batch": (cell_batch_key, CAnn.CELL_BATCH_KEY),
            "cell_cat_covs": (cell_cat_covs_key, CAnn.CELL_CAT_COVS_KEY),
            "cell_cont_covs": (cell_cont_covs_key, CAnn.CELL_CONT_COVS_KEY),
            "cell_indices": (cell_indices_key, CAnn.CELL_INDICES_KEY),
        }
        self._donor_fields = self._collect(donor_keys, dd._table(level), container.obsm, container)
        self._cell_fields = self._collect(cell_keys, dd.C.obs, dd.C.obsm, dd.C)
        self._donor_matrix = self._donor_features(donor_layer, container)
        self._cell_matrices = self._cell_features(cell_layer)

    def _collect(self, keys: dict, table: pd.DataFrame, obsm, container) -> dict[str, np.ndarray]:
        fields = {}
        for name, (given, default) in keys.items():
            key = given or default
            values = _column(table, key, container)
            if values is None and key in obsm:
                fields[name] = np.asarray(asarray(_rows(obsm[key])), dtype=np.float32)
                continue
            if values is None:
                if given is not None:
                    raise KeyError(f"Field '{given}' not found in the data.")
                continue
            fields[name], classes = _encode(values)
            if classes is not None:
                self.encodings[key] = classes
        return fields

    def _donor_features(self, donor_layer: str | None, container):
        if donor_layer is not None and donor_layer in container.obsm:
            return _rows(container.obsm[donor_layer])
        if container.n_vars == 0:
            if donor_layer is not None:
                raise KeyError(f"'{donor_layer}' not found in the obsm of the {self.level} level.")
            return None
        mods = list(container.mod.values()) if isinstance(container, MuData) else [container]
        matrices = [m for m in (_resolve(mod, donor_layer) for mod in mods) if m is not None]
        if not matrices:
            if donor_layer is not None:
                raise KeyError(f"'{donor_layer}' not found in D.")
            return None
        return matrices[0] if len(matrices) == 1 else np.concatenate([asarray(m) for m in matrices], axis=1)

    def _cell_features(self, cell_layer: str | None) -> list:
        C = self.dd.C
        if isinstance(C, MuData):
            if cell_layer is not None and cell_layer in C.obsm:
                return [_rows(C.obsm[cell_layer])]
            parts = [modality(C, name) for name in C.mod]
            if any(rows is not None for _, rows in parts):
                raise ValueError("Bags need every modality to cover every cell, select cells with dd.complete_cases().")
            matrices = [_resolve(adata, cell_layer) for adata, _ in parts]
            matrices = [m for m in matrices if m is not None]
        else:
            matrices = [m for m in [_resolve(C, cell_layer)] if m is not None]
        if not matrices:
            raise KeyError(f"Cell features '{cell_layer}' not found in C.")
        return matrices

    def __len__(self) -> int:
        return len(self._positions)

    def balanced_weights(self, field: str = "donor_y") -> torch.Tensor:
        """Weights for a ``WeightedRandomSampler`` that draws every class of ``field`` equally often.

        Parameters
        ----------
        field
            A unit-level field of the items, by default the label.

        Returns
        -------
        torch.Tensor
            One weight per item of the dataset.
        """
        if field not in self._donor_fields:
            raise KeyError(f"Field '{field}' is not available.")
        labels = self._donor_fields[field][self._positions]
        classes, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts[np.searchsorted(classes, labels)]
        return torch.tensor(weights, dtype=torch.double)

    def _bag_rows(self, position: int) -> tuple[np.ndarray, np.ndarray | None]:
        rows = np.arange(self._offsets[position], self._offsets[position + 1])
        if self.n_cells is None:
            return rows, None
        if rows.size >= self.n_cells:
            chosen = np.sort(self._rng.choice(rows.size, size=self.n_cells, replace=False))
            return rows[chosen], np.ones(self.n_cells, dtype=bool)
        mask = np.zeros(self.n_cells, dtype=bool)
        mask[: rows.size] = True
        return rows, mask

    def _pad(self, values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None or values.shape[0] == mask.size:
            return values
        padded = np.zeros((mask.size, *values.shape[1:]), dtype=values.dtype)
        padded[: values.shape[0]] = values
        return padded

    def __getitem__(self, idx: int) -> dict:
        position = int(self._positions[idx])
        rows, mask = self._bag_rows(position)
        selector = slice(rows[0], rows[-1] + 1) if rows[-1] - rows[0] + 1 == rows.size else rows

        sample: dict[str, Any] = {"donor_id": self.donor_names[position], "donor_x": None}
        if self._donor_matrix is not None:
            donor_x = np.ravel(get_array(self._donor_matrix[position : position + 1]))
            sample["donor_x"] = torch.tensor(donor_x, dtype=torch.float32)
        for name in DONOR_FIELDS[1:]:
            values = self._donor_fields.get(name)
            sample[name] = None if values is None else torch.tensor(values[position], dtype=torch.float32)

        cell_x = np.concatenate([get_array(m[selector]) for m in self._cell_matrices], axis=1)
        sample["cell_x"] = torch.tensor(self._pad(np.asarray(cell_x, dtype=np.float32), mask))
        for name in CELL_FIELDS[1:-1]:
            values = self._cell_fields.get(name)
            sample[name] = None if values is None else torch.tensor(self._pad(values[rows], mask))
        sample["cell_mask"] = None if mask is None else torch.from_numpy(mask)
        return sample


def mil_collate_fn(batch: list[dict]) -> dict:
    """Collate items of :class:`MILDataset` into a batch.

    Donor fields are stacked into tensors with a leading batch dimension. Cell fields are stacked
    as well when bags have a fixed size, and kept as lists of per donor tensors otherwise, which
    preserves bags of different lengths.

    Parameters
    ----------
    batch
        Items returned by :meth:`MILDataset.__getitem__`.

    Returns
    -------
    dict
        Batched fields, with ``"donor_id"`` as a list. Fields that are missing in the items are left
        out.
    """
    fixed = batch[0].get("cell_mask") is not None
    sample: dict[str, Any] = {"donor_id": [item["donor_id"] for item in batch]}
    for key in DONOR_FIELDS + CELL_FIELDS:
        values = [item[key] for item in batch if item.get(key) is not None]
        if not values:
            continue
        sample[key] = torch.stack(values) if key in DONOR_FIELDS or fixed else values
    return sample
