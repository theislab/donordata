import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from donordata.ml import MILDataset, mil_collate_fn


def test_variable_size_bags(dd):
    dataset = MILDataset(dd, donor_labels_key="disease", cell_batch_key="cell_type")
    assert len(dataset) == dd.n_donors
    assert dataset.encodings["disease"] == ["covid", "healthy"]
    item = dataset[3]
    donor = dd.obs_names[3]
    assert item["donor_id"] == donor
    np.testing.assert_allclose(item["cell_x"].numpy(), dd.cells_of(donor).X.toarray())
    assert item["donor_x"] is None
    assert item["cell_mask"] is None
    assert item["donor_y"].item() == dataset.encodings["disease"].index(dd.obs.loc[donor, "disease"])
    batch = mil_collate_fn([dataset[0], dataset[1]])
    assert batch["donor_y"].shape == (2,)
    assert isinstance(batch["cell_x"], list)
    assert batch["donor_id"] == list(dd.obs_names[:2])


def test_fixed_size_bags_and_loader(dd):
    dd.obsm["emb"] = np.random.default_rng(0).normal(size=(dd.n_donors, 3))
    dataset = MILDataset(
        dd,
        donor_layer="emb",
        donor_labels_key="age",
        cell_layer="X_pca",
        cell_labels_key="cell_type",
        n_cells=15,
        random_state=0,
        split_donors=["P1", "P6", "P2"],
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=mil_collate_fn)
    batch = next(iter(loader))
    assert batch["cell_x"].shape == (3, 15, 4)
    assert batch["cell_mask"].shape == (3, 15)
    assert batch["donor_x"].shape == (3, 3)
    sizes = dd.n_cells_per_donor.loc[["P1", "P6", "P2"]].to_numpy()
    np.testing.assert_array_equal(batch["cell_mask"].sum(1).numpy(), np.minimum(sizes, 15))
    np.testing.assert_allclose(batch["donor_y"].numpy(), dd.obs.loc[["P1", "P6", "P2"], "age"].to_numpy())
    padded = ~batch["cell_mask"]
    assert torch.all(batch["cell_x"][padded] == 0)


def test_subsampled_bag_comes_from_donor(dd):
    dataset = MILDataset(dd, cell_layer="X_pca", n_cells=5, random_state=1)
    item = dataset[0]
    own = dd.cells_of(dd.obs_names[0]).obsm["X_pca"]
    for row in item["cell_x"].numpy():
        assert np.isclose(own, row).all(axis=1).any()


def test_missing_explicit_key_raises(dd):
    with pytest.raises(KeyError, match="nope"):
        MILDataset(dd, donor_labels_key="nope")
    with pytest.raises(KeyError):
        MILDataset(dd, split_donors=["P0", "unknown"])
    with pytest.raises(ValueError):
        MILDataset(dd, split_donors=["P0"], split_indices=[0])


def test_genotype_donor_features(dd_genotypes):
    dataset = MILDataset(dd_genotypes, split_indices=[0, 2])
    item = dataset[1]
    np.testing.assert_allclose(item["donor_x"].numpy(), dd_genotypes.D.X[2])


def test_sample_level_bags_and_balanced_weights(dd3):
    dataset = MILDataset(dd3, level="sample", donor_labels_key="disease", cell_layer="X_pca")
    assert len(dataset) == dd3.n_samples
    item = dataset[0]
    assert item["cell_x"].shape[0] == dd3.n_cells_per_sample.iloc[0]
    weights = dataset.balanced_weights()
    labels = dataset._donor_fields["donor_y"]
    totals = {label: weights[labels == label].sum().item() for label in np.unique(labels)}
    np.testing.assert_allclose(list(totals.values()), 1.0)
