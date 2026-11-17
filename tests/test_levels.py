import warnings

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

import donordata
from donordata import DonorData, read_dd, read_h5_dd, read_zarr_dd, validate
from donordata._core import _utils

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def _open(path):
    return h5py.File(path) if str(path).endswith(".h5") else zarr.open(str(path), mode="r")


def test_from_anndata_with_samples_places_columns_at_their_level(dd3, patients):
    assert dd3.level_names == ("donor", "sample", "cell")
    assert set(patients.columns) <= set(dd3.obs.columns)
    assert {"day", "tissue", "patient"} <= set(dd3.S.columns)
    assert not {"day", "tissue", "age"} & set(dd3.C.obs.columns)
    assert dd3.n_samples == dd3.S.shape[0]
    np.testing.assert_array_equal(dd3.n_samples_per_donor.to_numpy(), [1 + (i % 3) for i in range(8)][: dd3.n_donors])
    assert np.all(np.diff(dd3.sample_donor_codes) >= 0)
    assert np.all(np.diff(dd3.sample_codes) >= 0)
    np.testing.assert_array_equal(dd3.donor_codes, dd3.sample_donor_codes[dd3.sample_codes])
    validate(dd3)
    levels = dd3.infer_levels()
    assert levels.loc["day", "level"] == "sample"
    assert levels.loc["cell_type", "level"] == "cell"


def test_samples_with_mixed_donors_are_rejected(visits):
    visits.obs.loc[visits.obs_names[0], "sample"] = visits.obs["sample"].iloc[-1]
    if visits.obs["patient"].iloc[0] != visits.obs["patient"].iloc[-1]:
        with pytest.raises(ValueError, match="several donors"):
            DonorData.from_anndata(visits, donor_id="patient", sample_id="sample")


def test_filters_cascade_across_levels(dd3):
    later = dd3.filter_samples("day > 0")
    assert (later.S["day"] > 0).all()
    assert set(later.obs_names) == set(later.S["patient"])
    assert set(later.C.obs["sample"]) == set(later.S.index)
    validate(later)
    old = dd3.filter_samples("age > 50 and tissue == 'blood'")
    assert (old.obs["age"] > 50).all()
    donors = dd3.filter_donors(["P1", "P5"])
    assert set(donors.S["patient"]) == {"P1", "P5"}
    b_cells = dd3.filter_cells("cell_type == 'B'")
    assert (b_cells.n_cells_per_sample > 0).all()
    validate(b_cells)
    with pytest.raises(ValueError, match="no donors"):
        dd3.filter_samples("day > 1000")


def test_level_aware_operations(dd3, visits):
    fractions = dd3.composition("cell_type", level="sample")
    assert fractions.shape[0] == dd3.n_samples
    assert "cell_type_fractions" in dd3.levels["sample"].obsm
    dd3.aggregate(obsm="X_pca", key_added="pca", level="sample")
    expected = visits.obsm["X_pca"][(visits.obs["sample"] == dd3.S.index[0]).to_numpy()].mean(axis=0)
    np.testing.assert_allclose(dd3.levels["sample"].obsm["pca"][0], expected)
    bulk = dd3.pseudobulk(layer="counts", level="sample")
    assert bulk.n_obs == dd3.n_samples
    assert {"sample", "patient", "day", "age", "n_cells", "total_counts"} <= set(bulk.obs.columns)
    np.testing.assert_allclose(bulk.obs["total_counts"], np.asarray(bulk.X.sum(axis=1)).ravel())
    dd3.push_obs("age", to_level="sample")
    assert "age" in dd3.S.columns
    dd3.push_obs("day", level="sample")
    np.testing.assert_array_equal(dd3.C.obs["day"].to_numpy(), dd3.S["day"].to_numpy()[dd3.sample_codes])
    collected = dd3.pull_obs({"n_counts": "median_counts"}, func="median", level="sample")
    assert collected == ["median_counts"] and "median_counts" in dd3.S.columns
    capped = dd3.subsample_cells(3, level="sample")
    assert (capped.n_cells_per_sample <= 3).all()
    big = dd3.filter_donors_by_cells(10, level="sample")
    assert (big.n_cells_per_sample >= 10).all()
    first = dd3.S.index[0]
    assert dd3.cells_of(first, level="sample").obs["sample"].unique().tolist() == [first]


def test_nest_flatten_and_collapse(dd, dd3):
    dd.obs["site"] = dd.obs["site"].astype(str)
    dd.obsm["emb"] = np.arange(dd.n_donors * 2.0).reshape(-1, 2)
    nested = dd.nest("site")
    assert nested.level_names == ("donor", "sample", "cell")
    assert nested.donor_id == "site" and nested.sample_id == "patient"
    assert nested.obs_names.tolist() == ["A", "B"]
    positions = dd.obs_names.get_indexer(nested.levels["sample"].obs_names)
    np.testing.assert_array_equal(nested.levels["sample"].obsm["emb"], dd.obsm["emb"][positions])
    validate(nested)
    flat = nested.flatten()
    assert flat.level_names == ("donor", "cell") and flat.donor_id == "patient"
    assert "site" in flat.obs.columns and flat.n_donors == dd.n_donors
    validate(flat)
    sites = dd3.collapse("site")
    assert sites.level_names == ("donor", "sample", "cell")
    assert set(sites.S["site"]) == {"A", "B"}
    validate(sites)


def test_get_df_moves_values_between_levels(dd3, visits):
    cells = dd3.get_df(["age", "day", "cell_type", "gene1"], level="cell")
    assert cells.index.equals(dd3.C.obs_names)
    np.testing.assert_array_equal(cells["age"].to_numpy(), dd3.obs["age"].to_numpy()[dd3.donor_codes])
    np.testing.assert_array_equal(cells["day"].to_numpy(), dd3.S["day"].to_numpy()[dd3.sample_codes])
    donors = dd3.get_df(["age", "n_counts", "gene1", "X_pca"], level="donor")
    expected = visits.obs.groupby("patient", observed=True)["n_counts"].mean().loc[dd3.obs_names]
    np.testing.assert_allclose(donors["n_counts"], expected)
    assert [c for c in donors.columns if c.startswith("X_pca")] == [f"X_pca_{i}" for i in range(4)]
    samples = dd3.get_df(["age", "day"], level="sample")
    np.testing.assert_array_equal(samples["age"].to_numpy(), dd3.obs["age"].to_numpy()[dd3.sample_donor_codes])
    with pytest.raises(ValueError, match="composition"):
        dd3.get_df("cell_type", level="donor")
    with pytest.raises(KeyError):
        dd3.get_df("nothing")
    dd3.C.obs["age"] = 1.0
    explicit = dd3.get_df(["D:age", "C:age"], level="cell")
    assert explicit["C:age"].eq(1.0).all() and not explicit["D:age"].eq(1.0).all()


def test_partial_modalities_coverage_and_complete_cases(cells, patients):
    blood = cells[cells.obs["patient"].isin(["P0", "P1", "P2", "P3"]).to_numpy()].copy()
    marrow = cells[cells.obs["patient"].isin(["P2", "P3", "P4"]).to_numpy()].copy()
    marrow.obs_names = [f"m_{name}" for name in marrow.obs_names]
    marrow.var_names = [f"peak{i}" for i in range(marrow.n_vars)]
    mdata = md.MuData({"blood": blood, "marrow": marrow})
    mdata.obs["patient"] = pd.concat([blood.obs["patient"], marrow.obs["patient"]]).loc[mdata.obs_names].astype(str)
    dd = DonorData(D=patients, C=mdata, donor_id="patient")
    coverage = dd.coverage()
    assert coverage.loc["P0", "marrow"] == 0 and coverage.loc["P4", "blood"] == 0
    assert coverage.loc["P2", "blood"] > 0 and coverage.loc["P2", "marrow"] > 0
    both = dd.complete_cases()
    assert both.obs_names.tolist() == ["P2", "P3"]
    dd.aggregate(mod="marrow", key_added="marrow_mean")
    assert np.isnan(dd.obsm["marrow_mean"].loc["P0"]).all()
    assert np.isfinite(dd.obsm["marrow_mean"].loc["P4"]).all()
    bulk = dd.pseudobulk(mod="blood")
    assert set(bulk.obs_names) == {"P0", "P1", "P2", "P3"}
    values = dd.get_df("marrow:peak0", level="donor")
    assert values["marrow:peak0"].isna().sum() == 2


def test_composition_transforms_and_membership(dd):
    clr = dd.composition("cell_type", clr=True)
    np.testing.assert_allclose(clr.sum(axis=1), 0, atol=1e-10)
    rng = np.random.default_rng(0)
    membership = pd.DataFrame(
        rng.random((dd.n_cells, 5)) > 0.6, index=dd.C.obs_names, columns=[f"nh{i}" for i in range(5)]
    )
    dd.C.obsm["nhoods"] = membership.astype(float)
    counts = dd.composition("nhoods", normalize=False)
    expected = membership.groupby(dd.C.obs["patient"].to_numpy()).sum().loc[dd.obs_names]
    np.testing.assert_array_equal(counts.to_numpy(), expected.to_numpy())


def test_cell_weights(dd):
    weights = dd.cell_weights()
    per_donor = weights.groupby(dd.C.obs["patient"].to_numpy()).sum()
    np.testing.assert_allclose(per_donor.to_numpy(), per_donor.iloc[0])
    np.testing.assert_allclose(weights.mean(), 1)
    balanced = dd.cell_weights(by="disease")
    per_class = balanced.groupby(dd.obs["disease"].to_numpy()[dd.donor_codes]).sum()
    np.testing.assert_allclose(per_class.to_numpy(), per_class.iloc[0])


def test_grouped_splits_keep_groups_together(dd):
    train, test = dd.train_test_split(0.5, groups="site", random_state=0)
    assert set(train.obs["site"]).isdisjoint(test.obs["site"])
    for fold_train, fold_test in dd.kfold(2, groups="site"):
        assert set(fold_train.obs["site"]).isdisjoint(fold_test.obs["site"])


def test_splits_keep_samples_of_a_donor_together(dd3):
    train, test = dd3.train_test_split(0.25, random_state=2)
    assert set(train.S["patient"]).isdisjoint(test.S["patient"])
    assert train.n_samples + test.n_samples == dd3.n_samples


def test_namespace_registration(dd):
    @donordata.register_donordata_namespace("toy")
    class Toy:
        def __init__(self, dd):
            self._dd = dd

        def largest(self):
            return self._dd.n_cells_per_donor.idxmax()

    assert dd.toy.largest() == dd.n_cells_per_donor.idxmax()
    assert dd.toy is dd.toy
    delattr(DonorData, "toy")


def test_update_after_in_place_edit(dd):
    dd.C.obs["patient"] = dd.C.obs["patient"].replace({"P0": "P1"})
    dd.update()
    assert "P0" not in dd.obs_names
    validate(dd)


def test_donor_column_in_cells_must_agree(visits):
    from donordata import DonorData

    dd = DonorData.from_anndata(visits, donor_id="patient", sample_id="sample")
    cells = dd.C.copy()
    cells.obs["patient"] = cells.obs["patient"].astype(str)
    cells.obs.iloc[0, cells.obs.columns.get_loc("patient")] = "P7" if cells.obs["patient"].iloc[0] != "P7" else "P0"
    with pytest.raises(ValueError, match="disagrees"):
        dd.C = cells


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_roundtrip_three_levels(tmp_path, dd3, extension):
    dd3.composition("cell_type", level="sample")
    dd3.levels["sample"].obsp["dist"] = np.eye(dd3.n_samples)
    dd3.obsm["emb"] = np.ones((dd3.n_donors, 2))
    path = tmp_path / f"visits{extension}"
    dd3.write_dd(str(path))
    loaded = read_h5_dd(path) if extension.endswith("h5") else read_zarr_dd(path)
    assert loaded.level_names == dd3.level_names
    pd.testing.assert_frame_equal(loaded.S, dd3.S)
    pd.testing.assert_frame_equal(
        loaded.levels["sample"].obsm["cell_type_fractions"],
        dd3.levels["sample"].obsm["cell_type_fractions"],
        check_names=False,
    )
    np.testing.assert_array_equal(loaded.levels["sample"].obsp["dist"], np.eye(dd3.n_samples))
    np.testing.assert_array_equal(loaded.obsm["emb"], dd3.obsm["emb"])
    validate(loaded)


def test_concat_three_levels(dd3):
    first = dd3.filter_donors(["P0", "P1"])
    second = dd3.filter_donors(["P2"])
    first.composition("cell_type", level="sample")
    second.composition("cell_type", level="sample")
    merged = donordata.concat([first, second])
    assert merged.n_samples == first.n_samples + second.n_samples
    assert merged.levels["sample"].obsm["cell_type_fractions"].index.equals(merged.levels["sample"].obs_names)
    validate(merged)


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_ehrdata_donor_level(tmp_path, cells, patients, extension, ehrdata):
    ed = ehrdata
    rng = np.random.default_rng(0)
    series = rng.normal(size=(len(patients), 3, 4))
    edata = ed.EHRData(
        X=series.mean(axis=2),
        obs=patients.copy(),
        var=pd.DataFrame(index=["sbp", "dbp", "crp"]),
        layers={"tem_data": series},
        tem=pd.DataFrame({"day": [0, 7, 14, 28]}, index=[f"t{i}" for i in range(4)]),
    )
    dd = DonorData(D=edata, C=cells, donor_id="patient")
    sub = dd.filter_donors(["P1", "P3"])
    assert type(sub.D).__name__ == "EHRData"
    assert sub.D.layers["tem_data"].shape == (2, 3, 4)
    assert "tem" in repr(dd)
    path = tmp_path / f"ehr{extension}"
    dd.write_dd(str(path))
    store = _open(path)
    assert set(store.keys()) == {"D", "C"}
    h5 = extension == ".dd.h5"
    assert store["D"].attrs["ehrdata-encoding-type" if h5 else "encoding-type"] == "ehrdata"
    np.testing.assert_allclose(ed.io.read_elem(store["D"]).layers["tem_data"], series)
    plain = read_elem(store["D"] if h5 else store["D"]["anndata"])
    assert type(plain).__name__ == "AnnData" and plain.n_obs == len(patients)
    if h5:
        store.close()
    loaded = read_dd(path)
    assert type(loaded.D).__name__ == "EHRData"
    np.testing.assert_allclose(loaded.D.layers["tem_data"], dd.D.layers["tem_data"])
    assert loaded.D.tem["day"].tolist() == [0, 7, 14, 28]
    crp = dd.get_df("crp", level="cell")
    np.testing.assert_allclose(crp["crp"].to_numpy(), dd.D.X[:, 2][dd.donor_codes])


def test_donor_level_is_built_from_the_sample_table(dd3):
    samples = dd3.S.copy()
    built = DonorData(S=samples, C=dd3.C, donor_id=dd3.donor_id, sample_id=dd3.sample_id)
    assert list(built.D.index) == list(dd3.D.index)
    assert built.D.shape[1] == 0
    assert built.n_samples == dd3.n_samples and built.n_cells == dd3.n_cells
    np.testing.assert_array_equal(built.donor_codes, dd3.donor_codes)
    with pytest.raises(TypeError, match="sample table S"):
        DonorData(C=dd3.C, donor_id=dd3.donor_id)


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_ehrdata_sample_level(tmp_path, dd3, extension, ehrdata):
    ed = ehrdata
    rng = np.random.default_rng(1)
    samples = dd3.S.copy()
    series = rng.normal(size=(len(samples), 2, 3))
    edata = ed.EHRData(
        X=series.mean(axis=2),
        obs=samples,
        var=pd.DataFrame(index=["crp", "il6"]),
        layers={"tem_data": series},
        tem=pd.DataFrame({"hour": [0, 12, 24]}, index=["h0", "h12", "h24"]),
    )
    dd = DonorData(D=dd3.D, S=edata, C=dd3.C, donor_id=dd3.donor_id, sample_id=dd3.sample_id)
    assert type(dd.S).__name__ == "EHRData"
    sub = dd.filter_donors(dd.obs_names[:2])
    assert sub.S.layers["tem_data"].shape == (sub.n_samples, 2, 3)
    path = tmp_path / f"ehr_samples{extension}"
    sub.write_dd(str(path))
    loaded = read_dd(path)
    assert type(loaded.S).__name__ == "EHRData"
    np.testing.assert_allclose(loaded.S.layers["tem_data"], sub.S.layers["tem_data"])


def test_group_sum_of_a_sparse_matrix_matches_dense_without_deprecations():
    dense = np.arange(12.0).reshape(4, 3)
    codes = np.array([0, 1, 0, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy 2 deprecates building a csr_matrix
        matrix = sp.csr_matrix(dense)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for transform in (None, np.log1p):
            expected = _utils.group_sum(dense, codes, 2, transform)
            np.testing.assert_allclose(_utils.group_sum(matrix, codes, 2, transform), expected)
            np.testing.assert_allclose(_utils.group_sum(sp.csr_array(dense), codes, 2, transform), expected)


def _rich_ehrdata(ed, patients):
    """EHRData with a 3D X, a sparse COO layer, datetimes, object columns with None and a numeric obsm frame."""
    import sparse

    n = len(patients)
    series = np.random.default_rng(4).normal(size=(n, 3, 4))
    obs = patients.assign(
        visit=pd.date_range("2024-01-01", periods=n, freq="D"),
        count=np.array([1, None] * (n // 2), dtype=object),
        note=np.array(["a", None] * (n // 2), dtype=object),
    )
    return ed.EHRData(
        X=series,
        obs=obs,
        var=pd.DataFrame(index=["sbp", "dbp", "crp"]),
        layers={"coo": sparse.COO.from_numpy(np.where(series > 1, series, np.nan), fill_value=np.nan)},
        obsm={"profile": pd.DataFrame(series[:, :, 0], index=obs.index, columns=["g1", "g2", "g3"])},
        tem=pd.DataFrame({"time": pd.date_range("2024-01-01", periods=4, freq="h")}, index=[f"t{i}" for i in range(4)]),
    )


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_ehrdata_level_is_stored_as_ehrdata_stores_it(tmp_path, cells, patients, extension, ehrdata):
    sparse = pytest.importorskip("sparse")
    edata = _rich_ehrdata(ehrdata, patients)
    path = tmp_path / f"rich{extension}"
    DonorData(D=edata, C=cells, donor_id="patient").write_dd(str(path))
    store = _open(path)
    slots = store["D"] if extension == ".dd.h5" else store["D"]["anndata"]
    assert "profile" in slots["obsm_columns"]
    # packed into one array
    assert slots["obsm"]["profile"].attrs["encoding-type"] == "array"
    if extension == ".dd.h5":
        store.close()
    loaded = read_dd(path)
    D = loaded.D
    np.testing.assert_allclose(D.X, edata.X)
    assert isinstance(D.layers["coo"], sparse.COO)
    np.testing.assert_allclose(D.layers["coo"].todense(), edata.layers["coo"].todense())
    assert D.obs["visit"].iloc[1] == "2024-01-02T00:00:00" and D.tem["time"].iloc[1] == "2024-01-01T01:00:00"
    assert D.obs["count"].dtype == "Int64" and D.obs["count"].isna().sum() == len(patients) // 2
    assert D.obs["note"].iloc[0] == "a" and D.obs["note"].isna().sum() == len(patients) // 2
    pd.testing.assert_frame_equal(D.obsm["profile"], edata.obsm["profile"], check_names=False)
    if extension == ".dd.zarr":
        assert isinstance(loaded.C.X, ad.abc.CSRDataset)
    validate(loaded)


def test_x_chunks_for_an_ehrdata_level_warn(tmp_path, cells, patients, ehrdata):
    dd = DonorData(D=_rich_ehrdata(ehrdata, patients), C=cells, donor_id="patient")
    with pytest.warns(UserWarning, match="x_chunks does not apply to the EHRData level D"):
        dd.write_zarr_dd(str(tmp_path / "chunks.dd.zarr"), x_chunks={"D": (2, 2), "C": (8, 4)})
    np.testing.assert_allclose(read_zarr_dd(tmp_path / "chunks.dd.zarr").D.X, dd.D.X)
    # a shared shape is meant for C
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dd.write_zarr_dd(str(tmp_path / "shared.dd.zarr"), x_chunks=(8, 4))


def test_ehrdata_cell_level_is_rejected(cells, patients, ehrdata):
    edata_cells = ehrdata.EHRData(X=cells.X, obs=cells.obs, var=cells.var)
    with pytest.raises(TypeError, match="an EHRData belongs to D or S"):
        DonorData(D=patients, C=edata_cells, donor_id="patient")


def test_ehrdata_level_without_stamp_is_rejected(tmp_path, cells, patients, ehrdata):
    path = tmp_path / "dev.dd.h5"
    DonorData(D=_rich_ehrdata(ehrdata, patients), C=cells, donor_id="patient").write_dd(str(path))
    with h5py.File(path, "a") as f:
        del f["D"].attrs["ehrdata-encoding-type"]
    with pytest.raises(ValueError, match="layout of a development version"):
        read_dd(path)
