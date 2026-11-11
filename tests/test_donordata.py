import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import donordata
from donordata import DonorData, DonorDataSchemaError, validate

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def test_from_anndata_moves_constant_columns(cells, dd, patients):
    assert dd.D_type == "table"
    assert isinstance(dd.D, pd.DataFrame)
    assert set(dd.obs.columns) == set(patients.columns)
    assert not set(patients.columns) & set(dd.C.obs.columns)
    assert {"cell_type", "n_counts", "patient"} <= set(dd.C.obs.columns)
    pd.testing.assert_frame_equal(dd.obs.loc[patients.index, patients.columns], patients, check_names=False)
    assert dd.obs["sex"].dtype == patients["sex"].dtype
    assert set(patients.columns) <= set(cells.obs.columns)
    assert dd.C.X is not None
    validate(dd)


def test_from_anndata_keep_in_cells_and_explicit_columns(cells):
    dd = DonorData.from_anndata(cells, donor_id="patient", columns=["age"], keep_in_cells=True)
    assert list(dd.obs.columns) == ["age"]
    assert "age" in dd.C.obs.columns
    with pytest.raises(ValueError, match="vary within donors"):
        DonorData.from_anndata(cells, donor_id="patient", columns=["cell_type"])


def test_cells_grouped_by_donor_in_G_order(dd, cells):
    codes = dd.donor_codes
    assert np.all(np.diff(codes) >= 0)
    for donor in dd.obs_names:
        original = cells.obs_names[cells.obs["patient"] == donor]
        assert dd.cells_of(donor).obs_names.tolist() == original.tolist()
    offsets = dd.cell_offsets
    assert offsets[-1] == dd.n_cells
    np.testing.assert_array_equal(np.diff(offsets), dd.n_cells_per_donor.to_numpy())


def test_table_with_donor_column_and_unnamed_index(cells, patients):
    with_column = patients.reset_index()
    dd = DonorData(D=with_column, C=cells, donor_id="patient")
    assert dd.obs.index.name == "patient"
    assert dd.n_donors == len(patients)
    unnamed = patients.rename_axis(None)
    assert DonorData(D=unnamed, C=cells, donor_id="patient").n_donors == len(patients)


def test_integer_donor_identifiers(cells, patients):
    table = patients.reset_index(drop=True)
    cells = cells.copy()
    cells.obs["patient"] = cells.obs["patient"].str[1:].astype(int)
    dd = DonorData(D=table, C=cells, donor_id="patient")
    assert dd.n_donors == len(table)
    assert dd.n_cells == cells.n_obs


def test_unmatched_donors_are_dropped(cells, patients):
    table = pd.concat([patients, patients.iloc[:1].rename(index={"P0": "P99"})])
    dd = DonorData(D=table, C=cells[cells.obs["patient"] != "P1"], donor_id="patient")
    assert "P99" not in dd.obs_names
    assert "P1" not in dd.obs_names
    assert dd.n_donors == len(patients) - 1


def test_errors(cells, patients):
    with pytest.raises(ValueError, match="No overlapping donors"):
        DonorData(D=patients.rename(index=lambda x: x + "x"), C=cells, donor_id="patient")
    with pytest.raises(ValueError, match="unique"):
        DonorData(D=pd.concat([patients, patients]), C=cells, donor_id="patient")
    with pytest.raises(ValueError, match="not found in C.obs"):
        DonorData(D=patients, C=cells, donor_id="nope")
    with pytest.raises(TypeError):
        DonorData(D=patients.to_numpy(), C=cells, donor_id="patient")


def test_anndata_G(dd_genotypes, genotypes):
    assert dd_genotypes.D_type == "anndata"
    assert dd_genotypes.D is genotypes
    dd_genotypes.obsm["emb"] = np.zeros((dd_genotypes.n_donors, 2))
    assert "emb" in genotypes.obsm
    assert dd_genotypes.shape == (8, 5, dd_genotypes.n_cells, 12)
    sub = dd_genotypes["P2", "snp1"]
    assert sub.shape[:2] == (1, 1)
    assert set(sub.C.obs["patient"]) == {"P2"}


def test_mudata_C(cells, patients):
    mdata = md.MuData({"rna": cells, "prot": cells[:, :3].copy()})
    mdata.obs["patient"] = cells.obs["patient"]
    dd = DonorData(D=patients, C=mdata, donor_id="patient")
    assert dd.n_cells == cells.n_obs
    sub = dd.filter_donors(["P0", "P1"])
    assert set(sub.C.obs["patient"]) == {"P0", "P1"}
    pseudobulk = dd.pseudobulk(mod="prot")
    assert pseudobulk.shape == (dd.n_donors, 3)
    with pytest.raises(ValueError, match="mod="):
        dd.pseudobulk()


def test_obsm_and_obsp_follow_selections(dd):
    embedding = pd.DataFrame(np.arange(dd.n_donors * 2).reshape(-1, 2), index=dd.obs_names, columns=["a", "b"])
    dd.obsm["emb"] = embedding
    dd.obsp["dist"] = np.arange(dd.n_donors**2, dtype=float).reshape(dd.n_donors, dd.n_donors)
    sub = dd.filter_donors(["P5", "P2"])
    assert sub.obs_names.tolist() == ["P2", "P5"]
    assert sub.obsm["emb"].index.tolist() == ["P2", "P5"]
    np.testing.assert_array_equal(sub.obsp["dist"], dd.obsp["dist"][np.ix_([2, 5], [2, 5])])
    assert dd.n_donors == 8


def test_filter_donors_selection_kinds(dd):
    by_query = dd.filter_donors("age > 40 and sex == 'F'")
    expected = dd.obs_names[(dd.obs["age"] > 40) & (dd.obs["sex"] == "F")]
    assert by_query.obs_names.tolist() == expected.tolist()
    by_callable = dd.filter_donors(lambda obs: obs["disease"] == "covid")
    assert set(by_callable.C.obs["patient"]) == set(dd.obs_names[dd.obs["disease"] == "covid"])
    by_mask = dd.filter_donors(np.arange(dd.n_donors) < 3)
    assert by_mask.n_donors == 3
    with pytest.raises(ValueError, match="length"):
        dd.filter_donors(np.array([True, False]))


def test_filter_inplace(dd):
    assert dd.filter_donors("site == 'A'", inplace=True) is None
    assert set(dd.obs["site"]) == {"A"}
    validate(dd)


def test_boolean_series_masks_align_by_index(cells, patients):
    dd = DonorData.from_anndata(cells, donor_id="patient")
    assert not dd.C.obs_names.equals(cells.obs_names)

    b_cells = dd.filter_cells(cells.obs["cell_type"] == "B")
    assert set(b_cells.C.obs_names) == set(cells.obs_names[cells.obs["cell_type"] == "B"])

    reversed_mask = (patients["age"] > 40).iloc[::-1]
    old = dd.filter_donors(reversed_mask)
    assert set(old.D.index) == set(patients.index[patients["age"] > 40])

    from_superset = b_cells.filter_cells(cells.obs["n_counts"] > 0)
    assert from_superset.C.obs_names.equals(b_cells.C.obs_names)

    with pytest.raises(ValueError, match="missing"):
        dd.filter_cells(cells.obs["cell_type"].iloc[:5] == "B")


def test_filter_cells_drops_empty_donors(dd):
    only_p0 = dd.filter_cells(lambda obs: obs["patient"] == "P0")
    assert only_p0.obs_names.tolist() == ["P0"]
    b_cells = dd.filter_cells("cell_type == 'B'")
    assert set(b_cells.C.obs["cell_type"]) == {"B"}
    validate(b_cells)


def test_table_G_has_no_variables(dd):
    with pytest.raises(IndexError, match="donor table"):
        dd[:, ["x"]]
    assert dd.shape[1] == 0


def test_filter_donors_by_cells(dd):
    counts = dd._group_counts("cell_type")
    assert "DC" not in counts.columns
    kept = dd.filter_donors_by_cells(5, groupby="cell_type", groups=["B", "T"])
    expected = counts.index[(counts[["B", "T"]] >= 5).all(axis=1)]
    assert kept.obs_names.tolist() == expected.tolist()
    total = dd.filter_donors_by_cells(20)
    assert (total.n_cells_per_donor >= 20).all()
    with pytest.raises(KeyError):
        dd.filter_donors_by_cells(1, groupby="cell_type", groups=["DC"])


def test_subsample_cells(dd):
    sub = dd.subsample_cells(7)
    np.testing.assert_array_equal(sub.n_cells_per_donor.to_numpy(), np.minimum(dd.n_cells_per_donor.to_numpy(), 7))
    stratified = dd.subsample_cells(2, groupby="cell_type", random_state=3)
    assert (stratified._group_counts("cell_type").to_numpy() <= 2).all()
    assert stratified.n_donors == dd.n_donors
    half = dd.subsample_cells(fraction=0.5)
    assert half.n_cells < dd.n_cells
    with pytest.raises(ValueError):
        dd.subsample_cells()


def test_composition(dd):
    fractions = dd.composition("cell_type")
    assert "cell_type_fractions" in dd.obsm
    np.testing.assert_allclose(fractions.sum(axis=1), 1)
    counts = dd.composition("cell_type", normalize=False)
    np.testing.assert_array_equal(counts.sum(axis=1).to_numpy(), dd.n_cells_per_donor.to_numpy())
    assert dd.filter_donors(["P1"]).obsm["cell_type_counts"].shape == (1, counts.shape[1])


def test_push_and_pull_obs(dd, patients):
    dd.push_obs(["age", "sex"])
    expected = patients.loc[dd.C.obs["patient"], "age"].to_numpy()
    np.testing.assert_array_equal(dd.C.obs["age"].to_numpy(), expected)
    assert isinstance(dd.C.obs["sex"].dtype, pd.CategoricalDtype)
    with pytest.raises(ValueError, match="already exist"):
        dd.push_obs("age")

    dd.C.obs["batch"] = dd.C.obs["patient"].map({d: f"b{i % 2}" for i, d in enumerate(dd.obs_names)})
    assert "batch" in dd.pull_obs(drop=True)
    assert "batch" not in dd.C.obs.columns
    assert dd.obs.loc["P1", "batch"] == "b1"

    with pytest.raises(ValueError, match="varies within donors"):
        dd.pull_obs("n_counts")
    dd.pull_obs("n_counts", func="median")
    expected = dd.C.obs.groupby("patient", observed=True)["n_counts"].median()
    np.testing.assert_allclose(dd.obs["n_counts"], expected.loc[dd.obs_names])


def test_push_obsm_and_to_anndata(dd, cells):
    dd.obsm["emb"] = np.eye(dd.n_donors)[:, :3]
    dd.push_obsm("emb")
    np.testing.assert_array_equal(dd.C.obsm["emb"], dd.obsm["emb"][dd.donor_codes])
    flat = dd.to_anndata(obsm=["emb"])
    assert {"age", "sex", "disease"} <= set(flat.obs.columns)
    assert flat.shape == dd.C.shape
    np.testing.assert_array_equal(flat.obsm["emb"], dd.C.obsm["emb"])
    assert "age" not in dd.C.obs.columns
    original = cells.obs.loc[flat.obs_names, "age"]
    np.testing.assert_array_equal(flat.obs["age"].to_numpy(), original.to_numpy())


def test_G_setter_keeps_donor_matrices(dd):
    dd.obsp["dist"] = np.arange(dd.n_donors**2, dtype=float).reshape(dd.n_donors, dd.n_donors)
    old_dist, old_names = dd.obsp["dist"].copy(), dd.obs_names.copy()
    dd.D = dd.D[dd.D["age"] > dd.D["age"].median()].assign(flag=True)
    positions = old_names.get_indexer(dd.obs_names)
    np.testing.assert_array_equal(dd.obsp["dist"], old_dist[np.ix_(positions, positions)])
    assert set(dd.C.obs["patient"]) == set(dd.obs_names)
    assert dd.obs["flag"].all()
    validate(dd)


def test_C_setter_restricts_donors(dd):
    dd.C = dd.C[dd.C.obs["patient"].isin(["P3", "P4"]).to_numpy()]
    assert dd.obs_names.tolist() == ["P3", "P4"]
    with pytest.raises(TypeError):
        dd.C = dd.obs


def test_aggregate_matrices_and_obs(dd, cells):
    dd.aggregate(key_added="mean", sync_var=True)
    for donor in ["P0", "P4"]:
        expected = np.asarray(cells[cells.obs["patient"] == donor].X.mean(axis=0)).ravel()
        np.testing.assert_allclose(dd.obsm["mean"].loc[donor].to_numpy(), expected, rtol=1e-6)
    dd.aggregate(layer="counts", func="sum", key_added="sum_T", filter_key="cell_type", filter_value="T")
    t_cells = cells[(cells.obs["patient"] == "P2") & (cells.obs["cell_type"] == "T")]
    np.testing.assert_allclose(dd.obsm["sum_T"].loc["P2"], np.asarray(t_cells.layers["counts"].sum(axis=0)).ravel())
    dd.aggregate(obsm="X_pca", func="var", key_added="pca_var")
    expected = cells.obsm["X_pca"][(cells.obs["patient"] == "P1").to_numpy()].var(axis=0, ddof=1)
    np.testing.assert_allclose(dd.obsm["pca_var"][1], expected)
    dd.aggregate(obs="n_counts", func="max", add_to_obs=True)
    assert dd.obs.loc["P3", "n_counts"] == cells.obs.loc[cells.obs["patient"] == "P3", "n_counts"].max()
    genes = dd[..., ["gene3", "gene1"]]
    assert genes.obsm["mean"].columns.tolist() == ["gene3", "gene1"]


@pytest.mark.parametrize("func", ["sum", "mean", "var", "count_nonzero", "median", "min", "max"])
def test_aggregation_functions_match_pandas(dd, func):
    from donordata._core._utils import aggregate_rows

    dense = dd.C.X.toarray()
    frame = pd.DataFrame(dense).groupby(dd.donor_codes)
    expected = {
        "sum": frame.sum(),
        "mean": frame.mean(),
        "var": frame.var(),
        "count_nonzero": frame.agg(lambda x: (x != 0).sum()),
        "median": frame.median(),
        "min": frame.min(),
        "max": frame.max(),
    }[func]
    for X in (dd.C.X, dense):
        np.testing.assert_allclose(aggregate_rows(X, dd.donor_codes, dd.n_donors, func), expected.to_numpy(), rtol=1e-5)


def test_pseudobulk(dd, cells):
    bulk = dd.pseudobulk(layer="counts")
    assert bulk.shape == (dd.n_donors, cells.n_vars)
    assert {"age", "sex", "patient", "n_cells"} <= set(bulk.obs.columns)
    np.testing.assert_allclose(bulk.X.sum(), cells.layers["counts"].sum())
    by_type = dd.pseudobulk(layer="counts", groupby="cell_type", min_cells=3)
    assert (by_type.obs["n_cells"] >= 3).all()
    row = by_type.obs.iloc[0]
    members = (cells.obs["patient"] == row["patient"]) & (cells.obs["cell_type"] == row["cell_type"])
    np.testing.assert_allclose(by_type.X[0], np.asarray(cells.layers["counts"][members.to_numpy()].sum(axis=0)).ravel())


def test_collapse(dd):
    dd.obsm["emb"] = pd.DataFrame(np.arange(dd.n_donors, dtype=float)[:, None], index=dd.obs_names, columns=["e"])
    sites = dd.collapse("site", func="mean")
    assert sites.obs_names.tolist() == ["A", "B"]
    assert sites.donor_id == "site"
    assert sites.obs["n_patient"].sum() == dd.n_donors
    assert "bmi" in sites.obs.columns
    expected = dd.obsm["emb"]["e"].groupby(dd.obs["site"].to_numpy()).mean()
    np.testing.assert_allclose(sites.obsm["emb"]["e"].to_numpy(), expected.to_numpy())
    assert sites.n_cells == dd.n_cells
    assert "patient" in sites.C.obs.columns
    validate(sites)


def test_train_test_split_and_kfold(dd):
    train, test = dd.train_test_split(0.25, stratify="disease", random_state=1)
    assert not set(train.obs_names) & set(test.obs_names)
    assert train.n_donors + test.n_donors == dd.n_donors
    assert test.n_donors == 2
    assert set(test.obs["disease"]) == {"healthy", "covid"}
    assert set(train.C.obs["patient"]).isdisjoint(test.C.obs["patient"])
    seen = []
    for fold_train, fold_test in dd.kfold(4, stratify="disease"):
        assert not set(fold_train.obs_names) & set(fold_test.obs_names)
        seen.extend(fold_test.obs_names)
    assert sorted(seen) == sorted(dd.obs_names)
    with pytest.raises(ValueError):
        dd.train_test_split(0.0)


def test_copy_is_independent(dd):
    dd.uns["note"] = {"a": 1}
    copied = dd.copy()
    copied.obs["new"] = 1
    copied.uns["note"]["a"] = 2
    copied.C.obs["x"] = 0
    assert "new" not in dd.obs.columns
    assert dd.uns["note"]["a"] == 1
    assert "x" not in dd.C.obs.columns


def test_validate_detects_corruption(dd):
    validate(dd)
    broken = dd.copy()
    broken._C = broken._C[::-1].copy()
    with pytest.raises(DonorDataSchemaError, match="grouped"):
        validate(broken)


def test_repr(dd, dd_genotypes):
    text = repr(dd)
    assert "DataFrame with 8 donors" in text
    assert "columns:" in text
    assert "AnnData object" in repr(dd_genotypes)
    assert str(dd).startswith("DonorData(n_D_obs=8")
    assert "\x1b[" not in text
    html = dd._repr_html_()
    assert html.startswith("<pre") and "DonorData(n_donors=8" in html


def test_anndata_G_obs_operations(dd_genotypes):
    dd = dd_genotypes
    assert "age" in dd.obs.columns
    sub = dd.filter_donors("age > 30")
    assert sub.D.n_obs == sub.n_donors
    dd.push_obs("age")
    assert "age" in dd.C.obs.columns
    dd.composition("cell_type")
    assert "cell_type_fractions" in dd.D.obsm


def test_empty_selection_raises(dd):
    with pytest.raises(ValueError, match="no donors"):
        dd.filter_donors("age > 1000")


def test_pull_obs_with_renaming(dd):
    names = dd.pull_obs({"n_counts": "median_n_counts"}, func="median")
    assert names == ["median_n_counts"]
    assert "n_counts" in dd.C.obs.columns
    expected = dd.C.obs.groupby("patient", observed=True)["n_counts"].median()
    np.testing.assert_allclose(dd.obs["median_n_counts"], expected.loc[dd.obs_names])


def test_pseudobulk_becomes_donor_side(dd):
    dd.obsm["emb"] = np.arange(dd.n_donors * 2.0).reshape(-1, 2)
    dd.obsp["dist"] = np.eye(dd.n_donors)
    dd.uns["sample_representations"] = ["emb"]
    bulk = dd.pseudobulk(func="mean")
    np.testing.assert_array_equal(bulk.obsm["emb"], dd.obsm["emb"])
    assert bulk.uns["sample_representations"] == ["emb"]
    dd.D = bulk
    assert dd.D_type == "anndata"
    assert dd.D is bulk
    dd.obsp["other"] = np.zeros((dd.n_donors, dd.n_donors))
    assert "other" in bulk.obsp
    subset = dd.filter_donors("age > 40")
    assert subset.D.shape == (subset.n_donors, dd.C.n_vars)
    validate(subset)


def test_donor_level_is_only_called_D(cells, genotypes):
    dd = DonorData(D=genotypes, C=cells, donor_id="patient")
    assert not hasattr(dd, "G") and not hasattr(dd, "G_type")
    with pytest.raises(TypeError):
        DonorData(G=genotypes, C=cells, donor_id="patient")


MISSING_DONORS = {
    "object None and nan": np.array(["a", "a", None, "b", np.nan], dtype=object),
    "string dtype pd.NA": pd.array(["a", "a", pd.NA, "b", pd.NA], dtype="string"),
    "categorical NaN": pd.Categorical(["a", "a", None, "b", np.nan]),
}


@pytest.mark.parametrize("donor_ids", MISSING_DONORS.values(), ids=MISSING_DONORS.keys())
def test_cells_with_missing_donor_are_dropped(caplog, donor_ids):
    obs = pd.DataFrame({"donor_id": donor_ids}, index=[f"c{i}" for i in range(5)])
    with caplog.at_level("WARNING"):
        dd = DonorData.from_anndata(AnnData(np.zeros((5, 1)), obs=obs), "donor_id")
    assert list(dd.obs_names) == ["a", "b"]
    assert dd.n_cells == 3
    assert "Dropping 2 cells with a missing 'donor_id'" in caplog.text


def test_samples_and_cells_with_missing_labels_are_dropped(caplog):
    samples = pd.DataFrame({"donor_id": ["a", np.nan, "b"]}, index=pd.Index(["s1", "s2", "s3"], name="sample_id"))
    obs = pd.DataFrame({"sample_id": ["s1", "s2", None, "s3"]}, index=[f"c{i}" for i in range(4)])
    with caplog.at_level("WARNING"):
        dd = DonorData(S=samples, C=AnnData(np.zeros((4, 1)), obs=obs), sample_id="sample_id")
    assert list(dd.obs_names) == ["a", "b"]
    assert list(dd.levels["sample"].obs_names) == ["s1", "s3"]
    assert list(dd.C.obs_names) == ["c0", "c3"]
    assert "Dropping 1 samples with a missing 'donor_id'" in caplog.text
    assert "Dropping 1 cells with a missing 'sample_id'" in caplog.text


# Strings that stand for a missing value are dropped like missing values, in any case and with spaces around them.
MISSING_STRINGS = [
    "none",
    "None",
    "NONE",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "Null",
    "NA",
    "<NA>",
    "N/A",
    "",
    " nan ",
    "NaT",
]


def test_donors_named_like_missing_values_are_dropped(caplog):
    donor_ids = ["a", "b", *MISSING_STRINGS]
    obs = pd.DataFrame({"donor_id": donor_ids}, index=[f"c{i}" for i in range(len(donor_ids))])
    with caplog.at_level("WARNING"):
        dd = DonorData.from_anndata(AnnData(np.zeros((len(obs), 1)), obs=obs), "donor_id")
    assert list(dd.obs_names) == ["a", "b"]
    assert dd.n_cells == 2
    (record,) = [r for r in caplog.records if "Dropping" in r.message]
    assert record.levelname == "WARNING"
    assert f"Dropping {len(MISSING_STRINGS)} cells with a missing 'donor_id'" in record.message
    assert "'c2' ('none')" in record.message  # names the dropped cells and their donor IDs
    assert f"and {len(MISSING_STRINGS) - 10} more" in record.message
    assert "rename them" in record.message


@pytest.mark.parametrize("extension", [None, ".dd.h5", ".dd.zarr"])
def test_donors_that_only_resemble_missing_values_are_kept(tmp_path, caplog, extension):
    names = ["Nancy", "nana", "nonexistent", "NA12878", "null_1", "N/A1"]
    obs = pd.DataFrame({"donor_id": names}, index=[f"c{i}" for i in range(len(names))])
    with caplog.at_level("WARNING"):
        dd = DonorData.from_anndata(AnnData(np.zeros((len(obs), 1)), obs=obs), "donor_id")
    if extension is not None:
        dd.write_dd(str(tmp_path / f"x{extension}"))
        dd = donordata.io.read_dd(str(tmp_path / f"x{extension}"))
    assert sorted(dd.obs_names) == sorted(names)
    assert "Dropping" not in caplog.text


def test_donor_and_sample_rows_named_like_missing_values_are_dropped(caplog):
    donors = pd.DataFrame(index=pd.Index(["a", "None"], name="donor_id"))
    samples = pd.DataFrame({"donor_id": ["a", "a", "None"]}, index=pd.Index(["s1", "NA", "s3"], name="sample_id"))
    obs = pd.DataFrame({"sample_id": ["s1", "NA", "s3"]}, index=["c0", "c1", "c2"])
    with caplog.at_level("WARNING"):
        dd = DonorData(
            D=donors, S=samples, C=AnnData(np.zeros((3, 1)), obs=obs), donor_id="donor_id", sample_id="sample_id"
        )
    assert list(dd.obs_names) == ["a"]
    assert list(dd.levels["sample"].obs_names) == ["s1"]
    assert list(dd.C.obs_names) == ["c0"]
    assert (
        "Dropping 1 donors with a missing 'donor_id' (empty, or a value such as NaN, None, 'nan' or 'NA'): 'None'."
        in (caplog.text)
    )
    assert "Dropping 1 samples with a missing 'sample_id'" in caplog.text
    assert "Dropping 1 samples with a missing 'donor_id'" in caplog.text and "'s3' ('None')" in caplog.text


def _site_dd(sites):
    donors = pd.DataFrame({"site": sites}, index=pd.Index([f"d{i}" for i in range(len(sites))], name="donor_id"))
    obs = pd.DataFrame({"donor_id": donors.index}, index=[f"c{i}" for i in range(len(sites))])
    return DonorData(D=donors, C=AnnData(np.ones((len(sites), 1)), obs=obs), donor_id="donor_id")


UNITS = {
    "nest": lambda dd: dd.nest("site"),
    "collapse": lambda dd: dd.collapse("site"),
}


@pytest.mark.parametrize("regroup", UNITS.values(), ids=UNITS.keys())
@pytest.mark.parametrize("missing", [None, np.nan, "NA", "nan", "None", ""], ids=repr)
def test_nest_and_collapse_drop_donors_with_a_missing_unit(caplog, regroup, missing):
    dd = _site_dd(np.array(["X", missing, "X", "Y"], dtype=object))
    with caplog.at_level("WARNING"):
        result = regroup(dd)
    assert list(result.obs_names) == ["X", "Y"]
    assert sorted(result.C.obs_names) == ["c0", "c2", "c3"]
    assert "Dropping 1 donors with a missing 'site'" in caplog.text
    assert "'d1'" in caplog.text  # names the dropped donor


@pytest.mark.parametrize("regroup", UNITS.values(), ids=UNITS.keys())
def test_nest_and_collapse_keep_all_donors_without_missing_units(caplog, regroup):
    with caplog.at_level("WARNING"):
        result = regroup(_site_dd(["X", "Nancy", "X", "NA12878"]))
    assert list(result.obs_names) == ["NA12878", "Nancy", "X"]
    assert result.n_cells == 4
    assert "Dropping" not in caplog.text
