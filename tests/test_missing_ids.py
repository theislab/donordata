"""Missing donor, sample and group values are dropped or ignored with a warning, in every function alike."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import donordata
from donordata import DonorData

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


def _grouped_dd(cell_types, sex=("F", "M", "F", "M")):
    """Four donors with two cells each, a cell type per cell and a sex per donor."""
    donors = pd.DataFrame({"sex": list(sex)}, index=pd.Index(["a", "b", "c", "d"], name="donor_id"))
    obs = pd.DataFrame(
        {"donor_id": np.repeat(donors.index, 2), "cell_type": np.array(cell_types, dtype=object)},
        index=[f"c{i}" for i in range(8)],
    )
    return DonorData(D=donors, C=AnnData(np.ones((8, 2)), obs=obs), donor_id="donor_id")


MISSING_GROUPS = [None, np.nan, "NA", "nan", "None", ""]


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
def test_composition_and_pseudobulk_ignore_cells_with_a_missing_group(caplog, missing):
    dd = _grouped_dd(["T", missing, "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        fractions = dd.composition("cell_type")
        profiles = dd.pseudobulk(groupby="cell_type")
    assert list(fractions.columns) == ["B", "T"]
    assert fractions.loc["a"].tolist() == [0.0, 1.0]  # over the one cell of donor a that has a cell type
    assert sorted(profiles.obs_names) == ["a_T", "b_B", "b_T", "c_B", "d_T"]
    assert profiles.obs.loc["a_T", "n_cells"] == 1
    assert caplog.text.count("Ignoring 1 cells with a missing 'cell_type'") == 2
    assert "'c1'" in caplog.text
    assert dd.n_cells == 8  # the object keeps the cell


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
def test_subsample_cells_drops_cells_with_a_missing_group(caplog, missing):
    dd = _grouped_dd(["T", missing, "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        result = dd.subsample_cells(1, groupby="cell_type")
    assert "c1" not in result.C.obs_names
    assert result.n_cells == 5  # one cell per donor and cell type
    assert "Dropping 1 cells with a missing 'cell_type'" in caplog.text


@pytest.mark.parametrize("missing", MISSING_GROUPS, ids=repr)
@pytest.mark.parametrize("as_column", [True, False], ids=["column", "values"])
def test_splits_drop_donors_with_a_missing_stratum(caplog, missing, as_column):
    sex = ["F", "M", missing, "M"]
    dd = _grouped_dd(["T"] * 8, sex=sex)
    stratify = "sex" if as_column else sex
    with caplog.at_level("WARNING"):
        train, test = dd.train_test_split(test_size=1, stratify=stratify)
        folds = list(dd.kfold(3, stratify=stratify))
    assert sorted([*train.obs_names, *test.obs_names]) == ["a", "b", "d"]
    for fold_train, fold_test in folds:
        assert sorted([*fold_train.obs_names, *fold_test.obs_names]) == ["a", "b", "d"]
    assert sorted(name for _, fold_test in folds for name in fold_test.obs_names) == ["a", "b", "d"]
    key = "sex" if as_column else "stratify"
    assert caplog.text.count(f"Dropping 1 donors with a missing '{key}'") == 2
    assert "'c'" in caplog.text


def test_splits_drop_donors_with_a_missing_group(caplog):
    dd = _grouped_dd(["T"] * 8)
    dd.obs["site"] = ["x", "NA", "y", "z"]
    with caplog.at_level("WARNING"):
        train, test = dd.train_test_split(test_size=1, groups="site")
    assert sorted([*train.obs_names, *test.obs_names]) == ["a", "c", "d"]
    assert "Dropping 1 donors with a missing 'site'" in caplog.text


def test_grouping_without_missing_values_does_not_warn(caplog):
    dd = _grouped_dd(["T", "Nancy", "T", "B", "B", "B", "T", "T"])
    with caplog.at_level("WARNING"):
        assert list(dd.composition("cell_type").columns) == ["B", "Nancy", "T"]
        dd.pseudobulk(groupby="cell_type")
        assert dd.subsample_cells(1, groupby="cell_type").n_cells == 6
        train, test = dd.train_test_split(test_size=2, stratify="sex")
    assert train.n_donors + test.n_donors == 4
    assert "Dropping" not in caplog.text and "Ignoring" not in caplog.text


def test_splits_match_a_stratify_series_by_name():
    dd = _grouped_dd(["T"] * 8, sex=["F", "F", "M", "M"])
    other_order = dd.obs["sex"].loc[["c", "a", "d", "b"]]
    with_extra = pd.concat([other_order, pd.Series(["F"], index=["not a donor"])])
    for stratify in ("sex", other_order, with_extra, dd.obs["sex"].to_numpy(), dd.obs["sex"].tolist()):
        for seed in range(8):
            _, test = dd.train_test_split(test_size=2, stratify=stratify, random_state=seed)
            assert sorted(test.obs["sex"]) == ["F", "M"]
            for _, fold_test in dd.kfold(2, stratify=stratify, random_state=seed):
                assert sorted(fold_test.obs["sex"]) == ["F", "M"]


@pytest.mark.parametrize(
    "stratify",
    [
        pd.Series(["F", "F", "M", "M"]),
        pd.Series(["F", "F", "M"], index=["a", "b", "c"]),
        pd.Series(["F", "F", "M", "M", "M"], index=["a", "b", "c", "d", "d"]),
    ],
    ids=["default index", "donor missing", "duplicate donor"],
)
def test_splits_reject_a_stratify_series_that_cannot_be_matched_by_name(stratify):
    dd = _grouped_dd(["T"] * 8)
    with pytest.raises(ValueError, match="cannot be matched by name"):
        dd.train_test_split(test_size=2, stratify=stratify)
    with pytest.raises(ValueError, match="cannot be matched by name"):
        next(dd.kfold(2, stratify=stratify))
