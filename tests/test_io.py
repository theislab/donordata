import gc

import h5py
import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata.io import read_elem

from donordata import DonorData, read_dd, read_h5_dd, read_zarr_dd, validate

if hasattr(md, "set_options"):
    md.set_options(pull_on_update=False)


def _decorate(dd):
    dd.obsm["emb"] = pd.DataFrame(np.arange(dd.n_donors * 2.0).reshape(-1, 2), index=dd.obs_names, columns=["a", "b"])
    dd.obsm["arr"] = np.ones((dd.n_donors, 3))
    dd.obsp["dist"] = np.eye(dd.n_donors)
    dd.aggregate(key_added="pb", sync_var=True)
    dd.uns["study"] = "test"
    return dd


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_roundtrip_table_G(tmp_path, dd, extension):
    dd = _decorate(dd)
    path = tmp_path / f"cohort{extension}"
    dd.write_dd(str(path))
    loaded = read_dd(path)
    assert loaded.D_type == "table"
    assert loaded.donor_id == "patient"
    assert loaded.shape == dd.shape
    pd.testing.assert_frame_equal(loaded.obs, dd.obs)
    pd.testing.assert_frame_equal(loaded.obsm["emb"], dd.obsm["emb"])
    np.testing.assert_array_equal(loaded.obsp["dist"], dd.obsp["dist"])
    assert loaded._var_dims_to_sync == ["pb"]
    assert loaded.uns["study"] == "test"
    np.testing.assert_array_equal(np.asarray(loaded.C.obs_names), np.asarray(dd.C.obs_names))
    validate(loaded)


def test_roundtrip_anndata_G(tmp_path, dd_genotypes):
    dd = _decorate(dd_genotypes)
    path = tmp_path / "cohort.dd.h5"
    dd.write_h5_dd(path)
    loaded = read_h5_dd(path)
    assert loaded.D_type == "anndata"
    np.testing.assert_array_equal(loaded.D.X, dd.D.X)
    assert set(loaded.obsm) == set(dd.obsm)


def test_zarr_reads_dense_X_lazily(tmp_path, cells, genotypes):
    cells.X = cells.X.toarray()
    dd = DonorData(D=genotypes, C=cells, donor_id="patient")
    path = tmp_path / "dense.dd.zarr"
    dd.write_zarr_dd(path, x_chunks={"D": (4, 2), "C": (16, 4)})
    loaded = read_zarr_dd(path)
    assert type(loaded.C.X).__module__.startswith("dask")
    np.testing.assert_allclose(loaded.C.X.compute(), dd.C.X)
    subset = loaded.filter_donors(["P1"])
    assert subset.C.X.shape[0] == dd.n_cells_per_donor["P1"]


def test_zarr_keeps_sparse_X_on_disk_and_aggregates_it_by_block(tmp_path, monkeypatch, dd_genotypes):
    from anndata.abc import CSRDataset

    from donordata._core import _utils

    path = tmp_path / "sparse.dd.zarr"
    dd_genotypes.write_zarr_dd(path)
    loaded = read_zarr_dd(path)
    assert isinstance(loaded.C.X, CSRDataset)
    monkeypatch.setattr(_utils, "BLOCK_NNZ", 25)
    monkeypatch.setattr(type(loaded.C.X), "to_memory", lambda self: pytest.fail("the whole matrix was read"))
    selections = [
        lambda d: d,
        lambda d: d.filter_donors(["P1", "P4", "P6"]),
        lambda d: d.sel(C_var=["gene3", "gene0", "gene7"]),
    ]
    for select in selections:
        for func in ("sum", "mean", "var", "count_nonzero"):
            for cell_filter in ({}, {"filter_key": "cell_type", "filter_value": "B"}):
                expected, result = select(dd_genotypes), select(loaded)
                expected.aggregate(key_added="agg", func=func, **cell_filter)
                result.aggregate(key_added="agg", func=func, **cell_filter)
                assert result.obsm["agg"].columns.equals(expected.obsm["agg"].columns)
                np.testing.assert_allclose(result.obsm["agg"].to_numpy(), expected.obsm["agg"].to_numpy())


def test_zarr_reads_selected_D_var_columns(tmp_path, dd_genotypes):
    path = tmp_path / "genotypes.dd.zarr"
    dd_genotypes.write_zarr_dd(path)
    loaded = read_zarr_dd(path, var_columns=["chrom", "pos"])
    assert list(loaded.D.var.columns) == ["chrom", "pos"]
    assert loaded.D.var_names.equals(dd_genotypes.D.var_names)
    np.testing.assert_array_equal(loaded.D.var["pos"], dd_genotypes.D.var["pos"])


def test_roundtrip_mudata_C_h5(tmp_path, cells, patients):
    protein = cells[:, :3].copy()
    protein.var_names = ["CD3", "CD19", "CD56"]
    mdata = md.MuData({"rna": cells.copy(), "prot": protein})
    mdata.obs["patient"] = cells.obs["patient"].to_numpy()
    dd = DonorData(D=patients, C=mdata, donor_id="patient")
    path = tmp_path / "multi.dd.h5"
    dd.write_h5_dd(path)
    loaded = read_h5_dd(path)
    assert isinstance(loaded.C, md.MuData)
    assert loaded.n_cells == dd.n_cells
    assert list(loaded.C.obs.columns) == list(dd.C.obs.columns)
    assert list(loaded.C.mod) == ["rna", "prot"]


def test_roundtrip_mudata_zarr_without_pulling_columns(tmp_path, cells, patients):
    protein = cells[:, :3].copy()
    protein.var_names = ["CD3", "CD19", "CD56"]
    mdata = md.MuData({"rna": cells.copy(), "prot": protein})
    mdata.obs = mdata.obs[[]].assign(patient=cells.obs["patient"].to_numpy())
    dd = DonorData(D=patients, C=mdata, donor_id="patient")
    dd.write_zarr_dd(tmp_path / "multi.dd.zarr")
    loaded = read_zarr_dd(tmp_path / "multi.dd.zarr")
    assert isinstance(loaded.C, md.MuData)
    assert list(loaded.C.obs.columns) == ["patient"]
    np.testing.assert_allclose(loaded.C.mod["prot"].X.to_memory().toarray(), dd.C.mod["prot"].X.toarray())
    assert list(dd.C.obs.columns) == ["patient"]


def test_reads_files_with_a_G_group(tmp_path, dd):
    path = tmp_path / "old.dd.h5"
    dd.write_h5_dd(path)
    with h5py.File(path, "r+") as f:
        f.move("D", "G")
        f.attrs["encoding-version"] = "0.2.0"
    with pytest.warns(UserWarning, match="stores the donor level in G"):
        loaded = read_h5_dd(path)
    assert loaded.n_donors == dd.n_donors
    assert loaded.D.shape == dd.D.shape


def test_encoding_checks(tmp_path, dd):
    path = tmp_path / "cohort.dd.h5"
    dd.write_h5_dd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-version"] = "0.1.0"
    read_h5_dd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-version"] = "9.0.0"
    with pytest.warns(UserWarning, match="encoding-version"):
        read_h5_dd(path)
    with h5py.File(path, "r+") as f:
        f.attrs["encoding-type"] = "anndata"
    with pytest.raises(ValueError, match="encoding-type"):
        read_h5_dd(path)


def test_format_detection(tmp_path, dd):
    with pytest.raises(ValueError, match="extension"):
        dd.write_dd(str(tmp_path / "cohort.txt"))
    dd.write_dd(str(tmp_path / "cohort"), fmt="h5")
    assert (tmp_path / "cohort.dd.h5").exists()
    with pytest.raises(ValueError, match="extension"):
        read_dd(tmp_path / "cohort")


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_mudata_with_shared_variable_names_and_named_indices(tmp_path, extension):
    from anndata import AnnData

    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(20)]
    rna = AnnData(
        rng.poisson(2, (20, 6)).astype(np.float32),
        obs=pd.DataFrame({"n_counts": rng.random(20)}, index=pd.Index(cells, name="index")),
        var=pd.DataFrame(index=pd.Index(["CD86", "g1", "g2", "g3", "g4", "g5"], name="index")),
    )
    adt = AnnData(
        rng.random((20, 3)).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(cells, name="index")),
        var=pd.DataFrame(index=pd.Index(["CD86", "PDL1", "p3"], name="index")),
    )
    mdata = md.MuData({"rna": rna, "adt": adt})
    mdata.obs = pd.DataFrame(
        {"guide": np.repeat(["g1", "g2", "g3", "g4"], 5), "gene": np.repeat(["A", "A", "B", "NT"], 5)},
        index=pd.Index(cells),
    )

    dd = DonorData.from_anndata(mdata, donor_id="gene", sample_id="guide")
    assert (dd.n_donors, dd.n_samples) == (3, 4)
    assert list(dd.C.obs.columns) == ["guide", "gene"]
    assert list(dd.copy().C.obs.columns) == ["guide", "gene"]
    path = tmp_path / f"screen{extension}"
    dd.write_dd(str(path))
    loaded = read_dd(path)
    assert list(loaded.C.obs.columns) == ["guide", "gene"]
    np.testing.assert_allclose(loaded.C.mod["rna"].X, dd.C.mod["rna"].X)
    np.testing.assert_array_equal(np.asarray(loaded.C.obsmap["adt"]), np.asarray(dd.C.obsmap["adt"]))
    assert list(loaded.C.obs_names) == list(dd.C.obs_names)
    validate(loaded)
    subset = loaded.filter_donors(["A"]).copy()
    assert subset.C.mod["adt"].n_obs == 10


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_wide_frames_in_obsm_are_written_as_one_array(tmp_path, dd3, extension):
    dd3.aggregate(key_added="profiles", level="sample")
    dd3.composition("cell_type", level="sample", normalize=False)
    dd3.obsm["mixed"] = pd.DataFrame({"a": np.arange(dd3.n_donors), "b": ["x"] * dd3.n_donors}, index=dd3.obs_names)
    path = tmp_path / f"cohort{extension}"
    dd3.write_dd(str(path))
    if extension == ".dd.h5":
        with h5py.File(path) as f:
            assert isinstance(f["S/obsm/profiles"], h5py.Dataset)
            assert isinstance(f["D/obsm/mixed"], h5py.Group)
            assert set(f.keys()) == {"D", "S", "C"}
            plain = read_elem(f["S"])
            assert plain.obsm["profiles"].shape == dd3.levels["sample"].obsm["profiles"].shape
    loaded = read_dd(path)
    samples = loaded.levels["sample"]
    for key in ("profiles", "cell_type_counts"):
        pd.testing.assert_frame_equal(samples.obsm[key], dd3.levels["sample"].obsm[key], check_names=False)
    pd.testing.assert_frame_equal(loaded.obsm["mixed"], dd3.obsm["mixed"], check_names=False)


def _matrices_equal(actual, expected):
    actual = actual.compute() if hasattr(actual, "compute") else actual
    actual = actual.to_memory() if hasattr(actual, "to_memory") else actual
    actual = actual.toarray() if hasattr(actual, "toarray") else np.asarray(actual)
    expected = expected.toarray() if hasattr(expected, "toarray") else np.asarray(expected)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dense_cells", [False, True], ids=["sparse C", "dense C"])
def test_h5_reads_matrices_lazily_and_gives_the_same_values(tmp_path, cells, genotypes, patients, dense_cells):
    from anndata.abc import CSRDataset

    cells = cells.copy()
    cells.obs = cells.obs.drop(columns=patients.columns)
    if dense_cells:
        cells.X = cells.X.toarray()
    dd = DonorData(D=genotypes, C=cells, donor_id="patient")
    path = tmp_path / "cohort.dd.h5"
    dd.write_h5_dd(path)
    eager, lazy = read_h5_dd(path), read_h5_dd(path, lazy=True)
    assert type(lazy.D.X).__module__.startswith("dask")
    assert type(lazy.C.X).__module__.startswith("dask") if dense_cells else isinstance(lazy.C.X, CSRDataset), type(
        lazy.C.X
    )
    _matrices_equal(lazy.D.X, eager.D.X)
    _matrices_equal(lazy.C.X, eager.C.X)
    subset = lazy.filter_donors(["P1", "P3"])
    _matrices_equal(subset.C.X, eager.filter_donors(["P1", "P3"]).C.X)
    lazy.aggregate(key_added="mean")
    eager.aggregate(key_added="mean")
    np.testing.assert_allclose(np.asarray(lazy.obsm["mean"]), np.asarray(eager.obsm["mean"]))
    validate(lazy)


def test_h5_reads_selected_D_var_columns(tmp_path, dd_genotypes):
    path = tmp_path / "cohort.dd.h5"
    dd_genotypes.write_h5_dd(path)
    for lazy in (False, True):
        assert list(read_h5_dd(path, lazy=lazy, var_columns=["chrom", "pos"]).D.var.columns) == ["chrom", "pos"]


def test_zarr_reads_into_memory_without_lazy(tmp_path, dd_genotypes):
    import scipy.sparse as sp

    path = tmp_path / "cohort.dd.zarr"
    dd_genotypes.write_zarr_dd(path)
    loaded = read_zarr_dd(path, lazy=False)
    assert isinstance(loaded.D.X, np.ndarray)
    assert sp.issparse(loaded.C.X)
    _matrices_equal(loaded.C.X, dd_genotypes.C.X)
    assert isinstance(read_dd(path, lazy=False).D.X, np.ndarray)
    assert type(read_dd(path).D.X).__module__.startswith("dask")


@pytest.mark.parametrize("extension", [".dd.zarr", ".dd.h5"])
def test_copy_of_a_lazily_read_object_is_independent(tmp_path, dd_genotypes, extension):
    path = tmp_path / f"cohort{extension}"
    dd_genotypes.write_dd(path)
    lazy = read_dd(path, lazy=True)
    copied = lazy.copy()
    _matrices_equal(copied.C.X, dd_genotypes.C.X)
    copied.C.obs["added"] = 1
    assert "added" not in lazy.C.obs.columns
    validate(copied)


def test_copy_of_a_lazily_read_mudata_level(tmp_path, cells, patients):
    import scipy.sparse as sp

    cells = cells.copy()
    cells.obs = cells.obs.drop(columns=patients.columns.drop("sex", errors="ignore"), errors="ignore")
    other = cells.copy()
    other.var_names = [f"p{i}" for i in range(other.n_vars)]
    mdata = md.MuData({"rna": cells, "prot": other})
    mdata.obs["patient"] = cells.obs["patient"].to_numpy()
    dd = DonorData(D=patients, C=mdata, donor_id="patient")
    path = tmp_path / "cohort.dd.zarr"
    dd.write_zarr_dd(path)
    copied = read_zarr_dd(path).copy()
    assert sp.issparse(copied.C.mod["rna"].X)
    validate(copied)


def _small_dd(X=None, layers=None):
    from anndata import AnnData

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2) if X is None else X
    obs = pd.DataFrame({"donor": ["A"] * 3 + ["B"] * 3}, index=[f"c{i}" for i in range(6)])
    cells = AnnData(X=X, obs=obs, layers=layers or {})
    donors = pd.DataFrame({"age": [30, 40]}, index=pd.Index(["A", "B"], name="donor"))
    return DonorData(D=donors, C=cells, donor_id="donor")


@pytest.mark.parametrize("sparse_X", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("extension", [".dd.zarr", ".dd.h5"])
def test_writing_into_the_file_a_lazy_object_reads_from_raises(tmp_path, sparse_X, extension):
    import scipy.sparse as sp

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    dd = _small_dd(sp.csr_matrix(X) if sparse_X else X)
    path = tmp_path / f"cohort{extension}"
    dd.write_dd(path)
    lazy = read_dd(path, lazy=True)
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_dd(path)
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.filter_donors(["A"]).write_dd(path)
    _matrices_equal(read_dd(path, lazy=False).C.X, X)
    del lazy
    gc.collect()
    read_dd(path, lazy=False).write_dd(path)
    _matrices_equal(read_dd(path, lazy=False).C.X, X)


def test_writing_into_the_store_a_zarr_backed_matrix_reads_from_raises(tmp_path):
    import zarr

    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    path = tmp_path / "cohort.dd.zarr"
    _small_dd(X).write_dd(path)
    lazy = read_dd(path, lazy=True)
    # readers such as cellink back X with the zarr.Array itself instead of a dask array
    lazy.C.X = zarr.open_group(str(path), mode="r")["C"]["X"]
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_dd(path)
    lazy.C.X = zarr.open_array(str(path / "C" / "X"), mode="r")
    with pytest.raises(ValueError, match="reads its matrices from"):
        lazy.write_dd(path)
    _matrices_equal(read_dd(path, lazy=False).C.X, X)


@pytest.mark.parametrize("extension", [".dd.h5", ".dd.zarr"])
def test_nullable_columns_of_donor_matrices_keep_their_type(tmp_path, extension):
    dd = _small_dd()
    dd.obsm["n"] = pd.DataFrame({"n": pd.array([1, None], dtype="Int64")}, index=dd.obs_names)
    dd.obsm["flag"] = pd.DataFrame({"flag": pd.array([True, None], dtype="boolean")}, index=dd.obs_names)
    dd.write_dd(tmp_path / f"nullable{extension}")
    loaded = read_dd(tmp_path / f"nullable{extension}", lazy=False)
    assert str(loaded.obsm["n"]["n"].dtype) == "Int64"
    assert str(loaded.obsm["flag"]["flag"].dtype) == "boolean"


def test_zarr_keeps_raw_and_writes_float16(tmp_path):
    dd = _small_dd(np.ones((6, 2), dtype=np.float16))
    dd.C.raw = dd.C.copy()
    dd.write_zarr_dd(tmp_path / "cohort.dd.zarr")
    loaded = read_zarr_dd(tmp_path / "cohort.dd.zarr", lazy=False)
    assert loaded.C.raw is not None
    assert loaded.C.X.dtype == np.float16


def test_a_selection_of_a_lazy_object_with_dense_layers_can_be_written(tmp_path):
    X = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
    _small_dd(X, layers={"counts": X * 2}).write_zarr_dd(tmp_path / "lazy.dd.zarr")
    subset = read_zarr_dd(tmp_path / "lazy.dd.zarr").filter_donors(["A"])
    subset.write_zarr_dd(tmp_path / "subset.dd.zarr")
    loaded = read_zarr_dd(tmp_path / "subset.dd.zarr", lazy=False)
    np.testing.assert_array_equal(loaded.C.layers["counts"], X[:3] * 2)


def test_written_mudata_maps_cells_to_modalities_by_name(tmp_path):
    from anndata import AnnData

    obs = pd.DataFrame({"donor": ["A", "A", "B", "B"]}, index=["c0", "c1", "c2", "c3"])
    mdata = md.MuData({"rna": AnnData(X=np.arange(4.0).reshape(4, 1), obs=obs.copy())})
    mdata.obs["donor"] = obs["donor"]
    dd = DonorData(D=pd.DataFrame(index=pd.Index(["B", "A"], name="donor")), C=mdata, donor_id="donor")
    dd.write_h5_dd(tmp_path / "mdata.dd.h5")
    with h5py.File(tmp_path / "mdata.dd.h5", "r") as f:
        names = read_elem(f["C/obs"]).index
        mod_names = read_elem(f["C/mod/rna/obs"]).index
        obsmap = read_elem(f["C/obsmap/rna"])
    assert list(mod_names[obsmap - 1]) == list(names)


def test_read_dd_accepts_a_store_path_with_a_trailing_slash(tmp_path):
    _small_dd().write_zarr_dd(tmp_path / "cohort.dd.zarr")
    assert read_dd(f"{tmp_path / 'cohort.dd.zarr'}/").n_donors == 2
