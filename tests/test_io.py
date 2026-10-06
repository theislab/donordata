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
    np.testing.assert_allclose(loaded.C.mod["prot"].X.toarray(), dd.C.mod["prot"].X.toarray())
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
