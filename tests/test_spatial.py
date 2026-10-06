import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from donordata import DonorData, read_dd, validate

sd = pytest.importorskip("spatialdata")
from spatialdata.models import ShapesModel, TableModel


@pytest.fixture
def sdata():
    rng = np.random.default_rng(0)
    n = 30
    shapes = ShapesModel.parse(rng.random((n, 2)) * 100, geometry=0, radius=np.ones(n))
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["cells"] * n),
            "instance_id": np.arange(n),
            "sample": np.repeat(["P1_a", "P1_b", "P2_a", "P3_a", "P3_b", "P3_c"], 5),
            "patient": np.repeat(["P1", "P1", "P2", "P3", "P3", "P3"], 5),
            "disease": np.repeat(["covid", "covid", "healthy", "covid", "covid", "covid"], 5),
            "cell_type": rng.choice(["B", "T"], n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    adata = AnnData(rng.poisson(1.0, (n, 4)).astype(np.float32), obs=obs)
    table = TableModel.parse(adata, region="cells", region_key="region", instance_key="instance_id")
    return sd.SpatialData(shapes={"cells": shapes}, tables={"table": table})


def test_spatialdata_cells(sdata):
    dd = DonorData.from_anndata(sdata, donor_id="patient", sample_id="sample")
    assert isinstance(dd.C, AnnData)
    assert list(dd.D.columns) == ["disease"]
    assert dd.n_samples == 6 and dd.n_cells == 30
    assert "SpatialData" in repr(dd)
    full = dd.spatialdata
    assert list(full.shapes) == ["cells"] and full.tables["table"].n_obs == 30
    covid = dd.filter_donors("disease == 'covid'")
    assert covid.spatialdata.tables["table"].n_obs == covid.n_cells == 25
    assert list(covid.spatialdata.shapes) == ["cells"]
    assert covid.copy().spatialdata is not None
    assert dd.flatten().spatialdata is not None
    validate(covid)


def test_spatialdata_given_as_C(sdata):
    patients = pd.DataFrame({"age": [50, 60, 70]}, index=pd.Index(["P1", "P2", "P3"], name="patient"))
    dd = DonorData(D=patients, C=sdata, donor_id="patient")
    assert dd.n_donors == 3 and dd.C.n_obs == 30
    dd.composition("cell_type")
    assert dd.get_df("age", level="cell").shape == (30, 1)


def test_spatialdata_roundtrip(tmp_path, sdata):
    dd = DonorData.from_anndata(sdata, donor_id="patient", sample_id="sample").filter_cells("cell_type == 'B'")
    path = tmp_path / "spatial.dd.zarr"
    dd.write_dd(str(path))
    loaded = read_dd(path)
    assert loaded.n_cells == dd.n_cells
    assert loaded.spatialdata is not None and list(loaded.spatialdata.shapes) == ["cells"]
    pd.testing.assert_index_equal(loaded.C.obs_names, dd.C.obs_names)
    with pytest.raises(ValueError, match="Zarr"):
        dd.write_dd(str(tmp_path / "spatial.dd.h5"))


def test_concat_rejects_spatialdata(sdata):
    import donordata

    dd = DonorData.from_anndata(sdata, donor_id="patient")
    with pytest.raises(NotImplementedError, match="spatialdata.concatenate"):
        donordata.concat([dd, dd])
