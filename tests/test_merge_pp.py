import numpy as np
import pytest

import donordata
from donordata import concat, validate


def test_concat_tables_with_label(dd):
    dd.obsm["emb"] = np.arange(dd.n_donors * 2.0).reshape(-1, 2)
    first, second = dd.filter_donors(["P0", "P1"]), dd.filter_donors(["P5"])
    merged = concat({"cohort_a": first, "cohort_b": second}, label="cohort")
    assert merged.obs_names.tolist() == ["P0", "P1", "P5"]
    assert merged.obs["cohort"].tolist() == ["cohort_a", "cohort_a", "cohort_b"]
    assert merged.n_cells == first.n_cells + second.n_cells
    np.testing.assert_array_equal(merged.obsm["emb"], dd.obsm["emb"][[0, 1, 5]])
    validate(merged)


def test_concat_anndata_G(dd_genotypes):
    merged = concat([dd_genotypes.filter_donors(["P2"]), dd_genotypes.filter_donors(["P3", "P4"])])
    assert merged.D_type == "anndata"
    assert merged.shape[:2] == (3, dd_genotypes.shape[1])


def test_concat_rejects_duplicates_and_mixed_kinds(dd, dd_genotypes):
    with pytest.raises(ValueError, match="not unique"):
        concat([dd, dd])
    with pytest.raises(ValueError, match="kinds"):
        concat([dd.filter_donors(["P0"]), dd_genotypes.filter_donors(["P1"])])


def test_cell_level_obs_filter(dd):
    out = donordata.pp.cell_level_obs_filter(dd, "cell_type", ["B", "NK"], inplace=False)
    assert set(out.C.obs["cell_type"]) <= {"B", "NK"}
    assert not out.C.is_view
    n_before = dd.n_cells
    assert donordata.pp.cell_level_obs_filter(dd, "cell_type", "T") is None
    assert set(dd.C.obs["cell_type"]) == {"T"}
    assert dd.n_cells < n_before


def test_donor_level_filters(dd, dd_genotypes):
    out = donordata.pp.donor_level_obs_filter(dd, "disease", "covid", inplace=False, copy=False)
    assert set(out.obs["disease"]) == {"covid"}
    donordata.pp.donor_level_obs_filter(dd, "site", np.array(["B"]))
    assert set(dd.obs["site"]) == {"B"}
    donordata.pp.donor_level_var_filter(dd_genotypes, donor_level_key="pos", donor_level_values=[1, 2])
    assert dd_genotypes.shape[1] == 2
    with pytest.raises(ValueError, match="no variables"):
        donordata.pp.donor_level_var_filter(dd, donor_level_key="pos", donor_level_values=1)


def _cohort(prefix, genes):
    import pandas as pd
    from anndata import AnnData

    from donordata import DonorData

    cells = AnnData(
        X=np.ones((2, len(genes))),
        obs=pd.DataFrame({"donor": [prefix + "1"] * 2}, index=[prefix + "c0", prefix + "c1"]),
        var=pd.DataFrame(index=genes),
    )
    dd = DonorData(D=pd.DataFrame(index=pd.Index([prefix + "1"], name="donor")), C=cells, donor_id="donor")
    dd.aggregate(func="mean", key_added="pb")
    return dd


def test_concat_outer_keeps_every_column_of_donor_matrices():
    merged = concat([_cohort("a", ["g0", "g1", "g2"]), _cohort("b", ["g0", "g1"])], join="outer")
    assert list(merged.obsm["pb"].columns) == ["g0", "g1", "g2"]
    assert np.isnan(merged.obsm["pb"].loc["b1", "g2"])


def test_concat_rejects_ehrdata_levels():
    ed = pytest.importorskip("ehrdata", exc_type=ImportError)
    import pandas as pd
    from anndata import AnnData

    from donordata import DonorData

    def cohort(prefix):
        donors = ed.EHRData(
            X=np.ones((1, 1, 2)),
            obs=pd.DataFrame(index=pd.Index([prefix + "1"], name="donor")),
            tem=pd.DataFrame(index=["t0", "t1"]),
        )
        cells = AnnData(np.ones((1, 1)), obs=pd.DataFrame({"donor": [prefix + "1"]}, index=[prefix + "c"]))
        return DonorData(D=donors, C=cells, donor_id="donor")

    with pytest.raises(ValueError, match="EHRData"):
        concat([cohort("a"), cohort("b")])
