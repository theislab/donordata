from dataclasses import dataclass


@dataclass(frozen=True)
class DAnn:
    """Default names of donor-level fields."""

    donor: str = "donor_id"
    donor_family: str = "donor_family_id"
    donor_paternal_id: str = "donor_paternal_id"
    donor_maternal_id: str = "donor_maternal_id"
    donor_sex: str = "sex"
    DONOR_X_KEY: str = "donor_X"
    DONOR_LABELS_KEY: str = "donor_labels"
    DONOR_BATCH_KEY: str = "donor_batch"
    DONOR_CAT_COVS_KEY: str = "donor_extra_categorical_covs"
    DONOR_CONT_COVS_KEY: str = "donor_extra_continuous_covs"
    DONOR_INDICES_KEY: str = "donor_ind_x"
    DONOR_ID_KEY: str = "donor_id"


@dataclass(frozen=True)
class CAnn:
    """Default names of cell-level fields."""

    celltype: str = "celltype"
    CELL_X_KEY: str = "cell_X"
    CELL_LABELS_KEY: str = "cell_labels"
    CELL_BATCH_KEY: str = "cell_batch"
    CELL_CAT_COVS_KEY: str = "cell_extra_categorical_covs"
    CELL_CONT_COVS_KEY: str = "cell_extra_continuous_covs"
    CELL_INDICES_KEY: str = "cell_ind_x"
    CELL_DONOR_KEY: str = "donor_id"
