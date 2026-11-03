from dataclasses import dataclass


@dataclass
class Settings:
    """Global options of donordata, available as ``donordata.settings``.

    Attributes
    ----------
    drop_missing_strings
        Treat donor and sample identifiers that are strings standing for a missing value, such as ``"nan"``,
        ``"None"``, ``"NA"``, ``"null"`` or ``""`` (in any case), as missing, and drop their samples and cells.
        Set to ``False`` when such strings are real names.
    """

    drop_missing_strings: bool = True


settings = Settings()
