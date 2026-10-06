from __future__ import annotations

from anndata import AnnData


def is_spatialdata(obj) -> bool:
    """Whether ``obj`` is a SpatialData, without importing spatialdata."""
    return any(cls.__name__ == "SpatialData" and cls.__module__.startswith("spatialdata") for cls in type(obj).__mro__)


def table_of(sdata, table_name: str | None) -> tuple[str, AnnData]:
    """The name and the table of a SpatialData that holds the cells."""
    names = list(sdata.tables)
    if table_name is None:
        if len(names) == 1:
            table_name = names[0]
        elif "table" in names:
            table_name = "table"
        else:
            raise ValueError(f"The SpatialData has the tables {names}, pass table_name to choose the cells.")
    if table_name not in sdata.tables:
        raise KeyError(f"Table '{table_name}' not found in the SpatialData, which has the tables {names}.")
    return table_name, sdata.tables[table_name]


def linking_columns(adata) -> set[str]:
    """Columns that link the rows of a SpatialData table to its spatial elements."""
    attrs = adata.uns.get("spatialdata_attrs", {}) if isinstance(adata, AnnData) else {}
    return {attrs[key] for key in ("region_key", "instance_key") if attrs.get(key)}


def with_table(sdata, table_name: str, table: AnnData):
    """A SpatialData that shares the elements of ``sdata`` and holds ``table`` under ``table_name``."""
    from spatialdata import SpatialData

    tables = {name: t for name, t in sdata.tables.items() if name != table_name}
    tables[table_name] = table
    elements = {
        "images": dict(sdata.images),
        "labels": dict(sdata.labels),
        "points": dict(sdata.points),
        "shapes": dict(sdata.shapes),
        "tables": tables,
    }
    attrs = getattr(sdata, "attrs", None)
    if attrs:
        return SpatialData(**elements, attrs=dict(attrs))
    return SpatialData(**elements)


def describe(sdata) -> str:
    """Short summary of the spatial elements of a SpatialData."""
    parts = []
    for kind in ("images", "labels", "points", "shapes"):
        names = list(getattr(sdata, kind))
        if names:
            parts.append(f"{kind}: " + ", ".join(f"'{n}'" for n in names))
    return "; ".join(parts) if parts else "no spatial elements"
