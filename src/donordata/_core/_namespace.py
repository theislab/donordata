from __future__ import annotations

import warnings
from collections.abc import Callable


class _CachedAccessor:
    def __init__(self, name: str, accessor: type):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        accessor = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor)
        return accessor


def register_donordata_namespace(name: str) -> Callable[[type], type]:
    """Register a class as an extension namespace on :class:`DonorData`.

    Packages that build on DonorData, for example for genetics, patient representations or
    perturbation analysis, can expose their functions as ``dd.<name>.<function>()``. The decorated
    class is instantiated with the DonorData on first access and cached on that object.

    Parameters
    ----------
    name
        Attribute name of the namespace.

    Returns
    -------
    Callable
        A class decorator.

    Examples
    --------
    >>> @donordata.register_donordata_namespace("qc")
    ... class QC:
    ...     def __init__(self, dd):
    ...         self._dd = dd
    ...
    ...     def small_donors(self, min_cells=100):
    ...         return self._dd.obs_names[self._dd.n_cells_per_donor < min_cells]
    >>> dd.qc.small_donors()
    """
    from donordata._core.donordata import DonorData

    def decorator(accessor: type) -> type:
        if hasattr(DonorData, name):
            warnings.warn(f"Registering namespace '{name}' overrides an existing attribute of DonorData.", stacklevel=2)
        setattr(DonorData, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator
