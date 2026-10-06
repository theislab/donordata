# API

```{eval-rst}
.. module:: donordata
```

The central class:

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DonorData
```

(combining-api)=

## Combining

Combining {class}`DonorData` objects.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   concat
```

(reading-api)=

## Reading

Reading DonorData's native formats `.dd.h5` and `.dd.zarr`.
(These functions are also exported as {func}`read_dd`, {func}`read_h5_dd` and {func}`read_zarr_dd`.)

```{eval-rst}
.. autosummary::
   :toctree: generated/

   io.read_dd
   io.read_h5_dd
   io.read_zarr_dd
```

(writing-api)=

## Writing

Writing a complete {class}`DonorData` object to disk.

```{eval-rst}
.. autosummary::

   DonorData.write_dd
   DonorData.write_h5_dd
   DonorData.write_zarr_dd
```

## Validation

Checking an object against the {doc}`specification <io/spec>`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   validate
   DonorDataSchemaError
```

## Extensions

```{eval-rst}
.. autosummary::
   :toctree: generated/

   register_donordata_namespace
```

## Preprocessing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pp.cell_level_obs_filter
   pp.donor_level_obs_filter
   pp.donor_level_var_filter
```

## Machine learning

Requires PyTorch, see {doc}`install`.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   ml.MILDataset
   ml.mil_collate_fn
```
