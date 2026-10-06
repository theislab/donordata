# Install DonorData

```{contents}
:local:
:depth: 3
```

## Development version

DonorData is not on PyPI yet. Install it from [the GitHub repository](https://github.com/theislab/donordata):

```console
pip install git+https://github.com/theislab/donordata
```

## Optional dependencies

The PyTorch dataset in {mod}`donordata.ml` needs the `ml` extra:

```console
pip install "donordata[ml] @ git+https://github.com/theislab/donordata"
```

Donor or sample levels stored as an EHRData need [ehrdata](https://ehrdata.readthedocs.io), and cells given as a
SpatialData need [spatialdata](https://spatialdata.scverse.org). Both are installed with the `ehr` and `spatial`
extras:

```console
pip install "donordata[ehr,spatial] @ git+https://github.com/theislab/donordata"
```

## Troubleshooting

If there are issues that have not been described, addressed, or documented, please consider
[opening an issue](https://github.com/theislab/donordata/issues).
