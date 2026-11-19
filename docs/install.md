# Install DonorData

```{contents}
:local:
:depth: 3
```

## PyPI

DonorData needs Python 3.12 or newer. Install it from [PyPI](https://pypi.org/project/donordata/):

```console
pip install donordata
```

## Development version

The latest version on [GitHub](https://github.com/theislab/donordata) is installed with:

```console
pip install git+https://github.com/theislab/donordata
```

## Optional dependencies

The PyTorch dataset in {mod}`donordata.ml` needs the `ml` extra:

```console
pip install "donordata[ml]"
```

Donor or sample levels stored as an EHRData need [ehrdata](https://ehrdata.readthedocs.io) 0.6 or later, which is
installed with the `ehr` extra:

```console
pip install "donordata[ehr]"
```

## Troubleshooting

If there are issues that have not been described, addressed, or documented, please consider
[opening an issue](https://github.com/theislab/donordata/issues).
