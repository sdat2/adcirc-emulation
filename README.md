# ADCIRC Emulation Project

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 [![Python package](https://github.com/sdat2/new-orleans/actions/workflows/pytest.yml/badge.svg)](https://github.com/sdat2/new-orleans/actions/workflows/pytest.yml)


## Requirements

### Necessary

- Python 3.8+
- `sithom>=0.0.3`
- `typeguard`
- `cdsapi`
- `xarray`
- `netCDF4`
- `dask`
- `uncertainties`
- `climada`
- `cartopy`
- `eccodes=1.3.3`
- `wandb`
- `comet-ml`

### Optional

- `noaa_coops`

## Getting started

Clone this repository locally.

Then pip install locally to the python environment you want to use.

```bash
pip install -e .
```

To get cartopy to work properly you might be better off with.

```bash
conda env update --file environment.yml --name base
```

## Project Organization

```txt
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`.
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
│   ├── exploratory    <- Notebooks for initial exploration.
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting.
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module.
│   │
│   ├── config         <- Config yaml files.
│   │
│   ├── data_loading   <- Scripts to download or generate data.
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling.
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions.
│   │
│   └── plot           <- Different plotting scripts for results.
│
└── setup.cfg          <- Setup configuration file for linting rules.
```

## Code formatting

To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black . ``` 
from within the project directory.


---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
