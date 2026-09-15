# cellcast_python

<div align="center">

[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This crate contains the Python bindings (via PyO3) for the [cellcast](https://github.com/uw-loci/cellcast)
core Rust library. Cellcast is a recast of cell segmentation models built on the Burn tensor and deep
learning framework. The goal of this project is to modernize (*i.e.* recast) established cell segmentation models
with a WebGPU backend. Cellcast aims to make access to cell segmentation models **easy** and **reproducible**.

## Installation

### Requirements

The `cellcast` Python package currently supports the following architectures:

| Operating System | Architecture         |
| :---             | :---                 |
| Linux            | x86-64, arm64        |
| macOS            | intel, arm64         |
| Windows          | x86-64               |

Cellcast is compatible with Python `>=3.8` and requires *only* `NumPy`.

### cellcast from PyPI

You can install the cellcast Python package from PyPI with:

```bash
$ pip install cellcast
```

### Build cellcast_python from source

To build the cellcat_python package from source, use the `maturin` build tool
(this requires the Rust toolchain). If you're using `uv` to manage your Python
virtual environments (venv) add `numpy` and `maturin` to your environment and run the
`maturin develop` command in the `cellcast_python` directory of the
[cellcast](https://github.com/uw-loci/cellcast) repository with your venv activated:

```bash
$ source ~/path/to/myenv/.venv/bin/activate
$ (myenv) cd cellcast_python
$ maturin develop
```

Alernatively if you're using `conda` or `mamba` you can do the following:

```bash
$ cd cellcat_python
$ mamba activate myenv
(myenv) $ mamba install numpy maturin
...
(myenv) $ maturin develop
```

This will compile a *non-optimized* cellcast binaries. Pass the `--release` flag to
compile optimized binaries (note that compilation time may take upwards of 10 minutes).

## Usage

### Using cellcast

The following example demonstrates how to use cellcast's StarDist2D model in Python with fetched *versatile fluo* pretrained weights (note: here we
assume you have your data in a 2D NumPy array):

```python
import cellcast.models.StarDist2D as StarDist2D

# assuming "data" is a 2D NumPy array
sd = StarDist2D.init_fluo(gpu=True)
labels = sd.predict_fluo(data)
```

Run `help()` on the `predict_fluo()` function to see the full function signature and default values. To initialize a model with custom weights, provide
the path to the weights in burnpack format (`.bpk`) when creating a model instance.

```python
sd = StarDist2D.init_fluo("path/to/custom_weights.bpk", True)
```

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](../LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](../LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](../cellcast/MODEL-LICENSES) file.
