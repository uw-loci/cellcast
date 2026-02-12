# cellcast_python

<div align="center">

[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

Python bindings for the [cellcast](https://github.com/uw-loci/cellcast) core Rust library.

## Installation

### Requirements

The `cellcast` Python package currently supports the following architectures:

| Operating System | Architecture         |
| :---             | :---                 |
| Linux            | x86-64, arm64        |
| macOS            | intel, arm64         |
| Windows          | x86-64               |

Cellcast is compatible with Python `>=3.7` and requires *only* `NumPy`.

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

Once cellcast has been installed, `cellcast` will be available to import. The example below
demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with
Python. Note that this example assumes you have access to 2D data and `tifffile` installed
in your Python environment with cellcast:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data_2d = imread("path/to/data_2d.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data, gpu=True)
```

Run `help()` on the `stardist_2d_versatile_fluo.predict` function to see the full function signature and default values. 

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](../LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](../LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](../cellcast/MODEL-LICENSES) file.
