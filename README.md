# cellcast: A recast of cell segmentation models

<div align="center">

[![crates.io](https://img.shields.io/crates/v/cellcast)](https://crates.io/crates/cellcast)
[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>
This repository contains `cellcast`, a recast of cell segmentation models built
on the Burn framework. The goal of this project is to modernize (_i.e._ recast)
established machine learning models in a modern deep learning framework with a
`WebGPU` backend, enabling easy to install and GPU enabled cell segmentation networks.

## Build `cellcast` from source

To build and install cellcast from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
Next create a Python environment (we recommend using `uv`) with the `maturin` development tool in the "cellcast_python" directory:

```bash
$ cd cellcast_python
$ uv venv
$ uv pip install numpy maturin
```

This will create the environment for you with maturin. Next activate your environment and install the cellcast library with:

```bash
$ source ./venv/bin/activate
$ (cellcast_python) maturin develop
```

This will compile cellcast as a *non-optimized* binary with debug symbols. This decreases compile time by skipping compiler optimizations
and retaining debug symbols. To build *optimized* binaries of cellcast you must pass the `--release` flag. Note that this significantly increases
compilation times to ~6-7 minutes.

```bash
$ (cellcast_python) maturin develop --release
```

You can also run `uv sync` in the "cellcast_python" directory to create a Python environment and compile cellcast. Note that this installation
path uses the `--release` flag to compile cellcast, expect longer compile and installation times.

## Example

You can run the following stardist example:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data = imread("path/to/data.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data, gpu=True)
```

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](cellcast/MODEL-LICENSES) file.
