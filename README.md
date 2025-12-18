# cellcast: A recast of cell segmentation models

⚠️ Warning: This is an experimental project!

This repository contains `cellcast`, a recast of cell segmentation models built
on the Burn framework. The goal of this project is to modernize (_i.e._ recast)
established machine learning models in a modern deep learning framework with a
`WebGPU` backend, enabling easy to install and GPU enabled cell segmentation networks.

## Build `cellcast` from souce

To install ast-net from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
Next create an environment (we recommend using `uv`) with the `maturin` development tool. This can be easily done with the
`uv` tool and this repository's `pyproject.toml`.

```bash
$ cd cellcast_python
$ uv venv
$ uv pip install numpy maturin
```

This will create the environment for you with maturin. Next activate your environment and install Rust library with:

```bash
$ maturin develop --release
```

**Note: This project also depends on development version of [imgal](https://github.com/imgal-sc/imgal) built from source.**

Until I release imgal version `0.2.0` you will need to link to the development version of the library. First clone imgal to
a directory of your choice. Then change the path to the imgal dependency in the `Cargo.toml` in the cellcast directory (_i.e._ `cellcast/cellcast/Cargo.toml`).

## Example

You can run the following stardist example:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data = imread("path/to/data.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data)
```
