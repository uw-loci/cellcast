# cellcast: A recast of cell segmentation models

⚠️ Warning: This is an experimental project!

This repository contains `cellcast`, a recast of cell segmentation models built
on the Burn framework. The goal of this project is to modernize (_i.e._ recast)
established machine learning models in a modern deep learning framework with a
`WebGPU` backend, enabling easy to install and GPU enabled cell segmentation networks.

## Install `cellcast` from souce

To install ast-net from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
Next create an environment (we recommend using `uv`) with the `maturin` development tool. This can be easily done with the
`uv` tool and this repository's `pyproject.toml`.

```bash
$ cd cellcast
$ uv sync
```

This will create the environment for you with maturin. Next install Rust library with:

```bash
$ maturin develop --release
```

**Note: This project also depends on development version of [imgal](https://github.com/imgal-sc/imgal) built from source.**

Until I release imgal version `0.2.0` you will need to link to the development version of the library. First clone imgal to
a directory of your choice. Then change the path to the imgal dependency in the `Cargo.toml` in the cellcast directory (_i.e._ `cellcast/cellcast/Cargo.toml`).

## Example

You can run the following stardist example:

```python
import imagej
import cellcast as cc
import numpy as np

# initialize imagej
ij = imagej.init(mode = "interactive")

# load the data and convert it to float32
data = ij.io().open("/path/to/stardist_test_data.tif")
narr = ij.py.to_xarray(data).data

# run stardist 2D inference with wgpu backend
result = cc.stardist.stardist_2d(narr)
ij.py.show(result)
