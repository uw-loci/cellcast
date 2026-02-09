# cellcast: A recast of cell segmentation models

<div align="center">

[![crates.io](https://img.shields.io/crates/v/cellcast)](https://crates.io/crates/cellcast)
[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This repository contains `cellcast`, a recast of cell segmentation models built
on the Burn framework. The goal of this project is to modernize (*i.e.* recast)
established machine learning models in a modern deep learning framework with a
`WebGPU` backend, enabling easy to install and GPU enabled cell segmentation networks.

## Usage

### Using cellcast with Rust

To use cellcast in your Rust project add it to your crate's dependencies and import the desired models.

```toml
[dependencies]
cellcast = "0.1.0"
```

The example below demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with Rust.
This example assumes you have the appropriate dependencies and helper functions to load your data as an
`Array2<T>` type:

```rust
use ndarray::Array2;
use cellcast::models::stardist_2d_versatile_fluo;

fn main() {
  let data_2d = load_image("/path/to/data_2d.tif");
  let labels = stardist_2d_versatile_fluo.predict(&data, Some(1.0), Some(99.8), None, None, True);
}

fn load_image(path: &str) -> Array2<u16> {
  // your logic to read/load from a file here
}
```

### Using cellcast with Python

You can use cellcast with Python by using the `cellcast_python` crate. Pre-compiled releases are available on PyPI as the `cellcast` package
and can be easily installed with `pip`:

```bash
$ pip install cellcast
```

The `cellcast` Python package currently supports the following architectures:

| Operating System | Architecture         |
| :---             | :---                 |
| Linux            | x86-64               |
| macOS            | intel, arm64         |
| Windows          | x86-64               |

Cellcast is compatible with Python `>=3.7`.

The example below demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with Python.
Note that this example assumes you have access to 2D data and `tifffile` installed in your Python
environment with cellcast:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data_2d = imread("path/to/data_2d.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data, gpu=True)
```

## Building from source

You can build the entire cellcast project from the root of this repository with:

```bash
$ cargo build
```

This will compile a *non-optimized* cellcast binaries. Pass the `--release` flag to
compile optimized binaries (note that compilation time may take upwards of 10 minutes).

### Build `cellcast_python` from source

To build and install cellcast for Python from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
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

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](cellcast/MODEL-LICENSES) file.
