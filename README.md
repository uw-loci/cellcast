# cellcast: A recast of cell segmentation models

<div align="center">

[![crates.io](https://img.shields.io/crates/v/cellcast)](https://crates.io/crates/cellcast)
[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This repository contains `cellcast`, a recast of cell segmentation models built
on the Burn framework. The goal of this project is to modernize (*i.e.* recast)
established cell segmentation machine learning models in a modern deep learning framework with a
WebGPU backend. Because cellcast targets the WebGPU backend it can provide GPU agnostic cell
segmentation models.

## Usage

### Using cellcast with Rust

To use cellcast in your Rust project add it to your crate's dependencies and import the desired models.

```toml
[dependencies]
cellcast = "0.1.1"
```

The example below demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with Rust.
This example assumes you have the appropriate dependencies and helper functions to load your data as an `Array2<T>` type:

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

*Note: `T` here can be any numeric value (*i.e.* `u8`, `i32`, `f64`).*

### Using cellcast with Python

You can use cellcast in your Python project by using the `cellcast_python` crate. Pre-compiled releases are available on PyPI as the `cellcast` package
and can be easily installed with `pip`:

```bash
$ pip install cellcast
```

The `cellcast` Python package currently supports the following architectures:

| Operating System | Architecture         |
| :---             | :---                 |
| Linux            | x86-64, arm64        |
| macOS            | intel, arm64         |
| Windows          | x86-64               |

Cellcast is compatible with Python `>=3.7` and requires *only* `NumPy`.

The example below demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with Python.
Note that this example assumes you have access to 2D data and `tifffile` installed in your Python environment with cellcast:

```python
import cellcast.models as ccm
from tifffile import imread

# load 2D data for inference
data_2d = imread("path/to/data_2d.tif")

# run stardist inference and produce instance segmentations
labels = ccm.stardist_2d_versatile_fluo.predict(data, gpu=True)
```

Run `help()` on the `stardist_2d_versatile_fluo.predict` function to see the full function signature and default values. 

## Building from source

You can build the entire cellcast project from the root of this repository with:

```bash
$ cargo build
```

This will compile a cellcast *without optimizations*. Pass the `--release` flag to compile an *optimized* release version (note that compilation time may take upwards
of 10 minutes). Because cellcast is a library, compiling it on it's own isn't very useful. However being able to successfully compile cellcast on your own computer
means that you can change the backend from `Wgpu` to whatever other [supported Burn backend](https://github.com/Tracel-AI/burn?tab=readme-ov-file#supported-backends)
you want. Recompiling cellcast with a *different* backend may allow you to take advantage of hardware specific optimizations not available to the `Wgpu` backend.

Each model defines it's own backend parameters at the start of the file. For example the StarDist2D versatile fluo model defines the `Wgpu` and `NdArray` (for CPU inference)
backends like this:

```rust
type NdArrayBackend = NdArray<f32, i32>;
type WgpuBackend = Wgpu<f32, i32>;
```

Change the `Wgpu` backend to whatever one you want (*e.g* `Cuda`) and recompile your Rust project. If you are using `cellcast_python`, then make the necessary backend
changes to the cellcast core library and recompile the project for Python.

### Build `cellcast_python` from source

To build and install cellcast for Python from source first install the Rust toolchain from [rust-lang.org](https://rust-lang.org/tools/install/).
Next create a Python environment (we recommend using `uv`) with the `maturin` development tool in the **crates/cellcast_python** directory:

```bash
$ cd crates/cellcast_python
$ uv venv
$ uv pip install numpy maturin
```

This will create the environment for you with maturin. Next activate your environment and install the cellcast library with:

```bash
$ source ./venv/bin/activate
$ (cellcast_python) maturin develop
```

This will compile cellcast as a *non-optimized* binary with debug symbols. This decreases compile time by skipping compiler optimizations
and retaining debug symbols. To build *optimized* binaries of cellcast you must pass the `--release` flag. Note that this significantly increases compilation times upwards of 10 minutes.

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
