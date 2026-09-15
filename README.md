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
cellcast = "0.3.0"
```

The following examples demonstrate how to use cellcast's StarDist2D model in Rust with fetched *versatile fluo* pretrained
and custom weights. Each supported cell segmentation model in cellcast is configured and initialized via it's model struct
in the imported from the `models` module. If no `weights_path` is provided then the model's published pretrained weights
are downloaded and cached (note that the the cache weights are ideally used if present instead of downloading):

```rust
use cellcast::CellcastError;
use cellcast::models::StarDist2D;
use ndarray::Array2;

fn main() -> Result<(), CellcastError>{
  let data = get_image("path/to/data.tif");
  // initialize a StarDist2D fluo model with fetched weights on the GPU
  let sd = StarDist2D::init_fluo(None, true)?;
  // run the model on the input data with default settings
  let labels = sd.predict_fluo(&data, None, None, None, None);
}

fn get_image(papth: &str) -> Array2<u16> {
  // your logic to get image data as an array. 
}
```

To initialize a model with custom weights, provide the path to the weights in burnpack format (`.bpk`) when creating a model
instance.

```rust
let sd = StarDist2D::init_fluo("path/to/custom_weights.bpk", true)?;
```

See the [burn-store](https://github.com/tracel-ai/burn/tree/main/crates/burn-store) and the
[burn-onnx](https://github.com/tracel-ai/burn-onnx) crates for more details.

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

Cellcast is compatible with Python `>=3.8` and requires *only* `NumPy`.

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

## Building from source

You can build the entire cellcast project from the root of this repository with:

```bash
$ cargo build
```

This will compile cellcast *without optimizations*. Pass the `--release` flag to compile an *optimized* release version (note that compilation time may take upwards
of 5 to 10 minutes, depending on your hardware). Compiling cellcast on your own allows you to change the backend from `Wgpu` to another that may better align with
your hardware, such as the `cuda` backend for nVidia GPUs. To change the CPU and/or GPU backends, edit the `backend.rs` file. For example, the configuration below
will compile cellcast with the `cuda` backend:

```rust
use burn::backend::{Flex, Cuda};

pub(crate) type CpuBackend<E, I> = Flex<E, I>;
pub(crate) type GpuBackend<E, I> = Cuda<E, I>;
```

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
