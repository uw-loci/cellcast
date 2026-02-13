# cellcast: A recast of cell segmentation models

<div align="center">

[![crates.io](https://img.shields.io/crates/v/cellcast)](https://crates.io/crates/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This crate contains the [cellcast](https://github.com/uw-loci/cellcast) core Rust library. Cellcast is a recast of cell segmentation models
built on the Burn tensor and deep learning framework. The goal of this project is to modernize (*i.e.* recast) established cell segmentation models
with a WebGPU backend. Cellcast aims to make access to cell segmentation models **easy** and **reproducible**.

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

## Building from source

You can build the cellcast core library with:

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
changes to the cellcast core library and recompile the project for python in the `cellcast_python` crate directory with `maturin`.

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](cellcast/MODEL-LICENSES) file.
