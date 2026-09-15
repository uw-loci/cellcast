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

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](cellcast/MODEL-LICENSES) file.
