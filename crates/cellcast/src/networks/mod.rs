//! Network backbones.
//!
//! This module contains network backbones for each supported model. These
//! networks have been converted into Rust from ONNX using the `burn-onnx`
//! crate, followed by manual modification to fit cellcast's needs.

pub mod stardist;
