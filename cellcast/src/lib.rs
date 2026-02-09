//! The `cellcast` crate is a recast of cell segmentation models built on the Burn
//! tensor and deep learning framework. This library aims to make access to cell
//! segmentation models easier and hardware agnostic.
//!
//! ## Crate Status
//!
//! This crate is still under active development and it's API is not stable.
pub mod geometry;
pub mod labeling;
pub mod models;
pub mod networks;
pub mod process;
pub mod utils;
