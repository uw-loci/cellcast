//! The `cellcast` crate is a recast of cell segmentation models built on the Burn
//! tensor and deep learning framework. This library aims to make access to cell
//! segmentation models easier and hardware agnostic.
//!
//! ## Crate Status
//!
//! This crate is still under active development and it's API is not stable.

mod config;
mod error;
mod geometry;
mod labeling;
pub mod models;
mod networks;
mod process;
mod utils;
pub use error::CellcastError;
