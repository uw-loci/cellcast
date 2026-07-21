//! Cell segmentation models.
//!
//! This module contains the supported cellcast cell segmentation models. Each
//! model is first initialized on the GPU or CPU with either fetched pre-trained
//! weights or custom weights.

mod stardist_2d;
mod stardist_3d;

pub use stardist_2d::StarDist2D;
pub use stardist_3d::StarDist3D;
