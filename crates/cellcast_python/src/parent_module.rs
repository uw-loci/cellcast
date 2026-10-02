use pyo3::prelude::*;

/// GPU agnostic cell segmentation models for Python.
///
/// cellcast is a collection of "End-of-life" (EOL) cell segmentation models
/// reimplemented (*i.e.* a recast) with the Burn tensor and deep learning
/// framework. This library aims to make access to cell segmentation models
/// easier and GPU hardware agnostic.
#[pymodule(name = "cellcast")]
mod cellcast_module {
    #[pymodule_export]
    use super::models;
}

/// Cell segmentation models.
///
/// This module contains the supported cellcast cell segmentation models. Each
/// model is first initialized on the GPU or CPU with either fetched pre-trained
/// weights or custom weights.
#[pymodule(submodule)]
mod models {
    #[pymodule_export]
    use crate::classes::stardist_classes::PyStarDist2D;
    #[pymodule_export]
    use crate::classes::stardist_classes::PyStarDist3D;
}
