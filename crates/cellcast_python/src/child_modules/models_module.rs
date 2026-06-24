use pyo3::prelude::*;

use crate::classes::stardist_classes::{PyStarDist2D, PyStarDist3D};
use crate::utils::py_import_module;

/// Registration function for the "models" module and their submodules.
pub fn register_models_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let models_module = PyModule::new(parent_module.py(), "models")?;
    py_import_module("models");
    py_import_module("models.StarDist2D");
    py_import_module("models.StarDist3D");
    models_module.add_class::<PyStarDist2D>()?;
    models_module.add_class::<PyStarDist3D>()?;
    parent_module.add_submodule(&models_module)
}
