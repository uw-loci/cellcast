use pyo3::prelude::*;

use crate::functions::stardist_functions;
use crate::utils::py_import_module;

/// Registration function for the "models" module and their submodules.
pub fn register_models_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let models_module = PyModule::new(parent_module.py(), "models")?;
    let stardist_2d_module = PyModule::new(parent_module.py(), "stardist_2d")?;
    py_import_module("models");
    py_import_module("models.stardist_2d");
    stardist_2d_module.add_function(wrap_pyfunction!(
        stardist_functions::stardist_2d_predict_versatile_fluo,
        &stardist_2d_module
    )?)?;
    stardist_2d_module.add_function(wrap_pyfunction!(
        stardist_functions::stardist_2d_predict_versatile_he,
        &stardist_2d_module
    )?)?;
    models_module.add_submodule(&stardist_2d_module)?;
    parent_module.add_submodule(&models_module)
}
