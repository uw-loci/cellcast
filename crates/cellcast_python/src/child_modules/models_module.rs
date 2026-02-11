use pyo3::prelude::*;

use crate::functions::stardist_functions;
use crate::utils::py_import_module;

/// Python bindings for the "stardist" submodule.
pub fn register_models_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let models_module = PyModule::new(parent_module.py(), "models")?;
    let stardist_2d_versatile_fluo_module =
        PyModule::new(parent_module.py(), "stardist_2d_versatile_fluo")?;

    // add module to Python's sys.modules
    py_import_module("models");
    py_import_module("models.stardist_2d_versatile_fluo");

    // add models::stardist submodule functions
    stardist_2d_versatile_fluo_module.add_function(wrap_pyfunction!(
        stardist_functions::models_stardist_2d_versatile_fluo,
        &stardist_2d_versatile_fluo_module
    )?)?;

    // attach "models" submodules before attaching to the parent module
    models_module.add_submodule(&stardist_2d_versatile_fluo_module)?;
    parent_module.add_submodule(&models_module)
}
