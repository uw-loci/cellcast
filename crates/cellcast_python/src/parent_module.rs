use pyo3::prelude::*;

use super::child_modules::models_module;

/// Python bindings for the cellcast parent model.
#[pymodule(name = "cellcast")]
fn cellcast_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // register child modules
    models_module::register_models_module(m)?;

    Ok(())
}
