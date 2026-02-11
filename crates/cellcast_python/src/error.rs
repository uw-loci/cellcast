use imgal::error::ImgalError;
use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

/// Convert an ImgalError into a RuntimeError PyErr
///
/// This is a quick/easy way to map Imgal's errors that avoids having to
/// duplicate imgal_python's map_imgal_error structure. This unfortunately,
/// casts all errors as RuntimeErrors which is untrue.
pub fn imgal_error_to_pyerr(err: ImgalError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
