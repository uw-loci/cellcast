use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

use cellcast::CellcastError;

/// Convert a CellcastError into a RuntimeError PyErr
///
/// This is a quick/easy way to map cellcast's errors.
pub fn cellcast_error_to_pyerr(err: CellcastError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
