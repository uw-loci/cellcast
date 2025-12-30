use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use cellcast::models::stardist_2d_versatile_fluo;

/// Perform inference with the Stardist 2-dimensional versatile fluo model.
#[pyfunction]
#[pyo3(name = "predict")]
#[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold=None))]
pub fn models_stardist_2d_versatile_fluo<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<u16>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
        let labels =
            stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold);
        return Ok(labels.into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        let labels =
            stardist_2d_versatile_fluo::predict(arr.as_array(), pmin, pmax, prob_threshold);
        return Ok(labels.into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        let labels =
            stardist_2d_versatile_fluo::predict(arr.as_array(), pmin, pmax, prob_threshold);
        return Ok(labels.into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let labels =
            stardist_2d_versatile_fluo::predict(arr.as_array(), pmin, pmax, prob_threshold);
        return Ok(labels.into_pyarray(py));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let labels =
            stardist_2d_versatile_fluo::predict(arr.as_array(), pmin, pmax, prob_threshold);
        return Ok(labels.into_pyarray(py));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
