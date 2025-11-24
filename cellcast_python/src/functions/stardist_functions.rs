use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use cellcast::models::stardist_2d_versatile_fluo;

/// Perform inference with the Stardist 2-dimensional versatile fluo model.
#[pyfunction]
#[pyo3(name = "predict")]
pub fn models_stardist_2d_versatile_fluo<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<f32>>)> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
        let (prob_arr, dist_arr) = stardist_2d_versatile_fluo::predict(&arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        let (prob_arr, dist_arr) = stardist_2d_versatile_fluo::predict(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        let (prob_arr, dist_arr) = stardist_2d_versatile_fluo::predict(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let (prob_arr, dist_arr) = stardist_2d_versatile_fluo::predict(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let (prob_arr, dist_arr) = stardist_2d_versatile_fluo::predict(arr.as_array());
        return Ok((prob_arr.into_pyarray(py), dist_arr.into_pyarray(py)));
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64."
        ))
    }
}
