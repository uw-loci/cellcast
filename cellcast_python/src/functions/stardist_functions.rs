use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::imgal_error_to_pyerr;
use cellcast::models::stardist_2d_versatile_fluo;

/// Predict object labels with the StarDist2D versatile fluo model.
///
/// Performs inference and instance segmentations with the StarDist2D versatile
/// fluo model. Input images into the StarDist2D network *must* be normalized
/// first. Specify the minimum and maximum percentage to normalize the input
/// image with `pmin` and `pmax`.
///
/// Args:
///     data: The input 2-dimensional image.
///     pmin: The minimum percentage to linear percentile normalize the input
///         image. If `None`, then `pmin = 1.0`.
///     pmax: The maximum percentage to linear percentile normalize the input
///         image. If `None`, then `pmax = 99.8`.
///     prob_threshold: Optional object/polygon probability threshold. If
///         `None`, then `prob_threshold == 0.479071463157368`.
///
/// Returns:
///     The StarDist2D model label image.
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
        stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold)
            .map(|output| output.into_pyarray(py))
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold)
            .map(|output| output.into_pyarray(py))
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold)
            .map(|output| output.into_pyarray(py))
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold)
            .map(|output| output.into_pyarray(py))
            .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        stardist_2d_versatile_fluo::predict(&arr.as_array(), pmin, pmax, prob_threshold)
            .map(|output| output.into_pyarray(py))
            .map_err(imgal_error_to_pyerr)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}
