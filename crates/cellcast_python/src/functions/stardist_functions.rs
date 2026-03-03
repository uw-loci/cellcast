use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::imgal_error_to_pyerr;
use cellcast::models::stardist_2d::{predict_versatile_fluo, predict_versatile_he};

/// Predict instance segmentation labels with the StarDist2D versatile fluo
/// model.
///
/// Performs model inference with the StarDist2D versatile fluo model, returning
/// instance segmentations of star-convex shapes.
///
/// Args:
///     data: The input 2D image.
///     pmin: The minimum percentage to linear percentile normalize the input
///         image. If `None`, then `pmin = 1.0`.
///     pmax: The maximum percentage to linear percentile normalize the input
///         image. If `None`, then `pmax = 99.8`.
///     prob_threshold: The object/polygon probability threshold. If `None`,
///         then `prob_threshold == 0.479071463157368`.
///     nms_threshold: The non-maximum suppression (NMS) threshold. If `None`,
///         then `nms_threshold == 0.3`.
///     gpu: If `True`, GPU computation is used with the `Wgpu` backend. If
///         `False`, then CPU computation is used with the `NdArray` backend. If
///         `None` then `gpu == True`.
///
/// Returns:
///     The StarDist2D fluo model instance segmentation label image.
#[pyfunction]
#[pyo3(name = "predict_versatile_fluo")]
#[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold=None, nms_threshold=None, gpu=None))]
pub fn stardist_2d_predict_versatile_fluo<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
    nms_threshold: Option<f64>,
    gpu: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let gpu = gpu.unwrap_or(true);
    if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
        predict_versatile_fluo(
            &arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        predict_versatile_fluo(
            &arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        predict_versatile_fluo(
            &arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        predict_versatile_fluo(
            &arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        predict_versatile_fluo(
            &arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ));
    }
}

/// Predict instance segmentation labels with the StarDist2D versatile HE model.
///
/// # Description
///
/// Performs model inference with the StarDist2D versatile HE model, returning
/// instance segmentations of star-convex shapes.
///
/// Args:
///     data: The input 3D image, where the third dimension is the channel axis.
///     pmin: The minimum percentage to linear percentile normalize the input
///         image. If `None`, then `pmin = 1.0`.
///     pmax: The maximum percentage to linear percentile normalize the input
///         image. If `None`, then `pmax = 99.8`.
///     prob_threshold: The object/polygon probability threshold. If `None`,
///         then `prob_threshold == 0.6924782541382084`.
///     nms_threshold: The non-maximum suppression (NMS) threshold. If `None`,
///         then `nms_threshold == 0.3`.
///     axis: The channel axis. If `None` then `axis == 2`.
///     gpu: If `True`, GPU computation is used with the `Wgpu` backend. If
///         `False`, then CPU computation is used with the `NdArray` backend. If
///         `None` then `gpu == True`.
///
/// Returns:
///     The StarDist2D HE model instance segmentation label image.
#[pyfunction]
#[pyo3(name = "predict_versatile_he")]
#[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold = None, nms_threshold=None, axis=None, gpu=None))]
pub fn stardist_2d_predict_versatile_he<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
    nms_threshold: Option<f64>,
    axis: Option<usize>,
    gpu: Option<bool>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let gpu = gpu.unwrap_or(true);
    if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
        predict_versatile_he(
            arr.as_array(),
            pmin,
            pmax,
            prob_threshold,
            nms_threshold,
            axis,
            gpu,
        )
        .map(|output| output.into_pyarray(py))
        .map_err(imgal_error_to_pyerr)
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
        ))
    }
}
