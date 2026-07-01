use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::error::cellcast_error_to_pyerr;
use cellcast::models::{StarDist2D, StarDist3D};

#[pyclass(name = "StarDist2D")]
pub struct PyStarDist2D(StarDist2D);

#[pymethods]
impl PyStarDist2D {
    /// TODO
    #[staticmethod]
    #[pyo3(signature = (weights_path=None, gpu=None))]
    pub fn init_fluo(weights_path: Option<&str>, gpu: Option<bool>) -> PyResult<Self> {
        Ok(Self(
            StarDist2D::init_fluo(weights_path, gpu.unwrap_or(true))
                .map_err(cellcast_error_to_pyerr)?,
        ))
    }

    /// TODO
    #[staticmethod]
    #[pyo3(signature = (weights_path=None, gpu=None))]
    pub fn init_he(weights_path: Option<&str>, gpu: Option<bool>) -> PyResult<Self> {
        Ok(Self(
            StarDist2D::init_he(weights_path, gpu.unwrap_or(true))
                .map_err(cellcast_error_to_pyerr)?,
        ))
    }

    /// TODO
    #[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold=None, nms_threshold=None))]
    pub fn predict_fluo<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        pmin: Option<f64>,
        pmax: Option<f64>,
        prob_threshold: Option<f64>,
        nms_threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
            self.0
                .predict_fluo(arr.as_array(), pmin, pmax, prob_threshold, nms_threshold)
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
            self.0
                .predict_fluo(arr.as_array(), pmin, pmax, prob_threshold, nms_threshold)
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
            self.0
                .predict_fluo(arr.as_array(), pmin, pmax, prob_threshold, nms_threshold)
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
            self.0
                .predict_fluo(arr.as_array(), pmin, pmax, prob_threshold, nms_threshold)
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
            self.0
                .predict_fluo(arr.as_array(), pmin, pmax, prob_threshold, nms_threshold)
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
            ))
        }
    }

    /// TODO
    #[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold=None, nms_threshold=None, axis=None))]
    pub fn predict_he<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        pmin: Option<f64>,
        pmax: Option<f64>,
        prob_threshold: Option<f64>,
        nms_threshold: Option<f64>,
        axis: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
            self.0
                .predict_he(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
            self.0
                .predict_he(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
            self.0
                .predict_he(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
            self.0
                .predict_he(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
            self.0
                .predict_he(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
            ))
        }
    }
}

#[pyclass(name = "StarDist3D")]
pub struct PyStarDist3D(StarDist3D);

#[pymethods]
impl PyStarDist3D {
    /// TODO
    #[staticmethod]
    #[pyo3(signature = (weights_path=None, anisotropy=None, gpu=None))]
    pub fn init_fluo(
        weights_path: Option<&str>,
        anisotropy: Option<Vec<f64>>,
        gpu: Option<bool>,
    ) -> PyResult<Self> {
        let anisotropy = anisotropy.as_deref();
        Ok(Self(
            StarDist3D::init_fluo(weights_path, anisotropy, gpu.unwrap_or(true))
                .map(|output| output)
                .map_err(cellcast_error_to_pyerr)?,
        ))
    }

    /// TODO
    #[pyo3(signature = (data, pmin=None, pmax=None, prob_threshold=None, nms_threshold=None, axis=None))]
    pub fn predict_fluo<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        pmin: Option<f64>,
        pmax: Option<f64>,
        prob_threshold: Option<f64>,
        nms_threshold: Option<f64>,
        axis: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray3<u64>>> {
        if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
            self.0
                .predict_fluo(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
            self.0
                .predict_fluo(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
            self.0
                .predict_fluo(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
            self.0
                .predict_fluo(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
            self.0
                .predict_fluo(
                    arr.as_array(),
                    pmin,
                    pmax,
                    prob_threshold,
                    nms_threshold,
                    axis,
                )
                .map(|output| output.into_pyarray(py))
                .map_err(cellcast_error_to_pyerr)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Unsupported array dtype, supported array dtypes are u8, u16, u64, f32, and f64.",
            ))
        }
    }
}
