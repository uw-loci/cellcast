use std::path::PathBuf;

use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::prelude::*;
use imgal::threshold::manual::manual_mask;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, AsArray, Axis, Ix3, ViewRepr};

use crate::CellcastError;
use crate::config::backend::{CpuBackend, GpuBackend};
use crate::labeling::distance_polyhedron_to_label;
use crate::networks::stardist::fluo_3d;
use crate::process::nms::polyhedron_nms;
use crate::utils::{axes, border};

const DIV: usize = 16;
const N_RAYS: usize = 96;
const PMIN: f64 = 1.0;
const PMAX: f64 = 99.8;
const PROB_THRESHOLD: f64 = 0.7079326182611463;
const NMS_THRESHOLD: f64 = 0.3;

type CpuConfigBackend = CpuBackend<f32, i32>;
type GpuConfigBackend = GpuBackend<f32, i32>;

#[derive(Debug)]
enum StarDist3DModels {
    FluoCpu(fluo_3d::Model<CpuConfigBackend>),
    FluoGpu(fluo_3d::Model<GpuConfigBackend>),
}

#[derive(Debug)]
pub struct StarDist3D {
    model: StarDist3DModels,
    anisotropy: [f64; 3],
    gpu: bool,
}

impl StarDist3D {
    /// Initialize a StarDist3D fluo model.
    ///
    /// # Description
    ///
    /// Initializes a StarDist3D fluo model using the versatile fluo pretrained
    /// weights or custom weights. A StarDist3D model can be initialized on either
    /// the GPU or CPU, but not both concurrently. The model is pre-warmed with as
    /// part of the initializtion process.
    ///
    /// # Arguments
    ///
    /// * `weights_path`: The path to custom StarDist3D weights in burnpack (`.bpk`)
    ///   format. If `None` then the versatile fluo pretrained weights are used.
    /// * `anisotropy`: The anisotropy the model was trained with for all three
    ///   axes. If `None` then anisotropy of `[2.0, 1.0, 1.0]` is used.
    /// * `gpu`: If `true`, the configured GPU backend is used. If `false` then the
    ///   configured CPU backend is used.
    ///
    /// # Returns
    ///
    /// * `Ok(StarDist2D)`: An initialized StarDist3D fluo model.
    /// * `Err(CellcastError)`: If the requested model can not be initialized. If
    ///   `anisotropy.len() != 3`.
    pub fn init_fluo(
        weights_path: Option<&str>,
        anisotropy: Option<&[f64]>,
        gpu: bool,
    ) -> Result<Self, CellcastError> {
        let weights_path = weights_path.map(PathBuf::from);
        let anisotropy = anisotropy.unwrap_or(&[2.0, 1.0, 1.0]);
        if anisotropy.len() != 3 {
            return Err(ImgalError::InvalidArrayLengthExpected {
                arr_name: "anisotropy",
                expected: 3,
                got: anisotropy.len(),
            })
            .map_err(CellcastError::Imgal);
        }
        let anisotropy = [anisotropy[0], anisotropy[1], anisotropy[2]];
        if gpu {
            let device = Default::default();
            let sd = Self {
                model: StarDist3DModels::FluoGpu(fluo_3d::Model::<GpuConfigBackend>::init(
                    &device,
                    weights_path.clone(),
                )),
                anisotropy,
                gpu,
            };
            sd.warm_up_fluo()?;
            Ok(sd)
        } else {
            let device = Default::default();
            let sd = Self {
                model: StarDist3DModels::FluoCpu(fluo_3d::Model::<CpuConfigBackend>::init(
                    &device,
                    weights_path.clone(),
                )),
                anisotropy,
                gpu,
            };
            sd.warm_up_fluo()?;
            Ok(sd)
        }
    }

    /// Predict instance segmentation labels with the StarDist3D fluo model.
    ///
    /// # Description
    ///
    /// Performs model inference with the StarDist3D fluo model, returning instance
    /// segmentations of star-convex shapes.
    ///
    /// # Arguments
    ///
    /// * `data`: The input 3D image.
    /// * `pmin`: The minimum percentage to linear percentile normalize the input
    ///   image. If `None`, then `pmin = 1.0`.
    /// * `pmax`: The maximum percentage to linear percentile normalize the input
    ///   image. If `None`, then `pmax = 99.8`.
    /// * `prob_threshold`: The object/polyhedron probability threshold. If `None`,
    ///   then `prob_threshold == 0.7079326182611463`.
    /// * `nms_threshold`: The non-maximum suppression (NMS) threshold. If `None`,
    ///   then `nms_threshold == 0.3`.
    /// * `axis`: The `pln` or `z` axis. If `None` then `axis == 0`.
    ///
    /// # Returns
    ///
    /// * `Ok(Array2<u64>)`: The StarDist3D fluo model instance segmentation label
    ///   image.
    /// * `Err(CellcastError)`: If `pmin` and/or `pmax` are outside of range
    ///   `0.0` to `1.0.` If `axis >= 3`.
    ///
    /// # Reference
    ///
    /// <https://doi.org/10.1109/WACV45572.2020.9093435>
    pub fn predict_fluo<'a, T, A>(
        &self,
        data: A,
        pmin: Option<f64>,
        pmax: Option<f64>,
        prob_threshold: Option<f64>,
        nms_threshold: Option<f64>,
        axis: Option<usize>,
    ) -> Result<Array3<u64>, CellcastError>
    where
        A: AsArray<'a, T, Ix3>,
        T: 'a + AsNumeric,
    {
        let axis = axis.unwrap_or(0);
        if axis >= 3 {
            return Err(ImgalError::InvalidAxis {
                axis_idx: axis,
                dim_len: 3,
            })
            .map_err(CellcastError::Imgal);
        }
        let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
        let pmin = pmin.unwrap_or(PMIN);
        let pmax = pmax.unwrap_or(PMAX);
        let prob_threshold = prob_threshold.unwrap_or(PROB_THRESHOLD) as f32;
        let nms_threshold = nms_threshold.unwrap_or(NMS_THRESHOLD) as f32;
        let norm = percentile_normalize(&data, pmin, pmax, false, None, None, None)?;
        let norm = norm.mapv(|v| v as f32);
        let [src_row, src_col] = {
            let mut shape = data.shape().to_vec();
            shape.remove(axis);
            [shape[0], shape[1]]
        };
        // this pattern determines how many pixels to pad in each axis to be
        // divisible by 16 as expected by the network, except for the planes (z)
        // axis which remains fixed (i.e. an asymmetrical pad)
        let pad_config: Vec<usize> = data
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i == axis {
                    0
                } else {
                    axes::divisible_pad(v, DIV)
                }
            })
            .collect();
        let norm_pad = reflect_pad(&norm, &pad_config, Some(0), None)?;
        let mut pad_shape = norm_pad.shape().to_vec();
        let plns = pad_shape.remove(axis);
        let (raw_data, _) = norm_pad.into_raw_vec_and_offset();
        let td = TensorData::new(raw_data, [1, 1, plns, pad_shape[0], pad_shape[1]]);
        let prob: Vec<f32>;
        let dist: Vec<f32>;
        if self.gpu {
            match &self.model {
                StarDist3DModels::FluoGpu(m) => {
                    let device = Default::default();
                    let tensor = Tensor::<GpuConfigBackend, 5>::from_data(td, &device);
                    let (p, d) = m.forward(
                        tensor,
                        (plns as i32, pad_shape[0] as i32, pad_shape[1] as i32),
                    );
                    prob = p.into_data().into_vec().unwrap();
                    dist = d.into_data().into_vec().unwrap();
                }
                _ => {
                    return Err(ImgalError::InvalidGeneric {
                        msg: "No initialized StarDist3D Fluo GPU model found.",
                    })
                    .map_err(CellcastError::Imgal);
                }
            }
        } else {
            match &self.model {
                StarDist3DModels::FluoCpu(m) => {
                    let device = Default::default();
                    let tensor = Tensor::<CpuConfigBackend, 5>::from_data(td, &device);
                    let (p, d) = m.forward(
                        tensor,
                        (plns as i32, pad_shape[0] as i32, pad_shape[1] as i32),
                    );
                    prob = p.into_data().into_vec().unwrap();
                    dist = d.into_data().into_vec().unwrap();
                }
                _ => {
                    return Err(ImgalError::InvalidGeneric {
                        msg: "No initialized StarDist3D Fluo CPU model found.",
                    })
                    .map_err(CellcastError::Imgal);
                }
            }
        }
        prob_dist_to_labels_3d(
            prob,
            dist,
            prob_threshold,
            nms_threshold,
            self.anisotropy,
            pad_shape,
            [plns, src_row, src_col],
        )
        .map_err(CellcastError::Imgal)
    }

    /// Warm up the StarDist3D fluo model.
    ///
    /// # Description
    ///
    /// Warms up the StarDist3D fluo model by creating a small tensor of zeros
    /// and passing it to the initialized model. During this time model
    /// optimizations like autotuning are performed.
    ///
    /// # Returns
    ///
    /// * `Ok(())`: If successful.
    /// * `Err(CellcastError)`: If the requested model can not be initialized.
    fn warm_up_fluo(&self) -> Result<(), CellcastError> {
        let zeros = vec![0.0; 131072];
        let td = TensorData::new(zeros, [1, 1, 32, 64, 64]);
        if self.gpu {
            match &self.model {
                StarDist3DModels::FluoGpu(m) => {
                    let device = Default::default();
                    let tensor = Tensor::<GpuConfigBackend, 5>::from_data(td, &device);
                    let _ = m.forward(tensor, (32, 64, 64));
                    Ok(())
                }
                _ => {
                    return Err(ImgalError::InvalidGeneric {
                        msg: "No initialized StarDist3D Fluo GPU model found.",
                    })
                    .map_err(CellcastError::Imgal);
                }
            }
        } else {
            match &self.model {
                StarDist3DModels::FluoCpu(m) => {
                    let device = Default::default();
                    let tensor = Tensor::<CpuConfigBackend, 5>::from_data(td, &device);
                    let _ = m.forward(tensor, (32, 64, 64));
                    Ok(())
                }
                _ => {
                    return Err(ImgalError::InvalidGeneric {
                        msg: "No initialized StarDist2D Fluo CPU model found.",
                    })
                    .map_err(CellcastError::Imgal);
                }
            }
        }
    }
}

/// Process StarDist3D object probabilities and ray distance arrays into
/// instance segmentations.
///
/// # Arguments
///
/// * `prob`: The object probabilities as a flat 1D array.
/// * `dist`: The ray distances as a flat 1D array.
/// * `prob_threshold`: The object probability threshold.
/// * `nms_threshold`: The non-maximum suppression threshold.
/// * `anisotropy`: The anisotropy the model was trained with for all three
///   axes.
/// * `pad_shape`: The padded image shape.
/// * `src_shape`: The original/source image shape.
///
/// # Returns
///
/// * `Array2<u64>`: The instance segmentation label image.
fn prob_dist_to_labels_3d(
    prob: Vec<f32>,
    dist: Vec<f32>,
    prob_threshold: f32,
    nms_threshold: f32,
    anisotropy: [f64; 3],
    pad_shape: Vec<usize>,
    src_shape: [usize; 3],
) -> Result<Array3<u64>, ImgalError> {
    // create arrays from the flat StarDist network output
    let res_row: usize = pad_shape[0] / 2;
    let res_col: usize = pad_shape[1] / 2;
    let prob_arr = Array3::from_shape_vec((src_shape[0], res_row, res_col), prob)
        .expect("StarDist 3D object probabilites reshape failed.");
    let dist_arr = Array4::from_shape_vec((N_RAYS, src_shape[0], res_row, res_col), dist)
        .expect("StarDist 3D radial distances reshape failed.");
    // ensure all values in the dist array are at least 1e-3, prevents negative and/or zero
    // distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    let mut valid_mask = manual_mask(&prob_arr, prob_threshold, None);
    border::clip_mask_border(&mut valid_mask.view_mut().into_dyn(), 2);
    let valid_mask = valid_mask.into_dimensionality::<Ix3>().unwrap();
    // collect all valid (pln, row, col) positions to avoid iterating the mask
    // repeatedly
    let valid_pnts: Vec<(usize, usize, usize)> = valid_mask
        .indexed_iter()
        .filter(|&((_, _, _), &v)| v)
        .map(|((p, r, c), _)| (p, r, c))
        .collect();
    let flat_pnts = valid_pnts.iter().flat_map(|&(p, r, c)| [p, r, c]).collect();
    let mut valid_pnts = Array2::from_shape_vec((valid_pnts.len(), 3), flat_pnts).unwrap();
    // filter probabilities and distances with valid indices, removing invalid
    // positions
    let mut valid_prob = Array1::from_iter(
        valid_pnts
            .axis_iter(Axis(0))
            .map(|v| prob_arr[[v[0], v[1], v[2]]]),
    );
    let mut valid_dist = Array2::<f32>::zeros((valid_pnts.dim().0, N_RAYS));
    (0..N_RAYS).for_each(|n| {
        valid_pnts
            .axis_iter(Axis(0))
            .enumerate()
            .for_each(|(i, v)| {
                valid_dist[[i, n]] = dist_arr[[n, v[0], v[1], v[2]]];
            });
    });
    // scale each valid position by 2 and collect the valid indices of positions
    // inside of the source image dimensions (used for point filtering)
    let poly_ax = Axis(0);
    valid_pnts.axis_iter_mut(poly_ax).for_each(|mut v| {
        v[1] = v[1] * 2;
        v[2] = v[2] * 2;
    });
    let valid_inds: Vec<usize> = valid_pnts
        .axis_iter(poly_ax)
        .enumerate()
        .filter_map(|(i, v)| {
            if v[0] < src_shape[0] || v[1] < src_shape[1] || v[2] < src_shape[2] {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    // remove invalid indices (if there are any) from dist, prob and pos
    if valid_pnts.len() > valid_inds.len() {
        valid_dist = valid_dist.select(poly_ax, &valid_inds);
        valid_prob = valid_prob.select(poly_ax, &valid_inds);
        valid_pnts = valid_pnts.select(poly_ax, &valid_inds);
    }
    // get the indices that would sort probs in descending order
    let n_polys = valid_prob.len();
    let mut sorted_poly_inds: Vec<usize> = (0..n_polys).collect();
    sorted_poly_inds.sort_by(|&a, &b| valid_prob[b].partial_cmp(&valid_prob[a]).unwrap());
    // sort dist, prob and pos arrays with prob descending order indices
    let poly_dist = valid_dist.select(poly_ax, &sorted_poly_inds);
    let poly_pnts = valid_pnts.select(poly_ax, &sorted_poly_inds);
    let poly_prob = valid_prob.select(poly_ax, &sorted_poly_inds);
    let poly_pnts = poly_pnts.mapv(|v| v as f32);
    let valid_poly_inds = polyhedron_nms(
        poly_dist.view(),
        poly_pnts.view(),
        anisotropy,
        n_polys,
        N_RAYS,
        nms_threshold,
    )
    .unwrap();
    // here we select the valid polyhedrons and construct the 3D labels
    let valid_poly_inds: Vec<usize> = valid_poly_inds
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v)
        .map(|(i, _)| i)
        .collect();
    let poly_dist = poly_dist.select(poly_ax, &valid_poly_inds);
    let poly_pnts = poly_pnts.select(poly_ax, &valid_poly_inds);
    let poly_prob = poly_prob.select(poly_ax, &valid_poly_inds);
    distance_polyhedron_to_label(
        poly_dist.view(),
        poly_pnts.view(),
        poly_prob.view(),
        prob_threshold,
        anisotropy,
        src_shape,
    )
}
