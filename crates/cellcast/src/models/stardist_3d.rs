use burn::prelude::*;
use imgal::error::ImgalError;
use imgal::image::normalize::percentile_normalize;
use imgal::threshold::manual::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, AsArray, Axis, Ix3, ViewRepr};

use crate::config::backend::{CpuBackend, GpuBackend};
use crate::networks::stardist::demo_3d;
use crate::utils::{axes, border};

const DIV: usize = 16;
const N_RAYS: usize = 96;
const PMIN: f64 = 1.0;
const PMAX: f64 = 99.8;
const PROB_THRESHOLD: f64 = 0.7079326182611463;
const NMS_THRESHOLD: f64 = 0.3;

type CpuConfigBackend = CpuBackend<f32, i32>;
type GpuConfigBackend = GpuBackend<f32, i32>;

pub fn predict_demo<'a, T, A>(
    data: A,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
    nms_threshold: Option<f64>,
    axis: Option<usize>,
    gpu: bool,
) -> Result<(Array3<f32>, Array4<f32>), ImgalError>
where
    A: AsArray<'a, T, Ix3>,
    T: 'a + AsNumeric,
{
    let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let pmin = pmin.unwrap_or(PMIN);
    let pmax = pmax.unwrap_or(PMAX);
    let prob_threshold = prob_threshold.unwrap_or(PROB_THRESHOLD) as f32;
    let nms_threshold = nms_threshold.unwrap_or(NMS_THRESHOLD) as f32;
    let axis = axis.unwrap_or(0);
    let norm = percentile_normalize(&data, pmin, pmax, None, None)?;
    let norm = norm.mapv(|v| v as f32);
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
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0))?;
    let mut pad_shape = norm_pad.shape().to_vec();
    let plns = pad_shape.remove(axis);
    let (raw_data, _) = norm_pad.into_raw_vec_and_offset();
    let td = TensorData::new(raw_data, [1, 1, plns, pad_shape[0], pad_shape[1]]);
    let prob: Vec<f32>;
    let dist: Vec<f32>;
    if gpu {
        let device = Default::default();
        let stardist_net = demo_3d::Model::<GpuConfigBackend>::default();
        let tensor = Tensor::<GpuConfigBackend, 5>::from_data(td, &device);
        let (p, d) = stardist_net.forward(
            tensor,
            (plns as i32, pad_shape[0] as i32, pad_shape[1] as i32),
        );
        prob = p.into_data().into_vec().unwrap();
        dist = d.into_data().into_vec().unwrap();
    } else {
        let device = Default::default();
        let stardist_net = demo_3d::Model::<CpuConfigBackend>::default();
        let tensor = Tensor::<CpuConfigBackend, 5>::from_data(td, &device);
        let (p, d) = stardist_net.forward(
            tensor,
            (plns as i32, pad_shape[0] as i32, pad_shape[1] as i32),
        );
        prob = p.into_data().into_vec().unwrap();
        dist = d.into_data().into_vec().unwrap();
    }
    Ok(prob_dist_to_labels_3d(
        prob,
        dist,
        prob_threshold,
        nms_threshold,
        pad_shape,
        plns,
    ))
}

fn prob_dist_to_labels_3d(
    prob: Vec<f32>,
    dist: Vec<f32>,
    prob_threshold: f32,
    nms_threshold: f32,
    pad_shape: Vec<usize>,
    plns: usize,
) -> (Array3<f32>, Array4<f32>) {
    // create arrays from the flat StarDist network output
    let res_row: usize = pad_shape[0] / 2;
    let res_col: usize = pad_shape[1] / 2;
    let prob_arr = Array3::from_shape_vec((plns, res_row, res_col), prob)
        .expect("StarDist 3D object probabilites reshape failed.");
    let dist_arr = Array4::from_shape_vec((N_RAYS, plns, res_row, res_col), dist)
        .expect("StarDist 3D radial distances reshape failed.");
    // ensure all values in the dist array are at least 1e-3, prevents negative and/or zero
    // distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    let mut valid_mask = manual_mask(&prob_arr, prob_threshold);
    border::clip_mask_border(&mut valid_mask.view_mut(), 2);
    let valid_mask = valid_mask.into_dimensionality::<Ix3>().unwrap();
    // collect all valid (pln, row, col) positions to avoid iterating the mask
    // repeatedly
    let valid_pos: Vec<(usize, usize, usize)> = valid_mask
        .indexed_iter()
        .filter(|&((_, _, _), &v)| v)
        .map(|((p, r, c), _)| (p, r, c))
        .collect();
    let flat_pos = valid_pos.iter().flat_map(|&(p, r, c)| [p, r, c]).collect();
    let mut valid_pos = Array2::from_shape_vec((valid_pos.len(), 3), flat_pos).unwrap();
    // filter probabilities and distances with valid indices, removing invalid
    // positions
    let mut valid_prob = Array1::from_iter(
        valid_pos
            .axis_iter(Axis(0))
            .map(|v| prob_arr[[v[0], v[1], v[2]]]),
    );
    dbg!(valid_prob);
    (prob_arr, dist_arr)
}
