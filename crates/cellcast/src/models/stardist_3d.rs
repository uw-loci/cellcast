use burn::prelude::*;
use imgal::error::ImgalError;
use imgal::image::percentile_normalize;
use imgal::threshold::manual::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, AsArray, Axis, Ix3, ViewRepr};

use crate::config::backend::{CpuBackend, GpuBackend};
use crate::networks::stardist::demo_3d;
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
    let axis = axis.unwrap_or(0);
    if axis >= 3 {
        return Err(ImgalError::InvalidAxis {
            axis_idx: axis,
            dim_len: 3,
        });
    }
    let data: ArrayBase<ViewRepr<&'a T>, Ix3> = data.into();
    let pmin = pmin.unwrap_or(PMIN);
    let pmax = pmax.unwrap_or(PMAX);
    let prob_threshold = prob_threshold.unwrap_or(PROB_THRESHOLD) as f32;
    let nms_threshold = nms_threshold.unwrap_or(NMS_THRESHOLD) as f32;
    let norm = percentile_normalize(&data, pmin, pmax, false, None, None, false)?;
    let norm = norm.mapv(|v| v as f32);
    let (src_row, src_col) = {
        let mut shape = data.shape().to_vec();
        shape.remove(axis);
        (shape[0], shape[1])
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
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0), false)?;
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
        (plns, src_row, src_col),
    ))
}

fn prob_dist_to_labels_3d(
    prob: Vec<f32>,
    dist: Vec<f32>,
    prob_threshold: f32,
    nms_threshold: f32,
    pad_shape: Vec<usize>,
    src_shape: (usize, usize, usize),
) -> (Array3<f32>, Array4<f32>) {
    // create arrays from the flat StarDist network output
    let res_row: usize = pad_shape[0] / 2;
    let res_col: usize = pad_shape[1] / 2;
    let prob_arr = Array3::from_shape_vec((src_shape.0, res_row, res_col), prob)
        .expect("StarDist 3D object probabilites reshape failed.");
    let dist_arr = Array4::from_shape_vec((N_RAYS, src_shape.0, res_row, res_col), dist)
        .expect("StarDist 3D radial distances reshape failed.");
    // ensure all values in the dist array are at least 1e-3, prevents negative and/or zero
    // distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    let mut valid_mask = manual_mask(&prob_arr, prob_threshold, false);
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
            if v[0] < src_shape.0 || v[1] < src_shape.1 || v[2] < src_shape.2 {
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
    let valid_poly_inds = polyhedron_nms(
        poly_dist.view(),
        poly_pnts.view(),
        poly_prob.view(),
        n_polys,
        N_RAYS,
        nms_threshold,
    );
    (prob_arr, dist_arr)
}
