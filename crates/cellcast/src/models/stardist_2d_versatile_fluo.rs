use burn::backend::{NdArray, Wgpu};
use burn::prelude::*;
use imgal::error::ImgalError;
use imgal::image::percentile_normalize;
use imgal::threshold::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array1, Array2, Array3, ArrayBase, AsArray, Axis, Ix2, ViewRepr};

use crate::labeling;
use crate::networks::stardist::versatile_fluo_2d::Model;
use crate::process::nms;
use crate::utils::{axes, border};

type NdArrayBackend = NdArray<f32, i32>;
type WgpuBackend = Wgpu<f32, i32>;

const DIV: usize = 16;
const N_RAYS: usize = 32;
const PROB_THRESHOLD: f64 = 0.479071463157368;
const NMS_THRESHOLD: f64 = 0.3;

/// Predict object labels with the StarDist2D versatile fluo model.
///
/// # Description
///
/// Performs inference and instance segmentations with the StarDist2D versatile
/// fluo model. Input images into the StarDist2D network *must* be normalized
/// first. Specify the minimum and maximum percentage to normalize the input
/// image with `pmin` and `pmax`.
///
/// # Arguments
///
/// * `data`: The input 2-dimensional image.
/// * `pmin`: The minimum percentage to linear percentile normalize the input
///   image. If `None`, then `pmin = 1.0`.
/// * `pmax`: The maximum percentage to linear percentile normalize the input
///   image. If `None`, then `pmax = 99.8`.
/// * `prob_threshold`: The object/polygon probability threshold. If `None`,
///    then `prob_threshold == 0.479071463157368`.
/// * `nms_threshold`: The non-maximum suppression (NMS) threshold. If `None`,
///    then `nms_threshold == 0.3`.
/// * `gpu`: If `true`, GPU computation is used with the `Wgpu` backend. If
///   `false` CPU computation is used with the `NdArray` backend.
///
/// # Returns
///
/// * `Ok(Array2<u16>)`: The StarDist2D model label image.
/// * `Err(ImgalError)`: If `pmin` and/or `pmax` are outside of range `0.0` to
///   `1.0.`
///
/// # Reference
///
/// <https://doi.org/10.48550/arXiv.1806.03535>
pub fn predict<'a, T, A>(
    data: A,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
    nms_threshold: Option<f64>,
    gpu: bool,
) -> Result<Array2<u16>, ImgalError>
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    let view: ArrayBase<ViewRepr<&'a T>, Ix2> = data.into();
    let (src_row, src_col) = view.dim();
    let pmin = pmin.unwrap_or(1.0);
    let pmax = pmax.unwrap_or(99.8);
    let prob_threshold = prob_threshold.unwrap_or(PROB_THRESHOLD) as f32;
    let nms_threshold = nms_threshold.unwrap_or(NMS_THRESHOLD) as f32;
    let norm = percentile_normalize(&view, pmin, pmax, None, None)?;
    let norm = norm.mapv(|v| v as f32);
    // the StarDist model expects 2D images with axes that are divisible by 16
    let pad_config: Vec<usize> = view
        .shape()
        .iter()
        .map(|&v| axes::divisible_pad(v, DIV))
        .collect();
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0))?;
    let pad_shape = norm_pad.shape().to_vec();
    // GPU and CPU computes must be in their own scope, the "device",
    // "stardist_net" and "tensor" types are all connected
    let prob: Vec<f32>;
    let dist: Vec<f32>;
    if gpu {
        let device = Default::default();
        let stardist_net = Model::<WgpuBackend>::default();
        let tensor = Tensor::<WgpuBackend, 1>::from_floats(
            norm_pad.into_flat().as_slice().unwrap(),
            &device,
        );
        let (p, d) = stardist_net.forward(tensor, (pad_shape[0] as i32, pad_shape[1] as i32));
        prob = p.into_data().into_vec().unwrap();
        dist = d.into_data().into_vec().unwrap();
    } else {
        let device = Default::default();
        let stardist_net = Model::<NdArrayBackend>::default();
        let tensor = Tensor::<NdArrayBackend, 1>::from_floats(
            norm_pad.into_flat().as_slice().unwrap(),
            &device,
        );
        let (p, d) = stardist_net.forward(tensor, (pad_shape[0] as i32, pad_shape[1] as i32));
        prob = p.into_data().into_vec().unwrap();
        dist = d.into_data().into_vec().unwrap();
    }
    // outputs from the StarDist network are flat 1D arrays, they need to be
    // reshapped back into their input shape divided by 2
    let res_row: usize = pad_shape[0] / 2;
    let res_col: usize = pad_shape[1] / 2;
    let prob_arr = Array2::from_shape_vec((res_row, res_col), prob)
        .expect("StarDist 2D object probabilites reshape failed.");
    let dist_arr = Array3::from_shape_vec((res_row, res_col, N_RAYS), dist)
        .expect("StarDist 2D radial distances reshape failed.");
    // this mapv call ensures all values in the ray distances array are at least
    // 1e-3 which prevents negative and/or zero distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    let mut valid_mask = manual_mask(&prob_arr, prob_threshold);
    border::clip_mask_border(&mut valid_mask.view_mut(), 2);
    let valid_mask = valid_mask.into_dimensionality::<Ix2>().unwrap();
    // collect all valid (row, col) positions to avoid iterating the mask
    // repeatedly
    let valid_pos: Vec<(usize, usize)> = valid_mask
        .indexed_iter()
        .filter(|&((_, _), &v)| v)
        .map(|((r, c), _)| (r, c))
        .collect();
    let flat_pos = valid_pos.iter().flat_map(|&(r, c)| [r, c]).collect();
    let mut valid_pos = Array2::from_shape_vec((valid_pos.len(), 2), flat_pos).unwrap();
    // filter probabilities and distances with valid indices, removing invalid
    // positions
    let mut valid_prob =
        Array1::from_iter(valid_pos.axis_iter(Axis(0)).map(|v| prob_arr[[v[0], v[1]]]));
    let mut valid_dist = Array2::<f32>::zeros((valid_pos.len(), N_RAYS));
    (0..N_RAYS).for_each(|n| {
        valid_pos.axis_iter(Axis(0)).enumerate().for_each(|(i, v)| {
            valid_dist[[i, n]] = dist_arr[[v[0], v[1], n]];
        });
    });
    // scale each valid position by 2 and collect the valid indices of positions
    // inside of the source image dimensions (used for point filtering)
    valid_pos.mapv_inplace(|v| v * 2);
    let valid_inds: Vec<usize> = valid_pos
        .axis_iter(Axis(0))
        .enumerate()
        .filter_map(|(i, v)| {
            if v[0] < src_row || v[1] < src_col {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    // remove invalid indices (if there are any) from dist, prob and pos
    let poly_ax = Axis(0);
    if valid_pos.len() > valid_inds.len() {
        valid_dist = valid_dist.select(poly_ax, &valid_inds);
        valid_prob = valid_prob.select(poly_ax, &valid_inds);
        valid_pos = valid_pos.select(poly_ax, &valid_inds);
    }
    // get the indices that would sort probs in descending order
    let n_polys = valid_prob.len();
    let mut sorted_poly_inds: Vec<usize> = (0..n_polys).collect();
    sorted_poly_inds.sort_by(|&a, &b| valid_prob[b].partial_cmp(&valid_prob[a]).unwrap());
    // sort dist, prob and pos arrays with prob descending order indices
    let poly_dist = valid_dist.select(poly_ax, &sorted_poly_inds);
    let poly_pos = valid_pos.select(poly_ax, &sorted_poly_inds);
    // perform non-maximum supression (NMS) and obtain indices of valid polygons
    let valid_poly_inds = nms::polygon_nms_2d(
        poly_dist.view(),
        poly_pos.view(),
        n_polys,
        N_RAYS,
        nms_threshold,
    );
    let valid_poly_inds: Vec<usize> = valid_poly_inds
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v)
        .map(|(i, _)| i)
        .collect();
    // filter dist, prob and pos arrays with for valid polygons after NMS
    let poly_dist = poly_dist.select(poly_ax, &valid_poly_inds);
    let poly_prob = valid_prob.select(poly_ax, &valid_poly_inds);
    let poly_pos = poly_pos.select(poly_ax, &valid_poly_inds);
    // filter dist, prob and pos arrays by probability threshold
    let valid_prob_inds: Vec<usize> = (0..poly_prob.len())
        .filter(|&i| poly_prob[i] > prob_threshold)
        .collect();
    let poly_dist = poly_dist.select(poly_ax, &valid_prob_inds);
    let poly_prob = poly_prob.select(poly_ax, &valid_prob_inds);
    let poly_pos = poly_pos.select(poly_ax, &valid_prob_inds);
    // convert radial distances and polygons to labels
    let labels = labeling::radial_polygon_to_label_2d(
        poly_dist.view(),
        poly_prob.view(),
        poly_pos.view(),
        (src_row, src_col),
        None,
    );

    Ok(labels)
}
