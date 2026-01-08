use burn::backend::Wgpu;
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

type Backend = Wgpu<f32, i32>;

const DIV: usize = 16;
const N_RAYS: usize = 32;
const PROB_THRESHOLD: f64 = 0.479071463157368;
const NMS_THRESHOLD: f32 = 0.3;

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
/// * `prob_threshold`: Optional object/polygon probability threshold. If
///   `None`, then `prob_threshold == 0.479071463157368`.
///
/// # Returns
///
/// * * `Array2<u16>`: The StarDist2D model label image.
pub fn predict<'a, T, A>(
    data: A,
    pmin: Option<f64>,
    pmax: Option<f64>,
    prob_threshold: Option<f64>,
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

    // percentile normalize the input data and reflect pad each axis to a size
    // that is divisiable by DIV
    let norm = percentile_normalize(&view, pmin, pmax, None, None)?;
    let norm = norm.mapv(|v| v as f32);
    let pad_config: Vec<usize> = view
        .shape()
        .iter()
        .map(|&v| axes::divisible_pad(v, DIV))
        .collect();
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0))?;
    let pad_shape = norm_pad.shape().to_vec();

    // initialize an instance of the StarDist network and reshape the data into
    // a 1D tensor
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();
    let tensor =
        Tensor::<Backend, 1>::from_floats(norm_pad.into_flat().as_slice().unwrap(), &device);

    // run StarDist network prediction, returns object probabilites and radial
    // distances
    let (prob, dist) = stardist_model.forward(tensor, (pad_shape[0] as i32, pad_shape[1] as i32));
    let prob: Vec<f32> = prob.into_data().into_vec().unwrap();
    let dist: Vec<f32> = dist.into_data().into_vec().unwrap();
    let res_row: usize = pad_shape[0] / 2;
    let res_col: usize = pad_shape[1] / 2;
    let prob_arr = Array2::from_shape_vec((res_row, res_col), prob)
        .expect("StarDist 2D object probabilites reshape failed.");
    let dist_arr = Array3::from_shape_vec((res_row, res_col, N_RAYS), dist)
        .expect("StarDist 2D radial distances reshape failed.");

    // ensure all values in ray distances are at least 1e-3, prevents negative
    // or zero distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    // create a valid object mask and clip the board by 2 pixels
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

    // filter probabilities and distances with valid indices
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
    let valid_poly_inds = nms::sparse_polygon_nms_2d(
        poly_dist.view(),
        poly_pos.view(),
        n_polys,
        N_RAYS,
        NMS_THRESHOLD,
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
