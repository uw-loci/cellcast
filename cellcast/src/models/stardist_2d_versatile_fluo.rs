use burn::backend::Wgpu;
use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::threshold::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array2, Array3, ArrayBase, AsArray, Axis, Ix2, ViewRepr};

use crate::networks::stardist::versatile_fluo_2d::Model;
use crate::utils::{axes, border};

type Backend = Wgpu<f32, i32>;

const N_RAYS: usize = 32;
const DIV: usize = 16;

/// Perform inference with the StarDist 2-dimensional versatile fluo model.
///
/// # Description
///
/// Predict instance segmentations with the StarDist2D versatile fluo model.
///
/// # Arguments
///
/// * `data`: The input 2-dimensional image.
/// * `pmin`: The minimum percentage to normalize the input image.
/// * `pmax`: The maximum percentage to normalize the input image.
///
/// # Returns
///
/// * `(Array2<f32>, Array2<f32>)`: A tuple containing the probability
///   distribution (probs) and ray distances (dists) arrays.
#[inline]
pub fn predict<'a, T, A>(
    data: A,
    pmin: Option<f64>,
    pmax: Option<f64>,
) -> (Array2<f32>, Array3<f32>)
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, Ix2> = data.into();
    let view_shape = view.shape();

    // set optional parameters if needed
    let pmin = pmin.unwrap_or(1.0);
    let pmax = pmax.unwrap_or(99.8);

    // percentile normalize the input data and reflect pad each axis to a size
    // that is divisiable by DIV
    let norm = percentile_normalize(&view, pmin, pmax, None, None);
    let norm = norm.mapv(|v| v as f32);
    let pad_config: Vec<usize> = view
        .shape()
        .iter()
        .map(|&v| axes::divisible_pad(v, DIV))
        .collect();
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0)).unwrap();
    let pad_shape = norm_pad.shape().to_vec();

    // initialize an instance of the StarDist network and reshape the data into
    // a 1D tensor
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();
    let tensor =
        Tensor::<Backend, 1>::from_floats(norm_pad.into_flat().as_slice().unwrap(), &device);

    // run StarDist network prediction, returns object probabilites and radial distances
    let (prob, dist) = stardist_model.forward(tensor, (pad_shape[0] as i32, pad_shape[1] as i32));
    let prob: Vec<f32> = prob.into_data().into_vec().unwrap();
    let dist: Vec<f32> = dist.into_data().into_vec().unwrap();
    let row: usize = pad_shape[0] / 2;
    let col: usize = pad_shape[1] / 2;
    let prob_arr = Array2::from_shape_vec((row, col), prob)
        .expect("StarDist 2D object probabilites reshape failed.");
    let dist_arr = Array3::from_shape_vec((row, col, N_RAYS), dist)
        .expect("StarDist 2D radial distances reshape failed.");

    // === post-processing ===
    // ensure all values in ray distances are at least 1e-3, prevents negative
    // or zero distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    // TODO: implement the optimal NMS prob threshold functions, until then use
    // this value is from StarDist2D for the "blobs.tif" sample data
    // create a valid object mask and clip the board by 2 pixels
    let mut valid_mask = manual_mask(&prob_arr, 0.479071463157368);
    border::clip_mask_border(&mut valid_mask.view_mut(), 2);
    let valid_mask = valid_mask.into_dimensionality::<Ix2>().unwrap();
    // collect a Vec<(usize, usize)> as valid indices coords and only visit those
    // this will save lots of time even with bounds checking. We only need to visit
    // every pixel in the valid mask array once.
    // collect all valid (row, col) positions to avoid iterating the mask repeatedly
    let valid_pos: Vec<(usize, usize)> = valid_mask
        .indexed_iter()
        .filter(|&((_, _), &v)| v)
        .map(|((r, c), _)| (r, c))
        .collect();
    // filter probabilities and distances with valid indices
    let mut valid_prob: Vec<f32> = valid_pos.iter().map(|&(r, c)| prob_arr[[r, c]]).collect();
    let mut valid_dist = Array2::<f32>::zeros((valid_pos.len(), N_RAYS));
    (0..N_RAYS).for_each(|n| {
        valid_pos.iter().enumerate().for_each(|(i, &(r, c))| {
            valid_dist[[i, n]] = dist_arr[[r, c, n]];
        });
    });
    // scale each valid position by 2 and collect the indices of positions outside
    // of the original image dimensions (used for point filtering)
    let mut valid_pos: Vec<(usize, usize)> =
        valid_pos.iter().map(|&(r, c)| (r * 2, c * 2)).collect();
    let invalid_inds: Vec<usize> = valid_pos
        .iter()
        .filter(|&&(r, c)| r > view_shape[0] || c > view_shape[1])
        .enumerate()
        .map(|(i, &(_, _))| i)
        .collect();
    // filter positions, probabilities, and distances if there are invalid indices
    if !invalid_inds.is_empty() {
        valid_pos = valid_pos
            .iter()
            .enumerate()
            .filter(|&(i, &(_, _))| !invalid_inds.contains(&i))
            .map(|(_, &(r, c))| (r, c))
            .collect();
        valid_prob = valid_prob
            .iter()
            .enumerate()
            .filter(|&(i, &_)| !invalid_inds.contains(&i))
            .map(|(_, &v)| v)
            .collect();
        let valid_inds: Vec<usize> = valid_pos.iter().enumerate().map(|(i, &(_, _))| i).collect();
        valid_dist = valid_dist.select(Axis(0), &valid_inds);
    }

    (prob_arr, dist_arr)
}
