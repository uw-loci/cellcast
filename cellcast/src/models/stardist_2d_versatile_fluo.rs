use burn::backend::Wgpu;
use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::threshold::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array1, Array2, Array3, ArrayBase, AsArray, Ix2, Ix3, ViewRepr, s};

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

    // set optional parameters if needed
    let pmin = pmin.unwrap_or(1.0);
    let pmax = pmax.unwrap_or(99.8);

    // percentile normalize the input data and ensure that each axis of the
    // input image into the network is divisiable by DIV factor, if not reflect
    // pad the image to the computed dimensions
    let norm = percentile_normalize(&view, pmin, pmax, None, None);
    let norm = norm.mapv(|v| v as f32);
    let pad_config: Vec<usize> = view
        .shape()
        .iter()
        .map(|&v| axes::divisible_pad(v, DIV))
        .collect();
    let norm_pad = reflect_pad(&norm, &pad_config, Some(0)).unwrap();
    let pad_shape = norm_pad.shape().to_vec();

    // create a 1-D tensor, the stardist network reshapes the 1D intput
    // TODO: move or avoid the need for 1D flattening for speed up
    // initialize an instance of the StarDist network
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();
    let tensor =
        Tensor::<Backend, 1>::from_floats(norm_pad.into_flat().as_slice().unwrap(), &device);

    // predict object probabilites and radial distances
    let (prob, dist) = stardist_model.forward(tensor, (pad_shape[0] as i32, pad_shape[1] as i32));
    let prob: Vec<f32> = prob.into_data().into_vec().unwrap();
    let dist: Vec<f32> = dist.into_data().into_vec().unwrap();
    let row: usize = pad_shape[0] / 2;
    let col: usize = pad_shape[1] / 2;
    let prob_arr = Array2::from_shape_vec((row, col), prob)
        .expect("StarDist 2D object probabilites reshape failed.");
    let dist_arr = Array3::from_shape_vec((row, col, N_RAYS), dist)
        .expect("StarDist 2D radial distances reshape failed.");
    let dist_arr = dist_arr.into_dimensionality::<Ix3>().unwrap();

    // post-processing
    // ensure all values in ray distances are at least 1e-3, prevents negative
    // or zero distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    // TODO: implement the optimal NMS prob threshold functions, until then
    // this value is from StarDist2D for the "blobs.tif" sample data
    let valid_mask = manual_mask(&prob_arr, 0.479071463157368);
    let mut valid_mask = valid_mask.into_dimensionality::<Ix2>().unwrap();
    border::clip_mask_border(&mut valid_mask.view_mut().into_dyn(), 2);
    // TODO: consider parallelize this or stay sequential?
    // collect a Vec<(usize, usize)> as valid inds coords and only visit those
    // this will save lots of time even with bounds checking. We only need to visit
    // every pixel in the valid mask array once.
    let valid_indices: Vec<(usize, usize)> = valid_mask
        .indexed_iter()
        .filter(|&((_, _), &v)| v)
        .map(|((r, c), _)| (r, c))
        .collect();
    let valid_prob: Vec<f32> = valid_indices
        .iter()
        .map(|&(r, c)| prob_arr[[r, c]])
        .collect();
    let valid_prob = Array1::from(valid_prob);
    let mut valid_dist = Array2::<f32>::zeros((valid_indices.len(), N_RAYS));
    (0..N_RAYS).for_each(|n| {
        valid_indices.iter().enumerate().for_each(|(i, &(r, c))| {
            valid_dist[[i, n]] = dist_arr[[r, c, n]];
        });
    });

    (prob_arr, dist_arr)
}
