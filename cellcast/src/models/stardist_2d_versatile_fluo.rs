use burn::backend::Wgpu;
use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::threshold::manual_mask;
use imgal::traits::numeric::AsNumeric;
use imgal::transform::pad::reflect_pad;
use ndarray::{Array2, Array3, ArrayBase, AsArray, Ix2, ViewRepr};

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
/// * `data`: A 2-dimensional image.
///
/// # Returns
///
/// * `(Array2<f32>, Array2<f32>)`: A tuple containing the probability
///   distribution (probs) and ray distances (dists) arrays.
#[inline]
pub fn predict<'a, T, A>(data: A) -> (Array2<f32>, Array3<f32>)
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    // create a view of the data
    let view: ArrayBase<ViewRepr<&'a T>, Ix2> = data.into();

    // percentile normalize the input data
    // TODO: expose the percentile min and max arguments?
    // ensure that each axis of the input image into the network is divisiable by
    // DIV factor, if not reflect pad the image to the computed dimensions
    let norm = percentile_normalize(&view, 1.0, 99.8, None, None);
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

    // post-processing
    // ensure all values in ray distances are at least 1e-3, prevents negative
    // or zero distances
    let dist_arr = dist_arr.mapv(|v| v.max(1e-3));
    // TODO: implement the optimal NMS prob threshold functions, until then
    // this value is from StarDist2D for the "blobs.tif" sample data
    let mut valid_obj_mask = manual_mask(&prob_arr, 0.479071463157368);
    border::clip_mask_border(&mut valid_obj_mask.view_mut(), 2);
    // TODO: mask select prob_arr and dist_arr with valid_obj_mask
    // (prob_arr, dist_arr)
    (prob_arr, dist_arr)
}
