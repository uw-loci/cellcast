use burn::backend::Wgpu;
use burn::prelude::*;
use imgal::image::percentile_normalize;
use imgal::traits::numeric::AsNumeric;
use ndarray::{ArrayBase, Array2, Array3, AsArray, Ix2, ViewRepr};

use crate::networks::stardist::versatile_fluo_2d::Model;

type Backend = Wgpu<f32, i32>;

const N_RAYS: usize = 32;

/// Perform inference with the StarDist 2-dimensonal versatile fluo model.
#[inline]
pub fn predict<'a, T, A, D>(data: A) -> (Array2<f32>, Array3<f32>)
where
    A: AsArray<'a, T, Ix2>,
    T: 'a + AsNumeric,
{
    // create an array view
    let view: ArrayBase<ViewRepr<&'a T>, Ix2> = data.into();

    // setup the model for
    let device = Default::default();
    let stardist_model = Model::<Backend>::default();

    // percentile normalize the input data
    // TODO: expose the percentile min and max arguments?
    // TODO: maybe add a boolean determine if normalization will happen or not
    let norm_data = percentile_normalize(&view, 1.0, 99.8, None, None);
    let norm_data = norm_data.mapv(|v| v as f32);

    // create a 1-D tensor, the stardist network reshapes the 1D intput
    // TODO: move or avoid the need for 1D flattening for speed up
    let tensor = Tensor::<Backend, 1>::from_floats(
        norm_data.into_flat().as_slice().unwrap(),
        &device,
    );

    // predict object probabilites and radial distances
    let (row, col) = view.dim();
    let (prob, dist) = stardist_model.forward(tensor, (row as i32, col as i32));
    let prob: Vec<f32> = prob.into_data().into_vec().unwrap();
    let dist: Vec<f32> = dist.into_data().into_vec().unwrap();
    let row: usize = row / 2;
    let col: usize = col / 2;
    let prob_arr = Array2::from_shape_vec((row, col), prob).expect("StarDist 2D object probabilites reshape failed");
    let dist_arr = Array3::from_shape_vec((row, col, N_RAYS), dist).expect("StarDist 2D radial distances reshape failed");

    (prob_arr, dist_arr)
}
