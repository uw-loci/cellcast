use ndarray::{Ix2, arr2};

use cellcast::models::stardist_2d::predict_versatile_fluo;
use imgal::error::ImgalError;
use imgal::simulation::blob::logistic_metaballs;
use imgal::simulation::noise::poisson_noise_mut;
use imgal::spatial::roi::roi_cloud_map;

const CENTERS: [[f64; 2]; 5] = [
    [55.5, 60.0],
    [120.0, 45.0],
    [150.0, 150.0],
    [110.0, 220.5],
    [200.5, 112.0],
];
const RADII: [f64; 5] = [33.0, 37.2, 40.8, 38.5, 29.7];
const INTENSITIES: [f64; 5] = [20.0, 22.3, 21.8, 19.3, 24.1];
const FALLOFFS: [f64; 5] = [3.5; 5];
const BACKGROUND: f64 = 0.0;
const SHAPE: [usize; 2] = [256, 256];

/// Tests that `predict_versatile_fluo` returns the expected results for a
/// simulated dataset of 5 blobs with Poisson noise. This test asserts the
/// number of blobs found and their size.
#[test]
fn stardist_2d_predict_versatile_fluo_expected_results() -> Result<(), ImgalError> {
    let mut data = logistic_metaballs(
        &arr2(&CENTERS),
        &RADII,
        &INTENSITIES,
        &FALLOFFS,
        BACKGROUND,
        &SHAPE,
    )?;
    poisson_noise_mut(data.view_mut(), 0.8, None, false);
    let data = data.into_dimensionality::<Ix2>().unwrap();
    let labels = predict_versatile_fluo(&data, None, None, None, None, false)?;
    let rcm = roi_cloud_map(&labels, false);
    assert_eq!(rcm.len(), 5);
    assert_eq!(rcm.get(&1).expect("ROI 1 not foud.").dim().0, 3177);
    assert_eq!(rcm.get(&2).expect("ROI 2 not foud.").dim().0, 5466);
    assert_eq!(rcm.get(&3).expect("ROI 3 not foud.").dim().0, 3828);
    assert_eq!(rcm.get(&4).expect("ROI 4 not foud.").dim().0, 5003);
    assert_eq!(rcm.get(&5).expect("ROI 5 not foud.").dim().0, 4794);
    Ok(())
}
