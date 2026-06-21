use imgal::prelude::*;
use imgal::simulation::blob::logistic_metaballs;
use imgal::spatial::roi::roi_cloud_map;
use ndarray::{Ix2, Ix3, arr2};

use cellcast::models::stardist_2d::predict_versatile_fluo;
use cellcast::models::stardist_3d::predict_demo;

const CENTERS_2D: [[f64; 2]; 20] = [
    [45.0, 57.5],
    [40.25, 74.25],
    [52.75, 65.25],
    [42.25, 103.25],
    [66.75, 57.25],
    [87.25, 101.25],
    [98.75, 40.75],
    [54.25, 13.25],
    [27.25, 52.25],
    [112.25, 109.25],
    [63.75, 94.25],
    [17.75, 27.25],
    [21.25, 113.75],
    [75.25, 37.25],
    [64.4, 125.0],
    [105.35, 10.4],
    [2.75, 100.0],
    [125.0, 60.0],
    [87.5, 87.5],
    [30.0, 70.0],
];
const CENTERS_3D: [[f64; 3]; 9] = [
    [4.0, 32.0, 32.0],
    [4.0, 16.25, 50.25],
    [4.0, 52.75, 12.5],
    [4.0, 8.0, 12.5],
    [4.0, 31.75, 5.25],
    [4.0, 57.25, 43.25],
    [4.0, 7.75, 31.75],
    [4.0, 41.25, 54.25],
    [4.0, 55.25, 27.25],
];
const RADII_2D: [f64; 20] = [
    2.0, 3.3, 4.0, 5.0, 11.0, 7.5, 9.0, 6.0, 5.5, 9.0, 13.0, 9.5, 6.0, 3.5, 9.5, 6.0, 2.5, 7.5,
    4.5, 7.0,
];
const RADII_3D: [f64; 9] = [9.0, 7.0, 4.0, 4.5, 3.0, 2.0, 3.0, 6.0, 2.5];
const INTENSITIES_2D: [f64; 20] = [10.0; 20];
const INTENSITIES_3D: [f64; 9] = [10.0; 9];
const FALLOFFS_2D: [f64; 20] = [2.0; 20];
const FALLOFFS_3D: [f64; 9] = [2.0; 9];
const BACKGROUND: f64 = 0.0;
const SHAPE_2D: [usize; 2] = [128, 128];
const SHAPE_3D: [usize; 3] = [8, 64, 64];

/// Tests that `predict_versatile_fluo` returns the expected results for a
/// simulated dataset of 20 blobs. This test asserts the number of blobs found
/// and their size.
#[test]
fn stardist_2d_predict_versatile_fluo_expected_results() -> Result<(), ImgalError> {
    let data = logistic_metaballs(
        &arr2(&CENTERS_2D),
        &RADII_2D,
        &INTENSITIES_2D,
        &FALLOFFS_2D,
        BACKGROUND,
        &SHAPE_2D,
        None,
    )?;
    let data = data.into_dimensionality::<Ix2>().unwrap();
    let labels = predict_versatile_fluo(&data, None, None, None, None, false)?;
    let rcm = roi_cloud_map(&labels, None);
    assert_eq!(rcm.len(), 20);
    assert_eq!(rcm.get(&1).expect("ROI 1 not foud.").dim().0, 244);
    assert_eq!(rcm.get(&2).expect("ROI 2 not foud.").dim().0, 109);
    assert_eq!(rcm.get(&3).expect("ROI 3 not foud.").dim().0, 359);
    assert_eq!(rcm.get(&4).expect("ROI 4 not foud.").dim().0, 62);
    assert_eq!(rcm.get(&5).expect("ROI 5 not foud.").dim().0, 79);
    assert_eq!(rcm.get(&6).expect("ROI 6 not foud.").dim().0, 376);
    assert_eq!(rcm.get(&7).expect("ROI 7 not foud.").dim().0, 141);
    assert_eq!(rcm.get(&8).expect("ROI 8 not foud.").dim().0, 167);
    assert_eq!(rcm.get(&9).expect("ROI 9 not foud.").dim().0, 224);
    assert_eq!(rcm.get(&10).expect("ROI 10 not foud.").dim().0, 609);
    assert_eq!(rcm.get(&11).expect("ROI 11 not foud.").dim().0, 359);
    assert_eq!(rcm.get(&12).expect("ROI 12 not foud.").dim().0, 104);
    assert_eq!(rcm.get(&13).expect("ROI 13 not foud.").dim().0, 85);
    assert_eq!(rcm.get(&14).expect("ROI 14 not foud.").dim().0, 96);
    assert_eq!(rcm.get(&15).expect("ROI 15 not foud.").dim().0, 215);
    assert_eq!(rcm.get(&16).expect("ROI 16 not foud.").dim().0, 171);
    assert_eq!(rcm.get(&17).expect("ROI 17 not foud.").dim().0, 451);
    assert_eq!(rcm.get(&18).expect("ROI 18 not foud.").dim().0, 248);
    assert_eq!(rcm.get(&19).expect("ROI 19 not foud.").dim().0, 203);
    assert_eq!(rcm.get(&20).expect("ROI 20 not foud.").dim().0, 240);
    Ok(())
}

/// Tests that `predict_demo` returns the expected results for a simulated
/// dataset of 9 blobs in 3D. This test asserts the number of blobs found and
/// their size.
#[test]
fn stardist_3d_predict_demo_expected_results() -> Result<(), ImgalError> {
    let data = logistic_metaballs(
        &arr2(&CENTERS_3D),
        &RADII_3D,
        &INTENSITIES_3D,
        &FALLOFFS_3D,
        BACKGROUND,
        &SHAPE_3D,
        None,
    )?;
    let data = data.into_dimensionality::<Ix3>().unwrap();
    let labels = predict_demo(&data, None, None, None, None, None, false)?;
    let rcm = roi_cloud_map(&labels, None);
    assert_eq!(rcm.len(), 9);
    assert_eq!(rcm.get(&1).expect("ROI 1 not found.").dim().0, 1287);
    assert_eq!(rcm.get(&2).expect("ROI 2 not found.").dim().0, 1528);
    assert_eq!(rcm.get(&3).expect("ROI 3 not found.").dim().0, 2364);
    assert_eq!(rcm.get(&4).expect("ROI 4 not found.").dim().0, 770);
    assert_eq!(rcm.get(&5).expect("ROI 5 not found.").dim().0, 483);
    assert_eq!(rcm.get(&6).expect("ROI 6 not found.").dim().0, 442);
    assert_eq!(rcm.get(&7).expect("ROI 7 not found.").dim().0, 427);
    assert_eq!(rcm.get(&8).expect("ROI 8 not found.").dim().0, 708);
    assert_eq!(rcm.get(&9).expect("ROI 9 not found.").dim().0, 304);
    Ok(())
}
