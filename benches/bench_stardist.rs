use cellcast::models::stardist_2d::{predict_versatile_fluo, warm_up_versatile_fluo};
use cellcast::models::stardist_3d::{predict_demo, warm_up_demo};
use criterion::{Criterion, criterion_group, criterion_main};
use imgal::simulation::blob::logistic_metaballs;
use ndarray::{Ix2, Ix3, arr2};

const SHAPE_2D: [usize; 2] = [64, 64];
const SHAPE_3D: [usize; 3] = [32, 64, 64];
const GPU: bool = true;

fn bench_stardist_2d(c: &mut Criterion) {
    let centers = arr2(&[
        [(SHAPE_3D[1] / 4) as f64, SHAPE_3D[2] as f64 / 1.5],
        [SHAPE_3D[1] as f64 / 1.2, SHAPE_3D[2] as f64 / 1.8],
        [(SHAPE_3D[1] / 10) as f64, (SHAPE_3D[2] / 12) as f64],
    ]);
    let radii = [3.0, 5.0, 7.0];
    let intensities = [10.0; 3];
    let falloffs = [2.0; 3];
    let data = logistic_metaballs(
        &centers,
        &radii,
        &intensities,
        &falloffs,
        0.0,
        &SHAPE_2D,
        None,
    )
    .unwrap();
    let data = data.into_dimensionality::<Ix2>().unwrap();
    warm_up_versatile_fluo(GPU);
    c.bench_function("stardist_2d", |b| {
        b.iter(|| {
            let _ = predict_versatile_fluo(&data, None, None, None, None, GPU).unwrap();
        });
    });
}

fn bench_stardist_3d(c: &mut Criterion) {
    let centers = arr2(&[
        [
            (SHAPE_3D[0] / 2) as f64,
            (SHAPE_3D[1] / 4) as f64,
            SHAPE_3D[2] as f64 / 1.5,
        ],
        [
            (SHAPE_3D[0] / 2) as f64,
            SHAPE_3D[1] as f64 / 1.2,
            SHAPE_3D[2] as f64 / 1.8,
        ],
        [
            (SHAPE_3D[0] / 2) as f64,
            (SHAPE_3D[1] / 10) as f64,
            (SHAPE_3D[2] / 12) as f64,
        ],
    ]);
    let radii = [3.0, 5.0, 7.0];
    let intensities = [10.0; 3];
    let falloffs = [2.0; 3];
    let data = logistic_metaballs(
        &centers,
        &radii,
        &intensities,
        &falloffs,
        0.0,
        &SHAPE_3D,
        None,
    )
    .unwrap();
    let data = data.into_dimensionality::<Ix3>().unwrap();
    warm_up_demo(GPU);
    c.bench_function("stardist_3d", |b| {
        b.iter(|| {
            let _ = predict_demo(&data, None, None, None, None, None, GPU).unwrap();
        });
    });
}

criterion_group!(benches, bench_stardist_2d, bench_stardist_3d);
criterion_main!(benches);
