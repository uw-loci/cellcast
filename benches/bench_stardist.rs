use criterion::{Criterion, criterion_group, criterion_main};
use imgal::simulation::blob::logistic_metaballs;
use ndarray::{Ix2, Ix3, arr2};

use cellcast::models::{StarDist2D, StarDist3D};

const SHAPE_2D: [usize; 2] = [128, 128];
const SHAPE_3D: [usize; 3] = [64, 128, 128];
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
    let mut group = c.benchmark_group("StarDist2D");
    let sd = StarDist2D::init_fluo(None, GPU);
    sd.warm_up_fluo();
    group.bench_function("predict_fluo", |b| {
        b.iter(|| {
            let _ = sd.predict_fluo(&data, None, None, None, None).unwrap();
        });
    });
    group.finish();
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
    let mut group = c.benchmark_group("StarDist3D");
    group.sample_size(10);
    let sd = StarDist3D::init_fluo(None, None, GPU).unwrap();
    sd.warm_up_fluo();
    group.bench_function("predict_fluo", |b| {
        b.iter(|| {
            let _ = sd
                .predict_fluo(&data, None, None, None, None, None)
                .unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_stardist_2d, bench_stardist_3d);
criterion_main!(benches);
