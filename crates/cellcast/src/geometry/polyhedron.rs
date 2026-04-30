use std::f64::consts::PI;

use imgal::error::ImgalError;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::tetrahedron_volume;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, stack};

/// Create a golden spiral unit sphere. The unit sphere is used to determine
/// which directions a ray points.
///
/// # Arguments
///
/// * `n_points`: The number of points (*i.e.* rays) for the golden spiral
///   sphere.
/// * `anisotropy`: The 1D anisotropy array. If `None` then
///   `anisotropy = [1.0_f64; 3]`.
///
/// # Returns
///
/// * `Ok((Array2<f64>, Array2<usize>))`: The golden spiral 3D convex hull
///   vertices and triangular face indices.
#[inline]
pub fn golden_spiral(
    n_points: usize,
    anisotropy: Option<[f64; 3]>,
) -> Result<(Array2<f64>, Array2<usize>), ImgalError> {
    let anisotropy = Array1::from_iter(anisotropy.unwrap_or([1.0_f64; 3]));
    let golden_angle = (3.0 - 5.0_f64.sqrt()) * PI;
    let phi = Array1::from_iter(0..n_points).mapv(|v| v as f64 * golden_angle);
    let z = Array1::linspace(-1.0, 1.0, n_points);
    let rho = z.mapv(|v| (1.0_f64 - v * v).sqrt());
    let a = &rho * phi.mapv(|v| v.sin());
    let b = &rho * phi.mapv(|v| v.cos());
    let ax = Axis(1);
    let points = stack(ax, &[z.view(), a.view(), b.view()])
        .expect("Failed to create Golden spiral point cloud.");
    let points = points / anisotropy;
    let (mut verts, faces) = quickhull_3d(&points, false)?;
    let norms = verts.map_axis(ax, |r| r.dot(&r).sqrt());
    verts /= &norms.insert_axis(ax);
    Ok((verts, faces))
}

/// Compute the volume of a polyhedron.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `distances`: The polyhedron distances.
/// * `gs_vertices`: The "Golden Spiral" unit sphere vertices with shape
///   `(n_points, 3)`.
/// * `gs_faces`: The "Golden Spiral" unit sphere face indices with shape
///   `(n_triangles, 3)`.
///
/// # Returns
///
/// * `f32`: The volume of the polyhedron.
pub fn polyhedron_volume(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> f32 {
    let origin = vec![0.0_f32; 3];
    let n_faces = gs_faces.dim().0;
    (0..n_faces).fold(0.0_f32, |acc, f| {
        let tri = gs_faces.row(f);
        let a: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[0];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let b: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[1];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let c: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[2];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let v = tetrahedron_volume(&a, &b, &c, &origin) as f32;
        acc + v
    })
}
