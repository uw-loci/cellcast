use std::f64::consts::PI;

use imgal::error::ImgalError;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::tetrahedron_volume;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, stack};

/// Create a golden spiral unit sphere. The unit sphere is used to determine
/// which direction a ray points.
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

/// Compute the axis-aligned bounding box of a polyhedron.
///
/// # Description
///
/// Computes the axis-aligned bounding box (*i.e.* `bbox`) of a polyhedron,
/// returning the voxel index coordinates.
///
/// # Arguments
///
/// * `distances`: The polyhedron distances.
/// * `center`: The center of the bounding box.
/// * `gs_vertices`: The "Golden Spiral" unit sphere vertices with shape
///   `(n_points, 3)`.
/// * `n_rays`: The number of ray angles.
///
/// # Returns
///
/// * `[usize; 6]`: The bounding box coordinates in
///   `[z_min, z_max, y_min, y_max, x_min, x_max]` order.
#[inline]
pub fn polyhedron_bbox(
    distances: ArrayView1<f32>,
    center: ArrayView1<usize>,
    gs_vertices: ArrayView2<f64>,
    n_rays: usize,
) -> [usize; 6] {
    let mut z1 = usize::MAX;
    let mut y1 = usize::MAX;
    let mut x1 = usize::MAX;
    let mut z2 = usize::MIN;
    let mut y2 = usize::MIN;
    let mut x2 = usize::MIN;
    (0..n_rays).for_each(|i| {
        let z = (center[0] as f32 + distances[i] * gs_vertices[[i, 0]] as f32).round() as usize;
        let y = (center[1] as f32 + distances[i] * gs_vertices[[i, 1]] as f32).round() as usize;
        let x = (center[2] as f32 + distances[i] * gs_vertices[[i, 2]] as f32).round() as usize;
        z1 = z1.min(z);
        y1 = y1.min(y);
        x1 = x1.min(x);
        z2 = z2.max(z);
        y2 = y2.max(y);
        x2 = x2.max(x);
    });
    [z1, z2, y1, y2, x1, x2]
}

/// Compute the volume of a polyhedron.
///
/// # Description
///
/// Computes the volume of a polyhedron by summing signed tetrahedra from the
/// origin, `[0, 0, 0]`.
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
