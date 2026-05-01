use std::f64::consts::PI;

use imgal::error::ImgalError;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::tetrahedron_volume;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, stack};

/// Estimate the average anisotropy of a slice of polyhedra bounding boxes.
///
/// # Arguments
///
/// * `bboxes`: The slice of bounding boxes.
/// * `n_polys`: The number of polyhedra.
///
/// # Returns
///
/// * `[f32; 3]`: The estimated average anisotropy.
#[inline]
pub fn estimate_anisotropy(bboxes: &[[usize; 6]], n_polys: usize) -> [f32; 3] {
    let eps = 1e-10;
    let avg_aniso: [f32; 3] = (0..n_polys).fold([0.0_f32; 3], |mut acc, i| {
        let n = n_polys as f32;
        acc[0] += (bboxes[i][1] - bboxes[i][0]) as f32 / n;
        acc[1] += (bboxes[i][3] - bboxes[i][2]) as f32 / n;
        acc[2] += (bboxes[i][5] - bboxes[i][4]) as f32 / n;
        acc
    });
    let tmp = avg_aniso[0].max(avg_aniso[1]).max(avg_aniso[2]);
    [
        tmp / avg_aniso[0].max(eps),
        tmp / avg_aniso[1].max(eps),
        tmp / avg_aniso[2].max(eps),
    ]
}

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

/// Get the inner bounding radius (*i.e* the shortest perpendicular distance
/// to any face plane). Note that this radius represents the largest sphere that
/// can fit inside the polyhedron.
#[inline]
pub fn bounding_inner_radius(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> f32 {
    let eps = 1e-10;
    let n_faces = gs_faces.dim().0;
    (0..n_faces).fold(0.0_f32, |acc, f| {
        let tri = gs_faces.row(f);
        let a: [f32; 3] = {
            let i = tri[0];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        let b: [f32; 3] = {
            let i = tri[1];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        let c: [f32; 3] = {
            let i = tri[2];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        // compute the edge vectors and cross product
        let (baz, bay, bax) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let (caz, cay, cax) = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let nz = bax * cay - bay * cax;
        let ny = baz * cax - bax * caz;
        let nx = bay * caz - baz * cay;
        let norm = 1.0 / (nz * nz + ny * ny + nx * nx).sqrt().max(eps);
        let (nz, ny, nx) = (nz * norm, ny * norm, nx * norm);
        let dist = a[0] * nz + a[1] * ny + a[2] * nx;
        acc.min(dist)
    })
}

/// TODO
#[inline]
pub fn bounding_inner_radius_iso(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
    anisotropy: [f32; 3],
) -> f32 {
    let eps = 1e-10;
    (0..gs_faces.dim().0).fold(0.0_f32, |acc, i| {
        let i_a = gs_faces[[i, 0]];
        let i_b = gs_faces[[i, 1]];
        let i_c = gs_faces[[i, 2]];
        let a = {
            let dist = distances[i_a];
            [
                anisotropy[0] * dist * gs_vertices[[i_a, 0]] as f32,
                anisotropy[1] * dist * gs_vertices[[i_a, 1]] as f32,
                anisotropy[2] * dist * gs_vertices[[i_a, 2]] as f32,
            ]
        };
        let b = {
            let dist = distances[i_b];
            [
                anisotropy[0] * dist * gs_vertices[[i_b, 0]] as f32,
                anisotropy[1] * dist * gs_vertices[[i_b, 1]] as f32,
                anisotropy[2] * dist * gs_vertices[[i_b, 2]] as f32,
            ]
        };
        let c = {
            let dist = distances[i_c];
            [
                anisotropy[0] * dist * gs_vertices[[i_c, 0]] as f32,
                anisotropy[1] * dist * gs_vertices[[i_c, 1]] as f32,
                anisotropy[2] * dist * gs_vertices[[i_c, 2]] as f32,
            ]
        };
        // compute the edge vectors and cross product
        let (baz, bay, bax) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let (caz, cay, cax) = (c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let nz = bax * cay - bay * cax;
        let ny = baz * cax - bax * caz;
        let nx = bay * caz - baz * cay;
        let norm = 1.0 / (nz * nz + ny * ny + nx * nx).sqrt().max(eps);
        let (nz, ny, nx) = (nz * norm, ny * norm, nx * norm);
        let dist = a[0] * nz + a[1] * ny + a[2] * nx;
        acc.min(dist)
    })
}

/// Get the outer bounding radius (*i.e.* the maximum distance ray). Note that
/// the polyhedron fits inside a sphere of this radius.
#[inline]
pub fn bounding_outer_radius(distances: ArrayView1<f32>) -> f32 {
    distances.iter().fold(0.0_f32, |acc, v| v.max(acc))
}

/// TODO
#[inline]
pub fn bounding_outer_radius_iso(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    anisotropy: [f32; 3],
) -> f32 {
    let radius = (0..distances.len()).fold(0.0_f32, |acc, i| {
        let dist = distances[i];
        let z = anisotropy[0] * dist * gs_vertices[[i, 0]] as f32;
        let y = anisotropy[1] * dist * gs_vertices[[i, 1]] as f32;
        let x = anisotropy[2] * dist * gs_vertices[[i, 2]] as f32;
        acc.max(z * z + y * y + x * x)
    });
    radius.sqrt()
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
) -> [usize; 6] {
    let mut z1 = usize::MAX;
    let mut y1 = usize::MAX;
    let mut x1 = usize::MAX;
    let mut z2 = usize::MIN;
    let mut y2 = usize::MIN;
    let mut x2 = usize::MIN;
    (0..distances.len()).for_each(|i| {
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
#[inline]
pub fn polyhedron_volume(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> f32 {
    let origin = [0.0_f32; 3];
    let n_faces = gs_faces.dim().0;
    (0..n_faces).fold(0.0_f32, |acc, f| {
        let tri = gs_faces.row(f);
        let a: [f32; 3] = {
            let i = tri[0];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        let b: [f32; 3] = {
            let i = tri[1];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        let c: [f32; 3] = {
            let i = tri[2];
            let di = distances[i];
            [
                di * gs_vertices[[i, 0]] as f32,
                di * gs_vertices[[i, 1]] as f32,
                di * gs_vertices[[i, 2]] as f32,
            ]
        };
        let v = tetrahedron_volume(&a, &b, &c, &origin) as f32;
        acc + v
    })
}
