use std::array;
use std::f64::consts::PI;

use imgal::prelude::*;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::{inside_polyhedron, tetrahedron_volume};
use imgal::spatial::halfspace::{face_to_halfspace, halfspace_intersection, hull_to_halfspace};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, arr1, concatenate, stack};

/// Compute the intersection volume of two axis-aligned 3D bounding boxes.
///
/// # Arguments
///
/// * `bbox_a`: The coordinates of bounding box `a`.
/// * `bbox_b`: The coordinates of bounding box `b`.
///
/// # Returns
///
/// * `f32`: The intersection volume of bounding box `a` and `b`.
#[inline]
pub fn bbox_intersect_vol(bbox_a: &[i32; 6], bbox_b: &[i32; 6]) -> f32 {
    let wz = (bbox_a[1].min(bbox_b[1]) - bbox_a[0].max(bbox_b[0])).max(0) as f32;
    let wy = (bbox_a[3].min(bbox_b[3]) - bbox_a[2].max(bbox_b[2])).max(0) as f32;
    let wx = (bbox_a[5].min(bbox_b[5]) - bbox_a[4].max(bbox_b[4])).max(0) as f32;
    wz * wy * wx
}

/// Get the inner bounding radius (*i.e* the shortest perpendicular distance
/// to any face plane). Note that this radius represents the largest sphere that
/// can fit inside the polyhedron.
#[inline]
pub fn bounding_inner_radius(
    // DEBUG: this function doesn't seem to be used. Its slated for removal
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> f32 {
    let eps = 1e-10;
    let n_faces = gs_faces.dim().0;
    (0..n_faces).fold(f32::MAX, |acc, f| {
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
    (0..gs_faces.dim().0).fold(f32::MAX, |acc, i| {
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

/// TODO
#[inline]
pub fn convex_hull_intersection_vol(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<f32>,
    center_b: ArrayView1<f32>,
) -> ImgalResult<f64> {
    let (hull_verts_a, hull_faces_a) = quickhull_3d(vertices_a, false)?;
    let (hull_verts_b, hull_faces_b) = quickhull_3d(vertices_b, false)?;
    let hs_a = hull_to_halfspace(&hull_verts_a, &hull_faces_a, false)?;
    let hs_b = hull_to_halfspace(&hull_verts_b, &hull_faces_b, false)?;
    let hs = concatenate(Axis(0), &[hs_a.view(), hs_b.view()])
        .expect("Failed to stack halfspaces into array.");
    let in_pnt: [f64; 3] = array::from_fn(|i| 0.5 * (center_a[i] + center_b[i]) as f64);
    let (inter_verts, inter_faces) = halfspace_intersection(&hs, &in_pnt, false)?;
    let n_if = inter_faces.dim().0;
    Ok((0..n_if).fold(0.0_f64, |acc, i| {
        let [a_idx, b_idx, c_idx] = array::from_fn(|j| inter_faces[[i, j]]);
        let [az, ay, ax] = array::from_fn(|j| inter_verts[[a_idx, j]] - in_pnt[j]);
        let [bz, by, bx] = array::from_fn(|j| inter_verts[[b_idx, j]] - in_pnt[j]);
        let [cz, cy, cx] = array::from_fn(|j| inter_verts[[c_idx, j]] - in_pnt[j]);
        let cross_z = bx * cy - by * cx;
        let cross_y = bz * cx - bx * cz;
        let cross_x = by * cz - bz * cy;
        let temp = az * cross_z + ay * cross_y + ax * cross_x;
        acc + (temp / 6.0).abs()
    }))
}

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
pub fn estimate_anisotropy(bboxes: &[[i32; 6]], n_polys: usize) -> [f32; 3] {
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
) -> ImgalResult<(Array2<f64>, Array2<usize>)> {
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

/// Computes the overlap volume of two convex hulls by intersecting their face
/// halfspaces and summing the volume of the result intersection hull. This
/// function assumes the Golden Spiral faces describe the polyhedrons `a` and
/// `b`.
///
/// # Arguments
///
/// * `vertices_a`: Vertices of polyhedron `a`.
/// * `vertices_b`: Vertices of polyhedron `b`.
/// * `center_a`: The center point of polyhedron `a`.
/// * `center_b`: The center point of polyhedron `b`.
/// * `gs_faces`: The "Golden Spiral" unit sphere face indices with shape
///   `(n_triangles, 3)`.
///
/// # Retruns
///
/// * `Ok(f64)`: The intersection volume of polyhedron `a` and `b`.
/// * `Err(ImgalError)`: If intersection halfspaces is `< 4`. If the halfspace
///   intersection interior point is not 3D.
#[inline]
pub fn golden_spiral_intersection_vol(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<f32>,
    center_b: ArrayView1<f32>,
    gs_faces: ArrayView2<usize>,
) -> ImgalResult<f64> {
    let n_gsf = gs_faces.dim().0;
    let hs: Vec<Array1<f64>> =
        (0..n_gsf).try_fold(Vec::with_capacity(n_gsf * 2), |mut acc, i| {
            let [a_idx, b_idx, c_idx] = array::from_fn(|j| gs_faces[[i, j]]);
            acc.push(face_to_halfspace(
                vertices_a.row(a_idx),
                vertices_a.row(b_idx),
                vertices_a.row(c_idx),
            )?);
            acc.push(face_to_halfspace(
                vertices_b.row(a_idx),
                vertices_b.row(b_idx),
                vertices_b.row(c_idx),
            )?);
            Ok(acc)
        })?;
    let in_pnt: [f64; 3] = array::from_fn(|i| 0.5 * (center_a[i] + center_b[i]) as f64);
    let hs = stack(
        Axis(0),
        &hs.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .expect("Failed to stack halfspaces into array.");
    let (inter_verts, inter_faces) = halfspace_intersection(&hs, &in_pnt, false)?;
    let n_if = inter_faces.dim().0;
    Ok((0..n_if).fold(0.0_f64, |acc, i| {
        let [a_idx, b_idx, c_idx] = array::from_fn(|j| inter_faces[[i, j]]);
        let [az, ay, ax] = array::from_fn(|j| inter_verts[[a_idx, j]] - in_pnt[j]);
        let [bz, by, bx] = array::from_fn(|j| inter_verts[[b_idx, j]] - in_pnt[j]);
        let [cz, cy, cx] = array::from_fn(|j| inter_verts[[c_idx, j]] - in_pnt[j]);
        let cross_z = bx * cy - by * cx;
        let cross_y = bz * cx - bx * cz;
        let cross_x = by * cz - bz * cy;
        let temp = az * cross_z + ay * cross_y + ax * cross_x;
        acc + (temp / 6.0).abs()
    }))
}

/// TODO
#[inline]
pub fn overlap_polyhedron_mask(
    vertices: ArrayView2<f32>,
    faces: ArrayView2<usize>,
    center: ArrayView1<f32>,
    mask: &[bool],
    bbox: [i32; 6],
    nz: usize,
    ny: usize,
    nx: usize,
    overlap_threshold: f32,
) -> ImgalResult<i32> {
    let mut count = 0;
    let nx_ny = nx * ny;
    for z in 0..nz {
        let z_nx_ny = z * nx_ny;
        let qz = (z as i32 + bbox[0]) as f32;
        for y in 0..ny {
            let y_nx = y * nx;
            let qy = (y as i32 + bbox[2]) as f32;
            for x in 0..nx {
                let idx = x + y_nx + z_nx_ny;
                if !mask[idx] {
                    continue;
                }
                let qx = (x as i32 + bbox[4]) as f32;
                let query = arr1(&[qz, qy, qx]);
                if inside_polyhedron(vertices, faces, center.view(), query.view(), false)? {
                    count += 1;
                    if (count as f32) > overlap_threshold {
                        return Ok(count);
                    }
                }
            }
        }
    }
    Ok(count)
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
/// * `[i32; 6]`: The bounding box coordinates in
///   `[z_min, z_max, y_min, y_max, x_min, x_max]` order.
#[inline]
pub fn polyhedron_bbox(
    distances: ArrayView1<f32>,
    center: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
) -> [i32; 6] {
    let mut z1 = i32::MAX;
    let mut y1 = i32::MAX;
    let mut x1 = i32::MAX;
    let mut z2 = i32::MIN;
    let mut y2 = i32::MIN;
    let mut x2 = i32::MIN;
    distances.iter().enumerate().for_each(|(i, &d)| {
        let z = (center[0] + d * gs_vertices[[i, 0]] as f32).round_ties_even() as i32;
        let y = (center[1] + d * gs_vertices[[i, 1]] as f32).round_ties_even() as i32;
        let x = (center[2] + d * gs_vertices[[i, 2]] as f32).round_ties_even() as i32;
        z1 = z1.min(z);
        y1 = y1.min(y);
        x1 = x1.min(x);
        z2 = z2.max(z);
        y2 = y2.max(y);
        x2 = x2.max(x);
    });
    [z1, z2, y1, y2, x1, x2]
}

/// Compute the scaled 3D vertices of a polyhedron.
///
/// # Description
///
/// Computes the 3D vertices of a polyhedron by scaling a unit direction vector
/// (from a "Golden Spiral" unit sphere) with its corresponding ray distance and
/// translating by the center point.
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
/// * `Array2<f32>`: A 2D array of shape `(n_rays, 3)` containing the polyhedron
///   scaled vertices.
#[inline]
pub fn polyhedron_verts(
    distances: ArrayView1<f32>,
    center: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
) -> Array2<f32> {
    let n_rays = distances.len();
    distances
        .iter()
        .enumerate()
        .fold(Array2::<f32>::zeros((n_rays, 3)), |mut acc, (i, &d)| {
            acc[[i, 0]] = center[0] + d * gs_vertices[[i, 0]] as f32;
            acc[[i, 1]] = center[1] + d * gs_vertices[[i, 1]] as f32;
            acc[[i, 2]] = center[2] + d * gs_vertices[[i, 2]] as f32;
            acc
        })
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
pub fn polyhedron_vol(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> ImgalResult<f32> {
    let origin = [0.0_f32; 3];
    let n_faces = gs_faces.dim().0;
    Ok((0..n_faces)
        .try_fold(0.0_f32, |acc, f| {
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
            let v = tetrahedron_volume(&a, &b, &c, &origin)? as f32;
            Ok(acc + v)
        })?
        .abs())
}

/// TODO
///
/// # Arguments
///
/// * `bbox`: The bounding box.
/// * `nz`: The Z-axis bounding box size (*i.e.* depth).
/// * `ny`: The Y-axis bounding box size (*i.e.* height).
/// * `nx`: The X-axis bounding box size (*i.e.* width).
///
/// # Returns
///
/// * `Vec<bool>`:
#[inline]
pub fn polyhedron_to_mask(
    vertices: ArrayView2<f32>,
    gs_faces: ArrayView2<usize>,
    center: ArrayView1<f32>,
    bbox: [i32; 6],
    nz: usize,
    ny: usize,
    nx: usize,
) -> ImgalResult<Vec<bool>> {
    let mut render = vec![false; nz * ny * nx];
    let center = center.mapv(|v| v as f32);
    (0..nz).try_for_each(|z| {
        (0..ny).try_for_each(|y| {
            (0..nx).try_for_each(|x| {
                let query = Array1::from_iter([
                    (z as i32 + bbox[0]) as f32,
                    (y as i32 + bbox[2]) as f32,
                    (x as i32 + bbox[4]) as f32,
                ]);
                render[x + y * nx + z * nx * ny] =
                    inside_polyhedron(vertices, gs_faces, center.view(), query.view(), false)?;
                Ok(())
            })?;
            Ok(())
        })?;
        Ok(())
    })?;
    Ok(render)
}

/// Compute the intersection volume of two spheres with isotropic distance. If
/// the two spheres do not intersect the returned volume is `0.0`.
///
/// # Arguments
///
/// * `center_a`: The center coordinates for sphere `a`.
/// * `center_b`: The center coordinates for sphere `b`.
/// * `radius_a`: The radius for sphere `a`.
/// * `radius_b`: The radius for sphere `b`.
/// * `anisotropy`: The estimated average anisotropy.
///
/// # Returns
///
/// * `f32`: The intersection volume between spheres `a` and `b`.
#[inline]
pub fn sphere_intersect_volume_iso(
    center_a: ArrayView1<f32>,
    center_b: ArrayView1<f32>,
    radius_a: f32,
    radius_b: f32,
    anisotropy: &[f32; 3],
) -> f32 {
    let dz = anisotropy[0] * (center_a[0] - center_b[0]);
    let dy = anisotropy[1] * (center_a[1] - center_b[1]);
    let dx = anisotropy[2] * (center_a[2] - center_b[2]);
    let dist_iso = (dz * dz + dy * dy + dx * dx).sqrt();
    let rad_min = radius_a.min(radius_b);
    let rad_max = radius_a.max(radius_b);
    let pi = PI as f32;
    if dist_iso > radius_a + radius_b {
        return 0.0;
    }
    if rad_max >= dist_iso + rad_min - 1e-10 {
        return pi * 4.0 / 3.0 * rad_min * rad_min * rad_min;
    }
    let t = (radius_a + radius_b - dist_iso) / (2.0 * dist_iso);
    let h1 = (radius_b - radius_a + dist_iso) * t;
    let h2 = (radius_a - radius_b + dist_iso) * t;
    let vol_a = pi / 3.0 * h1 * h1 * (3.0 * radius_a - h1);
    let vol_b = pi / 3.0 * h2 * h2 * (3.0 * radius_b - h2);
    (vol_a + vol_b) / (anisotropy[0] * anisotropy[1] * anisotropy[2])
}
