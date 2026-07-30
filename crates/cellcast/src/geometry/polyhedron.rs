use std::f32::consts::PI;

use imgal::prelude::*;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::{inside_polyhedron, tetrahedron_volume};
use imgal::spatial::halfspace::{face_to_halfspace, halfspace_intersection, hull_to_halfspace};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, stack};

/// Compute the inersection volume of two 3D bounding boxes.
///
/// # Description
///
/// Computes the intersection volume of two axis-aligned 3D bounding boxes.
///
/// # Arguments
///
/// * `bbox_a`: The coordinates of bounding box `a`.
/// * `bbox_b`: The coordinates of bounding box `b`.
///
/// # Returns
///
/// * `f32`: The intersection volume of bounding box `a` and `b`.
#[inline(always)]
pub fn bbox_intersect_vol(bbox_a: &[i32; 6], bbox_b: &[i32; 6]) -> f32 {
    let wz = (bbox_a[1].min(bbox_b[1]) - bbox_a[0].max(bbox_b[0])).max(0) as f32;
    let wy = (bbox_a[3].min(bbox_b[3]) - bbox_a[2].max(bbox_b[2])).max(0) as f32;
    let wx = (bbox_a[5].min(bbox_b[5]) - bbox_a[4].max(bbox_b[4])).max(0) as f32;
    wz * wy * wx
}

/// Compute the inscribed (inner) radius of a polyhedron with anisotropic
/// scaling.
///
/// # Description
///
/// Computes the minimum distance from the origin to the planes defined by
/// the triangular faces of the polyhedron after applying `anisotropy` to the
/// vertices (*i.e.* `anisotropy * distances * gs_vertices`). This value is
/// the radius of the largest sphere centred at the origin that fits inside
/// the anisotropically scaled polyhedron.
///
/// # Arguments
///
/// * `distances`: The polyhedron distances corresponding to each `gs_vertices`
///   direction.
/// * `gs_faces`: The golden spiral unit sphere triangular face indices with
///   shape `(n_triangles, 3)`.
/// * `gs_vertices`: The golden spiral unit sphere vertices with shape
///   `(n_points, 3)`.
/// * `anisotropy`: The 1D anisotropy array.
///
/// # Returns
///
/// * `f32`: The inscribed (inner) scaled radius.
#[inline(always)]
pub fn bounding_inner_radius_iso(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f32>,
    gs_faces: ArrayView2<usize>,
    anisotropy: [f32; 3],
) -> f32 {
    let eps = 1e-10;
    (0..gs_faces.dim().0).fold(f32::MAX, |acc, i| {
        let face = gs_faces.row(i);
        let i_a = face[0];
        let i_b = face[1];
        let i_c = face[2];
        let verts_a = gs_vertices.row(i_a);
        let verts_b = gs_vertices.row(i_b);
        let verts_c = gs_vertices.row(i_c);
        let [anz, any, anx] = anisotropy;
        let a = {
            let dist = distances[i_a];
            [
                anz * dist * verts_a[0],
                any * dist * verts_a[1],
                anx * dist * verts_a[2],
            ]
        };
        let b = {
            let dist = distances[i_b];
            [
                anz * dist * verts_b[0],
                any * dist * verts_b[1],
                anx * dist * verts_b[2],
            ]
        };
        let c = {
            let dist = distances[i_c];
            [
                anz * dist * verts_c[0],
                any * dist * verts_c[1],
                anx * dist * verts_c[2],
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

/// Compute the outer (circum) radius of a polyhedron with anisotropic
/// scaling.
///
/// # Description
///
/// Computes the maximum Euclidean distance of the anisotropically scaled
/// polyhedron vertices from the origin (i.e. `anisotropy * distances *
/// gs_vertices`) and returns its square root. This is the radius of the
/// smallest sphere centred at the origin that contains the scaled vertices.
///
/// # Arguments
///
/// * `distances`: The polyhedron distances corresponding to each `gs_vertices`
///   direction.
/// * `gs_vertices`: Unit sphere vertices of shape `(n_points, 3)`.
/// * `anisotropy`: The 1D anisotropy array.
///
/// # Returns
///
/// * `f32`: The outer (circum) scaled radius.
#[inline(always)]
pub fn bounding_outer_radius_iso(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f32>,
    anisotropy: [f32; 3],
) -> f32 {
    let [anz, any, anx] = anisotropy;
    let radius = (0..distances.len()).fold(0.0_f32, |acc, i| {
        let dist = distances[i];
        let verts = gs_vertices.row(i);
        let z = anz * dist * verts[0];
        let y = any * dist * verts[1];
        let x = anx * dist * verts[2];
        acc.max(z * z + y * y + x * x)
    });
    radius.sqrt()
}

/// Compute the intersection volume of two convex hulls.
///
/// # Description
///
/// Computes the intersection volume of between two sets of vertices, `a` and
/// `b`, by creating convex hulls and converting each hull to a halfspace
/// representation. The halfspaces are combined and the halfspace intersection
/// computed and its volume returned. The interior point for the halfspace
/// intersection is chosen as the midpoint of `center_a` and `center_b`.
///
/// # Arguments
///
/// * `vertices_a`: Vertices of polyhedron `a`.
/// * `vertices_b`: Vertices of polyhedron `b`.
/// * `center_a`: The center point of polyhedron `a` (used to compute an
///   interior point for the halfspace intersection).
/// * `center_b`: The center point of polyhedron `b`.
///
/// # Returns
///
/// * `Ok(f64)`: The intersection volume of polyhedron `a` and `b`.
/// * `Err(ImgalError)`: If `vertices_a` or `vertices_b` is empty or contains
///   less than 4 ponits. If `center_a` and `center_b` do not have length
///   equal to `3`.
#[inline(always)]
pub fn convex_hull_intersection_vol(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<f32>,
    center_b: ArrayView1<f32>,
) -> Result<f64, ImgalError> {
    let (hull_verts_a, hull_faces_a) = quickhull_3d(vertices_a, None)?;
    let (hull_verts_b, hull_faces_b) = quickhull_3d(vertices_b, None)?;
    let hs_a = hull_to_halfspace(&hull_verts_a, &hull_faces_a, None)?;
    let hs_b = hull_to_halfspace(&hull_verts_b, &hull_faces_b, None)?;
    let hs = concatenate(Axis(0), &[hs_a.view(), hs_b.view()])
        .expect("Failed to stack halfspaces into array.");
    let in_pnt = [
        0.5 * (center_a[0] + center_b[0]) as f64,
        0.5 * (center_a[1] + center_b[1]) as f64,
        0.5 * (center_a[2] + center_b[2]) as f64,
    ];
    let (inter_verts, inter_faces) = halfspace_intersection(&hs, &in_pnt, None)?;
    let n_if = inter_faces.dim().0;
    let [pz, py, px] = in_pnt;
    Ok((0..n_if).fold(0.0_f64, |acc, i| {
        let face = inter_faces.row(i);
        let inter_verts_a = inter_verts.row(face[0]);
        let inter_verts_b = inter_verts.row(face[1]);
        let inter_verts_c = inter_verts.row(face[2]);
        let az = inter_verts_a[0] - pz;
        let ay = inter_verts_a[1] - py;
        let ax = inter_verts_a[2] - px;
        let bz = inter_verts_b[0] - pz;
        let by = inter_verts_b[1] - py;
        let bx = inter_verts_b[2] - px;
        let cz = inter_verts_c[0] - pz;
        let cy = inter_verts_c[1] - py;
        let cx = inter_verts_c[2] - px;
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
#[inline(always)]
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

/// Create a golden spiral unit sphere
///
/// # Description
///
/// Creates a golden spiral unit sphere which is used to determine the direction
/// a ray points.
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
#[inline(always)]
pub fn golden_spiral(
    n_points: usize,
    anisotropy: Option<[f32; 3]>,
) -> Result<(Array2<f32>, Array2<usize>), ImgalError> {
    let anisotropy = anisotropy.unwrap_or([1.0; 3]);
    let aniso = ArrayView1::from(&anisotropy);
    let golden_angle = (3.0 - 5.0_f32.sqrt()) * PI;
    let phi = Array1::from_iter(0..n_points).mapv(|v| v as f32 * golden_angle);
    let z = Array1::linspace(-1.0, 1.0, n_points);
    let rho = z.mapv(|v| (1.0_f32 - v * v).sqrt());
    let a = &rho * phi.mapv(|v| v.sin());
    let b = &rho * phi.mapv(|v| v.cos());
    let ax = Axis(1);
    let points = stack(ax, &[z.view(), a.view(), b.view()])
        .expect("Failed to create Golden spiral point cloud.");
    let points = points / aniso;
    let (mut verts, faces) = quickhull_3d(&points, None)?;
    let norms = verts.map_axis(ax, |r| r.dot(&r).sqrt());
    verts /= &norms.insert_axis(ax);
    Ok((verts, faces))
}

/// Compute the overlap intersection volume.
///
/// # Description
///
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
#[inline(always)]
pub fn golden_spiral_intersection_vol(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<f32>,
    center_b: ArrayView1<f32>,
    gs_faces: ArrayView2<usize>,
) -> Result<f64, ImgalError> {
    let n_gsf = gs_faces.dim().0;
    let mut hs_stack = Array2::<f64>::zeros((n_gsf * 2, 4));
    let in_pnt = [
        0.5 * (center_a[0] + center_b[0]) as f64,
        0.5 * (center_a[1] + center_b[1]) as f64,
        0.5 * (center_a[2] + center_b[2]) as f64,
    ];
    (0..n_gsf).for_each(|i| {
        let face = gs_faces.row(i);
        let a_idx = face[0];
        let b_idx = face[1];
        let c_idx = face[2];
        // SAFE: safe because points a, b and c are all length 3
        let hs_a = face_to_halfspace(
            vertices_a.row(a_idx),
            vertices_a.row(b_idx),
            vertices_a.row(c_idx),
        )
        .unwrap();
        let hs_b = face_to_halfspace(
            vertices_b.row(a_idx),
            vertices_b.row(b_idx),
            vertices_b.row(c_idx),
        )
        .unwrap();
        hs_stack.row_mut(i * 2).assign(&hs_a);
        hs_stack.row_mut(i * 2 + 1).assign(&hs_b);
    });
    let (inter_verts, inter_faces) = halfspace_intersection(&hs_stack, &in_pnt, None)?;
    let n_if = inter_faces.dim().0;
    let [pz, py, px] = in_pnt;
    Ok((0..n_if).fold(0.0_f64, |acc, i| {
        let face = inter_faces.row(i);
        let inter_verts_a = inter_verts.row(face[0]);
        let inter_verts_b = inter_verts.row(face[1]);
        let inter_verts_c = inter_verts.row(face[2]);
        let az = inter_verts_a[0] - pz;
        let ay = inter_verts_a[1] - py;
        let ax = inter_verts_a[2] - px;
        let bz = inter_verts_b[0] - pz;
        let by = inter_verts_b[1] - py;
        let bx = inter_verts_b[2] - px;
        let cz = inter_verts_c[0] - pz;
        let cy = inter_verts_c[1] - py;
        let cx = inter_verts_c[2] - px;
        let cross_z = bx * cy - by * cx;
        let cross_y = bz * cx - bx * cz;
        let cross_x = by * cz - bz * cy;
        let temp = az * cross_z + ay * cross_y + ax * cross_x;
        acc + (temp / 6.0).abs()
    }))
}

/// Count the number of voxels in a mask that fall inside a polyhedron.
///
/// # Description
///
/// Counts the number of voxels within a mask (marked `true`) that fall inside
/// the given polyhedron by testing if a voxel with `inside_polyhedron`.
///
/// # Arguments
///
/// * `vertices`: The polyhedron vertices with shape `(n_vertices, 3)`.
/// * `faces`: The polyhedron triangular face indices with shape
///   `(n_triangles, 3)`.
/// * `center`: The centre point of the polyhedron.
/// * `mask`: A boolean slice of length `nz * ny * nx` indicating which
///   voxels to test.
/// * `bbox`: The bounding box coordinates in
///   `[z_min, z_max, y_min, y_max, x_min, x_max]` order.
/// * `nz`: The z axis bounding box length.
/// * `ny`: The y axis bounding box length.
/// * `nx`: The x axis bounding box length.
/// * `overlap_threshold`: The overlap count threshold.
///
/// # Returns
///
/// * `f32`: The number of mask voxels that lie inside the polyhedron that are
///   below the `overlap_threshold`.
#[inline(always)]
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
) -> f32 {
    let mut count = 0.0;
    let mut query = [0.0; 3];
    let nx_ny = nx * ny;
    let bz = bbox[0] as f32;
    let by = bbox[2] as f32;
    let bx = bbox[4] as f32;
    for z in 0..nz {
        let z_nx_ny = z * nx_ny;
        for y in 0..ny {
            let y_nx = y * nx;
            for x in 0..nx {
                let idx = x + y_nx + z_nx_ny;
                if !mask[idx] {
                    continue;
                }
                query[0] = bz + z as f32;
                query[1] = by + y as f32;
                query[2] = bx + x as f32;
                let query_view = ArrayView1::from(&query);
                if inside_polyhedron(vertices, faces, center, query_view, None).unwrap_or(false) {
                    count += 1.0;
                    if count > overlap_threshold {
                        return count;
                    }
                }
            }
        }
    }
    count
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
#[inline(always)]
pub fn polyhedron_bbox(
    distances: ArrayView1<f32>,
    center: ArrayView1<f32>,
    gs_vertices: ArrayView2<f32>,
) -> [i32; 6] {
    let mut z1 = i32::MAX;
    let mut y1 = i32::MAX;
    let mut x1 = i32::MAX;
    let mut z2 = i32::MIN;
    let mut y2 = i32::MIN;
    let mut x2 = i32::MIN;
    let cen_z = center[0];
    let cen_y = center[1];
    let cen_x = center[2];
    distances.iter().enumerate().for_each(|(i, &d)| {
        let vert = gs_vertices.row(i);
        let z = (cen_z + d * vert[0]).round_ties_even() as i32;
        let y = (cen_y + d * vert[1]).round_ties_even() as i32;
        let x = (cen_x + d * vert[2]).round_ties_even() as i32;
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
#[inline(always)]
pub fn polyhedron_verts(
    distances: ArrayView1<f32>,
    center: ArrayView1<f32>,
    gs_vertices: ArrayView2<f32>,
) -> Array2<f32> {
    let n_rays = distances.len();
    let cz = center[0];
    let cy = center[1];
    let cx = center[2];
    distances
        .iter()
        .enumerate()
        .fold(Array2::<f32>::zeros((n_rays, 3)), |mut acc, (i, &d)| {
            let gs_verts = gs_vertices.row(i);
            let mut poly_verts = acc.row_mut(i);
            poly_verts[0] = cz + d * gs_verts[0];
            poly_verts[1] = cy + d * gs_verts[1];
            poly_verts[2] = cx + d * gs_verts[2];
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
#[inline(always)]
pub fn polyhedron_vol(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f32>,
    gs_faces: ArrayView2<usize>,
) -> Result<f32, ImgalError> {
    let origin = [0.0_f32; 3];
    let n_faces = gs_faces.dim().0;
    Ok((0..n_faces)
        .try_fold(0.0_f32, |acc, f| {
            let tri = gs_faces.row(f);
            let a: [f32; 3] = {
                let i = tri[0];
                let di = distances[i];
                [
                    di * gs_vertices[[i, 0]],
                    di * gs_vertices[[i, 1]],
                    di * gs_vertices[[i, 2]],
                ]
            };
            let b: [f32; 3] = {
                let i = tri[1];
                let di = distances[i];
                [
                    di * gs_vertices[[i, 0]],
                    di * gs_vertices[[i, 1]],
                    di * gs_vertices[[i, 2]],
                ]
            };
            let c: [f32; 3] = {
                let i = tri[2];
                let di = distances[i];
                [
                    di * gs_vertices[[i, 0]],
                    di * gs_vertices[[i, 1]],
                    di * gs_vertices[[i, 2]],
                ]
            };
            let v = tetrahedron_volume(&a, &b, &c, &origin)? as f32;
            Ok(acc + v)
        })?
        .abs())
}

/// Render a polyhedron into a boolean voxel mask.
///
/// # Description
///
/// Renders the given polyhedron into a 1D boolean mask of lengh `nz * ny * nx`,
/// associated with the `bbox`. Each voxel in `bbox` is tested to determine if
/// it lies inside the polyhedron.
///
/// # Arguments
///
/// * `vertices`: The polyhedron vertices with shape `(n_vertices, 3)`.
/// * `gs_faces`: The triangular face indices with shape `(n_triangles, 3)`.
/// * `center`: The center point of the polyhedron.
/// * `bbox`: The bounding box coordinates in
///   `[z_min, z_max, y_min, y_max, x_min, x_max]` order.
/// * `nz`: The z axis bounding box length.
/// * `ny`: The y axis bounding box length.
/// * `nx`: The x axis bounding box length.
///
/// # Returns
///
/// * `Vec<bool>`: A boolean mask of length `nz * ny * nx` where `true`
///   indicates the voxel centre lies inside the polyhedron.
#[inline(always)]
pub fn polyhedron_to_mask(
    vertices: ArrayView2<f32>,
    gs_faces: ArrayView2<usize>,
    center: ArrayView1<f32>,
    bbox: [i32; 6],
    nz: usize,
    ny: usize,
    nx: usize,
) -> Vec<bool> {
    let mut render = vec![false; nz * ny * nx];
    let center: [f32; 3] = [center[0], center[1], center[2]];
    let mut query = [0.0; 3];
    let bz = bbox[0] as f32;
    let by = bbox[2] as f32;
    let bx = bbox[4] as f32;
    (0..nz).for_each(|z| {
        query[0] = bz + z as f32;
        (0..ny).for_each(|y| {
            query[1] = by + y as f32;
            (0..nx).for_each(|x| {
                query[2] = bx + x as f32;
                let idx = x + y * nx + z * nx * ny;
                render[idx] = inside_polyhedron(vertices, gs_faces, &center, &query, Some(1))
                    .unwrap_or(false);
            });
        });
    });
    render
}
/// Compute the intersection volume of two spheres.
///
/// # Description
///
/// Computes the intersection volume of two spheres with isotropic distance. If
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
#[inline(always)]
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
