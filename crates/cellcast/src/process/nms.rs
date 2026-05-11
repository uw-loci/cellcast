use std::array;

use imgal::error::ImgalError;
use imgal::spatial::KDTree;
use imgal::spatial::halfspace::{face_to_halfspace, halfspace_intersection};
use imgal::statistics::max;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis, stack};

use crate::geometry::polygon::{area_intersection, build_polygons, check_bbox_intersect};
use crate::geometry::polyhedron::{
    bbox_intersect_volume, bounding_inner_radius, bounding_inner_radius_iso, bounding_outer_radius,
    bounding_outer_radius_iso, estimate_anisotropy, golden_spiral, polyhedron_bbox,
    polyhedron_vertices, polyhedron_volume, sphere_intersect_volume_iso,
};

/// Perform Non-Maximum Suppression (NMS) on 2-dimensional polygons.
///
/// # Description
///
/// Performs None-Maximum Suppression (NMS) that suppresses overlappin polygons
/// based on their intersection area. Input distances and polygon positions are
/// expected in descending order, with the highest probability first.
///
/// # Arguments
///
/// * `polygon_dist`: Input radial distance array with shape `(n_polys, n_rays)`
///   containing the radial distances from polygon centers to their boundaries
///   at each ray angle. This array must be pre-sorted in descending order by
///   polygon probability.
/// * `polygon_pos`: Input polygon positions array with shape `(n_polys, 2)`
///   containing the (row, col) coordinates of polygon centers. This array must
///   be pre-sorted in descending order by polygon probability.
/// * `n_polys`: The number of polygons.
/// * `n_rays`: The number of ray angles.
/// * `threshold`: The overlap threshold in range `0` to `1`. Polygons exceeding
///   this overlap threshold value are suppressed.
///
/// # Returns
///
/// * `Vec<bool>`: A boolean array of length `n_polys` where `True` indicates
///   valid or non-suppressed polygon indices (*i.e.* polygons that should be
///   kept).
pub fn polygon_nms(
    polygon_dist: ArrayView2<f32>,
    polygon_pos: ArrayView2<usize>,
    n_polys: usize,
    n_rays: usize,
    threshold: f32,
) -> Vec<bool> {
    // create 2D polygons vector and perform NMS
    let mut suppressed: Vec<bool> = vec![false; n_polys];
    let polygons = build_polygons(polygon_dist.view(), polygon_pos.view(), n_polys, n_rays);
    let kdtree = KDTree::build(polygon_pos.view());
    let max_dist = polygons.iter().map(|p| p.dist).fold(0.0, f32::max);
    // iterate through each polygon and skip already suppressed polygons
    // the key here is that each polygon's probability is encoded in it's order
    // as it was sorted in descending order (highest prob first)
    for p in 0..n_polys.saturating_sub(1) {
        // skip already suppressed polygons
        if suppressed[p] {
            continue;
        }
        // find neighboring polygons within the computed search radius
        let query = [polygon_pos[[p, 0]], polygon_pos[[p, 1]]];
        let radius = (max_dist + polygons[p].dist) as f64;
        let neighbors = kdtree.search_for_indices(&query, radius).unwrap();
        // skip already suppressed polygons
        for n in neighbors {
            // skip already process polygons
            if n <= p || suppressed[n] {
                continue;
            }
            // skip bounding boxes that *do not* intersect
            if !check_bbox_intersect(&polygons[p].bbox, &polygons[n].bbox) {
                continue;
            }
            let poly_area_inter = area_intersection(&polygons[p].vertices, &polygons[n].vertices);
            let min_area = polygons[p].area.min(polygons[n].area) + 1e-10;
            let overlap = poly_area_inter / min_area;
            if overlap > threshold {
                suppressed[n] = true;
            }
        }
    }
    // invert the suppressed array, `true` indicates valid or non-suppressed
    // polygon indices
    suppressed.iter().map(|&v| !v).collect()
}

/// TODO
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `polyhedron_dist`:
/// * `polyhedron_pnts`:
/// * `polyhedron_prob`:
/// * `n_polys`:
/// * `n_rays`:
/// * `threshold`:
///
/// # Returns
///
/// * `Ok(Vec<bool>)`:
/// * `Err(ImgalError)`:
pub fn polyhedron_nms(
    polyhedron_dist: ArrayView2<f32>,
    polyhedron_pnts: ArrayView2<usize>,
    polyhedron_prob: ArrayView1<f32>,
    n_polys: usize,
    n_rays: usize,
    threshold: f32,
) -> Result<Vec<bool>, ImgalError> {
    let eps = 1e-10;
    let gs = golden_spiral(n_rays, None)?;
    let verts = gs.0.view();
    let faces = gs.1.view();
    let (bboxes, vols, rad_in, rad_out): (Vec<[usize; 6]>, Vec<f32>, Vec<f32>, Vec<f32>) = (0
        ..n_polys)
        .map(|i| {
            let cur_dist = polyhedron_dist.row(i);
            let cur_pnt = polyhedron_pnts.row(i);
            let bbox = polyhedron_bbox(cur_dist, cur_pnt, verts);
            let vol = polyhedron_volume(cur_dist, verts, faces);
            let ri = bounding_inner_radius(cur_dist, verts, faces);
            let ro = bounding_outer_radius(cur_dist);
            (bbox, vol, ri, ro)
        })
        .collect();
    let aniso = estimate_anisotropy(&bboxes.as_slice(), n_polys);
    let (rad_in_iso, rad_out_iso): (Vec<f32>, Vec<f32>) = (0..n_polys)
        .map(|i| {
            let cur_dist = polyhedron_dist.row(i);
            let rii = bounding_inner_radius_iso(cur_dist, verts, faces, aniso);
            let roi = bounding_outer_radius_iso(cur_dist, verts, aniso);
            (rii, roi)
        })
        .collect();
    let max_dist = max(&rad_out, false)?;
    let kdtree = KDTree::build(&polyhedron_pnts);
    // note that this implementation is avoiding external buffers with the hope
    // of making parallization easier
    let suppressed =
        (0..n_polys.saturating_sub(1)).try_fold(vec![false; n_polys], |mut sup, i| {
            if sup[i] {
                return Ok(sup);
            }
            let cur_dist = polyhedron_dist.row(i);
            let cur_pnt = polyhedron_pnts.row(i);
            let cur_bbox = bboxes[i];
            let cur_poly_verts = polyhedron_vertices(cur_dist, cur_pnt, verts);
            let search_rad = ((max_dist + rad_out[i]) * (max_dist + rad_out[i])) as f64;
            let neighbors = kdtree.search_for_indices(&cur_pnt, search_rad)?;
            let sup_inds: Vec<usize> =
                neighbors
                    .iter()
                    .filter(|&&v| !sup[v])
                    .fold(Vec::new(), |mut si, &v| {
                        let mut iou: f32 = 0.0;
                        let ngh_dist = polyhedron_dist.row(v);
                        let ngh_pnt = polyhedron_pnts.row(v);
                        let vol_min = vols[i].min(vols[v]);
                        // this checks the upper bound of intersection and IoU
                        let upper_inter_vol = sphere_intersect_volume_iso(
                            cur_pnt,
                            ngh_pnt,
                            rad_out_iso[i],
                            rad_out_iso[v],
                            &aniso,
                        );
                        let bbox_inter_vol = bbox_intersect_volume(&cur_bbox, &bboxes[v]);
                        let upper_inter_vol = upper_inter_vol.min(bbox_inter_vol);
                        iou = (upper_inter_vol / (vol_min + eps)).min(1.0);
                        if upper_inter_vol < eps || iou <= threshold {
                            return si;
                        }
                        // this checks the lower bound of intersection and IoU
                        let lower_inter_vol = sphere_intersect_volume_iso(
                            cur_pnt,
                            ngh_pnt,
                            rad_in_iso[i],
                            rad_in_iso[v],
                            &aniso,
                        );
                        iou = (lower_inter_vol / (vol_min + eps)).max(0.0);
                        if iou > threshold {
                            si.push(v);
                            return si;
                        }
                        // this computes the kernel intersection of the lower bound
                        let ngh_poly_verts = polyhedron_vertices(ngh_dist, ngh_pnt, verts);
                        let x = hull_overlap_volume(
                            cur_poly_verts.view(),
                            ngh_poly_verts.view(),
                            cur_pnt,
                            ngh_pnt,
                            faces,
                        ).unwrap();
                        dbg!(x);
                        si
                    });
            // TODO use the suppressed indices to update date the sup accumulator and
            // return it -- this is parallel friendly
            // these coordinates come later
            let nz = cur_bbox[1] - cur_bbox[0] + 1;
            let ny = cur_bbox[3] - cur_bbox[2] + 1;
            let nx = cur_bbox[5] - cur_bbox[4] + 1;
            Ok(sup)
        })?;
    todo!();
}

/// Computes the overlap volume of two convex hulls by intersecting their face
/// halfspaces and summing the volume of the result intersection hull.
///
/// # Arguments
///
/// * `vertices_a`:
/// * `vertices_b`:
/// * `center_a`:
/// * `center_b`:
/// * `gs_faces`:
///
/// # Retruns
///
/// * `Ok()`:
/// * `Err()`:
#[inline]
fn hull_overlap_volume(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<usize>,
    center_b: ArrayView1<usize>,
    gs_faces: ArrayView2<usize>,
) -> Result<f64, ImgalError> {
    let n_gsf = gs_faces.dim().0;
    let hs: Vec<Array1<f64>> = (0..n_gsf).try_fold(Vec::with_capacity(n_gsf * 2), |mut acc, i| {
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
    let in_pnt: [f64; 3] =
        array::from_fn(|i| 0.5 * (center_a[i] + center_b[i]) as f64);
    let hs = stack(
        Axis(0),
        &hs.iter()
            .map(|v| v.view())
            .collect::<Vec<ArrayView1<f64>>>(),
    )
    .expect("Failed to stack halfspaces into array.");
    let (inter_verts, inter_faces) = halfspace_intersection(&hs, &in_pnt)?;
    let n_if = inter_faces.dim().0;
    Ok((0..n_if).fold(0.0_f64, |_, i| {
        let [a_idx, b_idx, c_idx] = array::from_fn(|j| inter_faces[[i, j]]);
        let [az, ay, ax] = array::from_fn(|j| inter_verts[[a_idx, j]] - in_pnt[j] as f64);
        let [bz, by, bx] = array::from_fn(|j| inter_verts[[b_idx, j]] - in_pnt[j] as f64);
        let [cz, cy, cx] = array::from_fn(|j| inter_verts[[c_idx, j]] - in_pnt[j] as f64);
        let cross_z = bx * cy - by * cx;
        let cross_y = bz * cx - bx * cz;
        let cross_x = by * cz - bz * cy;
        let temp = az * cross_z + ay * cross_y + ax * cross_x;
        (temp / 6.0).abs()
    }))
}
