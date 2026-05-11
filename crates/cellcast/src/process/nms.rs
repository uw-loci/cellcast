use std::array;

use imgal::error::ImgalError;
use imgal::spatial::KDTree;
use imgal::spatial::halfspace::{face_to_halfspace};
use imgal::statistics::max;
use ndarray::{Array1, ArrayView1, ArrayView2};

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
                        let x = hull_overlap_kernel(
                            cur_poly_verts.view(),
                            ngh_poly_verts.view(),
                            cur_pnt,
                            ngh_pnt,
                            faces,
                        );
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

/// TODO
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
fn hull_overlap_kernel(
    vertices_a: ArrayView2<f32>,
    vertices_b: ArrayView2<f32>,
    center_a: ArrayView1<usize>,
    center_b: ArrayView1<usize>,
    gs_faces: ArrayView2<usize>,
) -> Result<Vec<Array1<f64>>, ImgalError> {
    let n_f = gs_faces.dim().0;
    let hs: Vec<Array1<f64>> = (0..n_f).try_fold(Vec::with_capacity(n_f * 4), |mut acc, i| {
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
    Ok(hs)
}
