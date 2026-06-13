use std::sync::atomic::{AtomicBool, Ordering};

use imgal::prelude::*;
use imgal::spatial::KDTree;
use imgal::statistics::max;
use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::geometry::polygon::{area_intersection, build_polygons, check_bbox_intersect};
use crate::geometry::polyhedron::{
    bbox_intersect_vol, bounding_inner_radius_iso, bounding_outer_radius,
    bounding_outer_radius_iso, convex_hull_intersection_vol, estimate_anisotropy, golden_spiral,
    golden_spiral_intersection_vol, overlap_polyhedron_mask, polyhedron_bbox, polyhedron_to_mask,
    polyhedron_verts, polyhedron_vol, sphere_intersect_volume_iso,
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
    let suppressed: Vec<AtomicBool> = (0..n_polys).map(|_| AtomicBool::new(false)).collect();
    let polygons = build_polygons(polygon_dist.view(), polygon_pos.view(), n_polys, n_rays);
    let kdtree = KDTree::build(polygon_pos.view());
    let max_dist = polygons.iter().map(|p| p.dist).fold(0.0, f32::max);
    // iterate through each polygon and skip already suppressed polygons
    // the key here is that each polygon's probability is encoded in it's order
    // as it was sorted in descending order (highest prob first)
    (0..n_polys.saturating_sub(1)).for_each(|i| {
        if suppressed[i].load(Ordering::Relaxed) {
            return;
        }
        let query = [polygon_pos[[i, 0]], polygon_pos[[i, 1]]];
        let radius = (max_dist + polygons[i].dist) as f64;
        let neighbors = kdtree.search_for_indices(&query, radius).unwrap();
        neighbors.par_iter().for_each(|&j| {
            if j <= i || suppressed[j].load(Ordering::Relaxed) {
                return;
            }
            if !check_bbox_intersect(&polygons[i].bbox, &polygons[j].bbox) {
                return;
            }
            let poly_area_inter = area_intersection(&polygons[i].vertices, &polygons[j].vertices);
            let min_area = polygons[i].area.min(polygons[j].area) + 1e-10;
            let overlap = poly_area_inter / min_area;
            if overlap > threshold {
                suppressed[j].store(true, Ordering::Relaxed);
            }
        });
    });
    suppressed
        .iter()
        .map(|v| !v.load(Ordering::Relaxed))
        .collect()
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
    polyhedron_pnts: ArrayView2<f32>,
    anisotropy: [f64; 3],
    n_polys: usize,
    n_rays: usize,
    threshold: f32,
) -> Result<Vec<bool>, ImgalError> {
    let eps = 1e-10;
    let gs = golden_spiral(n_rays, Some(anisotropy))?;
    let verts = gs.0.view();
    let faces = gs.1.view();
    let (bboxes, vols, rad_out): (Vec<[i32; 6]>, Vec<f32>, Vec<f32>) = (0..n_polys)
        .map(|i| {
            let cur_dist = polyhedron_dist.row(i);
            let cur_pnt = polyhedron_pnts.row(i);
            let bbox = polyhedron_bbox(cur_dist, cur_pnt, verts);
            // SAFE: this unwrap is safe because we know that the parameters are
            // valid lengths here
            let vol = polyhedron_vol(cur_dist, verts, faces).unwrap();
            let ro = bounding_outer_radius(cur_dist);
            (bbox, vol, ro)
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
    let max_dist = max(&rad_out, None)?;
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
            let cur_poly_verts = polyhedron_verts(cur_dist, cur_pnt, verts);
            let search_rad = (max_dist + rad_out[i]) as f64;
            let neighbors = kdtree.search_for_indices(&cur_pnt, search_rad)?;
            // TODO use the suppressed indices to update date the sup accumulator and
            // return it -- this is parallel friendly
            let nz = (cur_bbox[1] - cur_bbox[0] + 1) as usize;
            let ny = (cur_bbox[3] - cur_bbox[2] + 1) as usize;
            let nx = (cur_bbox[5] - cur_bbox[4] + 1) as usize;
            let sup_inds: Vec<usize> = neighbors.iter().filter(|&&j| j > i && !sup[j]).try_fold(
                Vec::new(),
                |mut si, &j| {
                    let mut iou: f32;
                    let ngh_dist = polyhedron_dist.row(j);
                    let ngh_pnt = polyhedron_pnts.row(j);
                    let vol_min = vols[i].min(vols[j]);
                    // this checks the upper bound of intersection and IoU
                    let upper_inter_vol = sphere_intersect_volume_iso(
                        cur_pnt,
                        ngh_pnt,
                        rad_out_iso[i],
                        rad_out_iso[j],
                        &aniso,
                    );
                    let bbox_inter_vol = bbox_intersect_vol(&cur_bbox, &bboxes[j]);
                    let upper_inter_vol = upper_inter_vol.min(bbox_inter_vol);
                    iou = (upper_inter_vol / (vol_min + eps)).min(1.0);
                    if upper_inter_vol < eps || iou <= threshold {
                        return Ok(si);
                    }
                    // this checks the lower bound of intersection and IoU
                    let lower_inter_vol = sphere_intersect_volume_iso(
                        cur_pnt,
                        ngh_pnt,
                        rad_in_iso[i],
                        rad_in_iso[j],
                        &aniso,
                    );
                    iou = (lower_inter_vol / (vol_min + eps)).max(0.0);
                    if iou > threshold {
                        si.push(j);
                        return Ok(si);
                    }
                    // this computes the polyhedron intersection of the lower bound
                    let ngh_poly_verts = polyhedron_verts(ngh_dist, ngh_pnt, verts);
                    let poly_inter_vol = golden_spiral_intersection_vol(
                        cur_poly_verts.view(),
                        ngh_poly_verts.view(),
                        cur_pnt,
                        ngh_pnt,
                        faces,
                    )
                    .unwrap_or(0.0) as f32;
                    iou = poly_inter_vol / (vol_min + eps);
                    if iou > threshold {
                        si.push(j);
                        return Ok(si);
                    }
                    let conv_inter_vol = convex_hull_intersection_vol(
                        cur_poly_verts.view(),
                        ngh_poly_verts.view(),
                        cur_pnt,
                        ngh_pnt,
                    )
                    .unwrap_or(1e10) as f32;
                    iou = conv_inter_vol / (vol_min + eps);
                    if iou <= threshold {
                        return Ok(si);
                    }
                    // this computes a polygon rendering check, the final check
                    let cur_poly_mask = polyhedron_to_mask(
                        cur_poly_verts.view(),
                        faces,
                        cur_pnt,
                        cur_bbox,
                        nz,
                        ny,
                        nx,
                    )?;
                    let overlap_count = overlap_polyhedron_mask(
                        ngh_poly_verts.view(),
                        faces,
                        ngh_pnt,
                        &cur_poly_mask,
                        cur_bbox,
                        nz,
                        ny,
                        nx,
                        (vol_min + eps) * threshold,
                    )? as f32;
                    iou = overlap_count / (vol_min + eps);
                    if iou > threshold {
                        si.push(j)
                    }
                    Ok(si)
                },
            )?;
            sup_inds.iter().for_each(|&i| sup[i] = true);
            Ok(sup)
        })?;
    Ok(suppressed.iter().map(|&v| !v).collect())
}
