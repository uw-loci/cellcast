use imgal::error::ImgalError;
use imgal::spatial::KDTree;
use ndarray::{ArrayView1, ArrayView2};

use crate::geometry::polygon;
use crate::geometry::polyhedron::{golden_spiral, polyhedron_bbox, polyhedron_volume};

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
    let polygons =
        polygon::build_polygons(polygon_dist.view(), polygon_pos.view(), n_polys, n_rays);
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
            if !bbox_intersect_2d(&polygons[p].bbox, &polygons[n].bbox) {
                continue;
            }
            let poly_area_inter =
                polygon::area_intersection(&polygons[p].vertices, &polygons[n].vertices);
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
    let gs = golden_spiral(n_rays, None)?;
    let mut suppressed: Vec<bool> = vec![false; n_polys];
    let (bboxes, vols): (Vec<[usize; 6]>, Vec<f32>) = (0..n_polys)
        .map(|i| {
            let cur_dist = polyhedron_dist.row(i);
            let cur_pnt = polyhedron_pnts.row(i);
            let vol = polyhedron_volume(cur_dist, gs.0.view(), gs.1.view());
            let bbox = polyhedron_bbox(cur_dist, cur_pnt, gs.0.view(), n_rays);
            (bbox, vol)
        })
        .collect();
    let aniso = estimate_anisotropy(&bboxes.as_slice(), n_polys);
    todo!();
}

/// Determine if two bounding boxes intersect.
///
/// # Arguments
///
/// * `a`: Bounding box `a` as `(y_min, y_max, x_min, x_max)`.
/// * `b`: Bounding box `b` as `(y_min, y_max, x_min, x_max)`.
///
/// # Returns
///
/// * `bool`: Returns `true` if the bounding boxes overlap, `false` if they do
///   not.
#[inline]
fn bbox_intersect_2d(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> bool {
    b.0 <= a.1 && a.0 <= b.1 && b.2 <= a.3 && a.2 <= b.3
}

/// TODO
///
/// # Arguments
///
/// * `bboxes`:
/// * `n_polys`:
///
/// # Returns
///
/// * `[f32; 3]`:
#[inline]
fn estimate_anisotropy(bboxes: &[[usize; 6]], n_polys: usize) -> [f32; 3] {
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
