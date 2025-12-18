use imgal::spatial::KDTree;
use ndarray::ArrayView2;

use crate::geometry::polygon;

/// TODO
///
/// # Description
///
/// the polygon distances and positions should both be sorted based on their
/// probability before being passed to this function.
///
/// # Arguments
///
/// * `polygon_dist`:
/// * `polygon_pos`:
/// * `threshold`:
///
/// # Returns
///
/// * `Vec<usize>`: An array of valid polygon indices after performing NMS. If
///   an element is `true` then that polygon is not suppressed. If an element
///   is `false` then that polygon has been suppressed by NMS.
pub fn sparse_polygon_nms_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_pos: ArrayView2<usize>,
    n_polys: usize,
    n_rays: usize,
    threshold: f32,
) -> Vec<bool> {
    // create 2D polygons vector and perform NMS
    let mut suppressed: Vec<bool> = vec![false; n_polys];
    let polygons =
        polygon::build_polygons_2d(polygon_dist.view(), polygon_pos.view(), n_polys, n_rays);
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
        let neighbors = kdtree.search_for_indices(&query, radius);
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
                polygon::area_intersection_2d(&polygons[p].vertices, &polygons[n].vertices);
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

// Determine if two bounding boxes intersect.
fn bbox_intersect_2d(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> bool {
    b.0 <= a.1 && a.0 <= b.1 && b.2 <= a.3 && a.2 <= b.3
}
