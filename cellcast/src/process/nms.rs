use imgal::spatial::KDTree;
use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::geometry::polygon;

/// TODO
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `polygon_dist`:
/// * `polygon_prob`:
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
    polygon_prob: ArrayView1<f32>,
    polygon_pos: ArrayView2<usize>,
    threshold: f32,
) -> Vec<bool> {
    // get the indices that would sort polygon_prob in descending order
    let (n_polys, n_rays) = polygon_dist.dim();
    let mut sorted_inds: Vec<usize> = (0..n_polys).collect();
    sorted_inds.sort_by(|&a, &b| polygon_prob[b].partial_cmp(&polygon_prob[a]).unwrap());

    // sort dist, prob and pos arrays with prob descending order indices
    let a = Axis(0);
    let poly_dist_sort = polygon_dist.select(a, &sorted_inds);
    let poly_pos_sort = polygon_pos.select(a, &sorted_inds);

    // create 2D polygons vector and perform NMS
    let mut suppressed: Vec<bool> = vec![false; n_polys];
    let polygons =
        polygon::build_polygons_2d(poly_dist_sort.view(), poly_pos_sort.view(), n_polys, n_rays);
    let kdtree = KDTree::build(poly_pos_sort.view());
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
        let query = [poly_pos_sort[[p, 0]], poly_pos_sort[[p, 1]]];
        let radius = (max_dist + polygons[p].dist) as f64;
        let neighbors = kdtree.search(&query, radius);
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
