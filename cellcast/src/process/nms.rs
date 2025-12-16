use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::geometry::polygon;

pub fn sparse_polygon_nms_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_prob: ArrayView1<f32>,
    polygon_pos: ArrayView2<usize>,
    threshold: f32,
) -> Vec<bool> {
    // get the indices that would sort polygon_prob in descending order
    let n_polys = polygon_prob.dim();
    let mut sorted_inds: Vec<usize> = (0..n_polys).collect();
    sorted_inds.sort_by(|&a, &b| polygon_prob[b].partial_cmp(&polygon_prob[a]).unwrap());

    // sort dist, prob and pos arrays with prob descending order indices
    let a = Axis(0);
    let src_inds = Array1::from_iter(0..n_polys);
    let poly_dist_sort = polygon_dist.select(a, &sorted_inds);
    let poly_prob_sort = polygon_prob.select(a, &sorted_inds);
    let poly_pos_sort = polygon_pos.select(a, &sorted_inds);
    let src_inds_sort = src_inds.select(a, &sorted_inds);

    // create NMS polygons and perform NMS
    let (n_polygons, n_rays) = poly_dist_sort.dim();
    let nms_polys = polygon::build_nms_polygons_2d(
        poly_dist_sort.view(),
        poly_pos_sort.view(),
        n_polys,
        n_rays,
    );

    todo!("Implement polygon NMS.");
}
