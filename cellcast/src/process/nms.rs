use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

pub fn sparse_polygon_nms_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_prob: ArrayView1<f32>,
    polygon_pos: ArrayView2<usize>,
    threshold: f32,
) {
    // get the indices that would sort polygon_prob in descending order
    let n = polygon_prob.dim();
    let mut sorted_inds: Vec<usize> = (0..n).collect();
    sorted_inds.sort_by(|&a, &b| polygon_prob[b].partial_cmp(&polygon_prob[a]).unwrap());

    // sort dist, prob and pos arrays with prob descending order indices
    let a = Axis(0);
    let src_inds = Array1::from_iter(0..n);
    let poly_dist_sort = polygon_dist.select(a, &sorted_inds);
    let poly_prob_sort = polygon_prob.select(a, &sorted_inds);
    let poly_pos_sort = polygon_pos.select(a, &sorted_inds);
    let src_inds_sort = src_inds.select(a, &sorted_inds);
}
