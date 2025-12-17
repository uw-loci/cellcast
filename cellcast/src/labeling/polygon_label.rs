use std::f32::consts::TAU;

use ndarray::{Array3, ArrayView1, ArrayView2};

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
///
/// # Returns
///
/// * `Array2<u16>`: The 2-dimensional label image.
pub fn radial_polygon_to_label_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_prob: ArrayView1<f32>,
    polygon_pos: ArrayView2<usize>,
    scale: Option<(f32, f32)>,
) {
    // convert radial distances and point positions from polar to cartesian
    // coordinates
    let (n_polys, n_rays) = polygon_dist.dim();
    let coords = radial_dist_to_coords_2d(polygon_dist, polygon_pos, n_polys, n_rays, scale);

    todo!("Implement radial polygon to label function");
}

/// Convert polar distance representation to Cartesian coordinates.
fn radial_dist_to_coords_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_pos: ArrayView2<usize>,
    n_polys: usize,
    n_rays: usize,
    scale: Option<(f32, f32)>,
) -> Array3<f32> {
    let scale = scale.unwrap_or((1.0, 1.0));

    // get evenly spaced angles for 0 to 2*pi (TAU) and compute row, col
    // coordinates
    let angles: Vec<f32> = (0..n_rays)
        .map(|r| (r as f32) * TAU / (n_rays as f32))
        .collect();
    let mut coords = Array3::<f32>::zeros((n_polys, n_rays, 2));
    (0..n_polys).for_each(|p| {
        let poly_y = polygon_pos[[p, 0]] as f32;
        let poly_x = polygon_pos[[p, 1]] as f32;
        (0..n_rays).for_each(|r| {
            let d = polygon_dist[[p, r]];
            let a = angles[r];
            coords[[p, r, 0]] = poly_y + d * a.sin() * scale.0;
            coords[[p, r, 1]] = poly_x + d * a.cos() * scale.1;
        })
    });

    coords
}
