//! Polygons structs and functions.
//!
//! This module provides specialized structs and functions for creating polygons
//! needed for pre- and post-processing steps.

use std::f32::consts::PI;

use ndarray::ArrayView2;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct NMSPolygon2D {
    /// This polygon bounding box coordinates `(y1, y2, x1, x2)`.
    pub bbox: (f32, f32, f32, f32),
    /// The polygon area.
    pub area: f32,
    /// The polygon maximum radius.
    pub dist: f32,
    /// The polygon vertices.
    pub vertices: Vec<(f32, f32)>,
}

/// Create a vector of 2-dimensional NMS polygons.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `dist`: The radial distances from the center to the polygon vertices in
///   shape `(p, D)`, where `p` is the point and `D` is the dimension of that
///   point value.
/// * `pos`: The positions of center point of each polygon.
/// * `n_polys`: The total number of polygons.
/// * `n_rays`: The total number of radial rays.
///
/// # Returns
///
/// * `Vec<NMSPolygon2D>`: A vector of NMS polygons to be used for geo-spatial
///   compute.
pub fn build_nms_polygons_2d(
    dist: ArrayView2<f32>,
    pos: ArrayView2<usize>,
    n_polys: usize,
    n_rays: usize,
) -> Vec<NMSPolygon2D> {
    // iterate through the radial distances in parrallel for each ray angle and
    // construct the NMS polygon vector
    let angle_step = 2.0 * PI / n_rays as f32;
    let nms_polygons: Vec<NMSPolygon2D> = (0..n_polys)
        .into_par_iter()
        .map(|p| {
            // get the current polygon center, set up the vars and bounding box
            let py = pos[[p, 0]] as f32;
            let px = pos[[p, 1]] as f32;
            let mut max_radius: f32 = 0.0;
            let mut vertices: Vec<(f32, f32)> = Vec::with_capacity(n_rays);
            let mut y_min = f32::MAX;
            let mut y_max = f32::MIN;
            let mut x_min = f32::MAX;
            let mut x_max = f32::MIN;
            (0..n_rays).for_each(|r| {
                // for each ray compute the angle and (y, x) position at that angle
                let d = dist[[p, r]];
                let angle = angle_step * r as f32;
                let y = py + d * angle.sin();
                let x = px + d * angle.cos();
                // update the bounding box (y_min, y_max, x_min, x_max) with max
                // and min best bounds
                y_min = y_min.min(y);
                y_max = y_max.max(y);
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                // add the vertex to vertices and update the max radius with the
                // current ray's distances
                vertices.push((y, x));
                max_radius = max_radius.max(d);
            });
            // compute the polygon area and create a new 2D NMS polygon
            let area = polygon_area(&vertices);
            NMSPolygon2D {
                bbox: (y_min, y_max, x_min, x_max),
                area: area,
                dist: max_radius,
                vertices: vertices,
            }
        })
        .collect();

    todo!("Implement NMS polygon building");
}

fn polygon_area(vertices: &[(f32, f32)]) -> f32 {
    todo!("Implement polygon area compute.");
}
