use std::f32::consts::TAU;

use ndarray::{Array2, Array3, ArrayView1, ArrayView2, Axis};

/// Convert radial polygon representation into a 2-dimensional label image.
///
/// # Description
///
/// Converts radial polygons representation in polar coordinates (*i.e.* the
/// radial distances from the center points) into 2-dimensional label images
/// where each label is assigned a unique integer label. Low probability
/// polygons are rendered first with higher probability polygons overwritting
/// lower ones.
///
/// # Arguments
///
/// * `polygon_dist`: A 2D array of radial polygon distances with shape
///   `(n_polys, n_rays)`.
/// * `polygon_prob`: A 1D array of polygon probabilites with shape
///   `(n_polys,)`.
/// * `polygon_pos`: A 2D array of polygon center positions with shape
///   `(n_polys, 2)`. The dimension order expected is (row, col).
/// * `shape`: The shape of the output label image.
/// * `scale`: Optional scaling factor for each axis. If `None` then no scaling
///   is applied.
///
/// # Returns
///
/// * `Array2<u16>`: The 2-dimensional label image where the background pixels
///   are labeled as `0` and polygons labeled with range `(1..n_polys)`.
pub fn radial_polygon_to_label_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_prob: ArrayView1<f32>,
    polygon_pos: ArrayView2<usize>,
    shape: (usize, usize),
    scale: Option<(f32, f32)>,
) -> Array2<u16> {
    // sort valid polygon indices in ascending order (higher prob drawn last)
    // and select these polygons based on new prob order
    let (n_polys, n_rays) = polygon_dist.dim();
    let mut sorted_inds: Vec<usize> = (0..n_polys).collect();
    sorted_inds.sort_by(|&a, &b| polygon_prob[a].partial_cmp(&polygon_prob[b]).unwrap());
    let poly_ax = Axis(0);
    let polygon_dist = polygon_dist.select(poly_ax, &sorted_inds);
    let polygon_pos = polygon_pos.select(poly_ax, &sorted_inds);

    // convert radial distances and point positions from polar to cartesian
    // coordinates and render the label image with original indices as labels
    let poly_coords = radial_dist_to_coords_2d(
        polygon_dist.view(),
        polygon_pos.view(),
        n_polys,
        n_rays,
        scale,
    );

    // create output label image and render the polygons
    let label_ids: Vec<u16> = (0..n_polys).map(|p| (p + 1) as u16).collect();
    let mut labels = Array2::<u16>::zeros(shape);
    (0..n_polys).zip(label_ids.iter()).for_each(|(p, &l)| {
        let poly_rows: Vec<f32> = (0..n_rays).map(|r| poly_coords[[p, r, 0]]).collect();
        let poly_cols: Vec<f32> = (0..n_rays).map(|r| poly_coords[[p, r, 1]]).collect();
        let (raster_row, raster_col) = render_polygon_2d(&poly_rows, &poly_cols, n_rays, shape);
        (0..raster_row.len()).for_each(|j| {
            labels[[raster_row[j], raster_col[j]]] = l;
        });
    });

    labels
}

/// Check if the query point is inside the given polygon using ray casting.
///
/// # Arguments
///
/// * `row`: The "y" coordinate of a polygon.
/// * `col`: The "x" coordinate of a polygon.
/// * `size`: The size of the polygon.
/// * `row_coords`: A slice of row coordinates for a polygon.
/// * `col_coords`: A slice of col coordinates for a polygon.
///
/// # Returns
///
/// * `bool`: Returns `true`, if `row` and `col` are inside the polygon. Returns
///   `false` if they are not.
#[inline]
fn inside_polygon(
    row: usize,
    col: usize,
    size: usize,
    row_coords: &[f32],
    col_coords: &[f32],
) -> bool {
    let mut inside = false;
    let row = row as f32;
    let col = col as f32;
    let mut j = size - 1;
    (0..size).for_each(|i| {
        let (xi, yi) = (col_coords[i], row_coords[i]);
        let (xj, yj) = (col_coords[j], row_coords[j]);
        if ((yi > row) != (yj > row)) && (col < (xj - xi) * (row - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    });

    inside
}

/// Convert polar distance representation to Cartesian coordinates.
///
/// # Arguments
///
/// * `polygon_dist`: A 2D array of radial polygon distances with shape
///   `(n_polys, n_rays)`.
/// * `polygon_prob`: A 1D array of polygon probabilites with shape
///   `(n_polys,)`.
/// * `n_polys`: The number of polygons.
/// * `n_rays`: The number of ray angles.
/// * `scale`: The scaling factor per axis. If `None` then no scaling is
///   applied.
///
/// # Returns
///
/// * `Array3<f32>`: The polar distance coordinates in Cartesian form. The
///   output array has shape `(p, r, D)`. Where `p` is the polygon, `r` is the
///   ray and `D` is the dimension.
#[inline]
fn radial_dist_to_coords_2d(
    polygon_dist: ArrayView2<f32>,
    polygon_pos: ArrayView2<usize>,
    n_polys: usize,
    n_rays: usize,
    scale: Option<(f32, f32)>,
) -> Array3<f32> {
    let scale = scale.unwrap_or((1.0, 1.0));

    // get evenly spaced angles for 0 to 2*pi (TAU) and compute row (Y), col (X)
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

/// Render/draw a polygon from a set of polygon cartesian coordinates.
///
/// # Arguments
///
/// * `row_coords`: A slice of row coordinates for the polygon.
/// * `col_coords`: A slice of col coordinates for the polygon.
/// * `size`: The size of the polygon (*i.e.* the number of points).
/// * `shape`: The shape of the output image where the polygon is drawn.
///
/// # Returns
///
/// * `(Vec<usize>, Vec<usize>)`: A tuple of row and col vectors indicating
///   which pixels are valid for polygon drawing.
#[inline]
fn render_polygon_2d(
    row_coords: &[f32],
    col_coords: &[f32],
    size: usize,
    shape: (usize, usize),
) -> (Vec<usize>, Vec<usize>) {
    // find the bounding box clipped to the image bounds
    let min_row = row_coords
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(0.0) as usize;
    let max_row = row_coords
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .min((shape.0 - 1) as f32) as usize;
    let min_col = col_coords
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(0.0) as usize;
    let max_col = col_coords
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .min((shape.1 - 1) as f32) as usize;

    // determine if pixels within the bounding box are within the polygon
    // if pixel in question is within the polygon, save the coordinates
    let mut raster_row: Vec<usize> = Vec::new();
    let mut raster_col: Vec<usize> = Vec::new();
    (min_row..=max_row)
        .flat_map(|y| (min_col..=max_col).map(move |x| (y, x)))
        .filter(|&(y, x)| inside_polygon(y, x, size, &row_coords, &col_coords))
        .for_each(|(y, x)| {
            raster_row.push(y);
            raster_col.push(x)
        });

    (raster_row, raster_col)
}
