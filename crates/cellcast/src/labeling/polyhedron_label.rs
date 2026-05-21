use imgal::ImgalError;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::inside_polyhedron;
use imgal::spatial::halfspace::{hull_to_halfspace, inside_halfspace_interior};
use ndarray::{Array1, Array3, ArrayView1, ArrayView2, Axis};

use crate::geometry::polyhedron::{golden_spiral, polyhedron_bbox, polyhedron_verts};

/// TODO
pub fn distance_polyhedron_to_label(
    polyhedron_dist: ArrayView2<f32>,
    polyhedron_pnts: ArrayView2<usize>,
    polyhedron_prob: ArrayView1<f32>,
    prob_threshold: f32,
    shape: [usize; 3],
) -> Result<Array3<u64>, ImgalError> {
    let n_polys = polyhedron_dist.dim().0;
    let n_rays = polyhedron_dist.dim().1;
    // here we are filtering the distances and centers to select polyhedra to
    // render
    let ids: Array1<u64> = (1..n_polys + 1).map(|i| i as u64).collect();
    let mut inds: Vec<usize> = polyhedron_prob
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v >= prob_threshold)
        .map(|(i, _)| i)
        .collect();
    let poly_ax = Axis(0);
    let dist = polyhedron_dist.select(poly_ax, &inds);
    let pnts = polyhedron_pnts.select(poly_ax, &inds);
    let prob = polyhedron_prob.select(poly_ax, &inds);
    let ids = ids.select(poly_ax, &inds);
    inds.sort_by(|&a, &b| prob[b].partial_cmp(&prob[a]).unwrap());
    let dist = dist.select(poly_ax, &inds);
    let pnts = pnts.select(poly_ax, &inds);
    let ids = ids.select(poly_ax, &inds);
    // each label is rendered inside it's bounding box
    let (gs_verts, gs_faces) = golden_spiral(n_rays, None)?;
    let [nz, ny, nx] = shape;
    let mut labels = Array3::<u64>::zeros(shape);
    (0..n_polys).try_for_each(|i| -> Result<(), ImgalError> {
        let cur_dist = dist.row(i);
        let cur_pnt = pnts.row(i);
        let [z_min, z_max, y_min, y_max, x_min, x_max] =
            polyhedron_bbox(cur_dist, cur_pnt, gs_verts.view());
        let cur_poly_verts = polyhedron_verts(cur_dist, cur_pnt, gs_verts.view());
        let cur_convex = quickhull_3d(&cur_poly_verts, false)?;
        let hs_convex = hull_to_halfspace(&cur_convex.0, &cur_convex.1, false)?;
        let hs_kernel = hull_to_halfspace(&cur_poly_verts, &gs_faces, false)?;
        let cur_pnt = cur_pnt.mapv(|v| v as f32);
        (z_min.max(0)..z_max.min(nz - 1)).for_each(|z| {
            (y_min.max(0)..y_max.min(ny - 1)).for_each(|y| {
                (x_min.max(0)..x_max.min(nx - 1)).for_each(|x| {
                    let pnt = [z, y, x];
                    // SAFE: these unwraps are safe because we have validated the inputs
                    if inside_halfspace_interior(&hs_kernel, &pnt, true, false).unwrap() {
                        labels[pnt] = ids[i];
                        return;
                    } else if !inside_halfspace_interior(&hs_convex, &pnt, true, false).unwrap() {
                        return;
                    } else if inside_polyhedron(
                        &cur_poly_verts,
                        &gs_faces,
                        cur_pnt.view(),
                        ArrayView1::from(&[z as f32, y as f32, x as f32]),
                        false,
                    )
                    .unwrap()
                    {
                        labels[pnt] = ids[i];
                        return;
                    } else {
                        return;
                    }
                })
            })
        });
        Ok(())
    })?;
    Ok(labels)
}
