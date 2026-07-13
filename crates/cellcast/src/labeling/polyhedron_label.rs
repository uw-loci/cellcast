use imgal::prelude::*;
use imgal::spatial::convex_hull::quickhull_3d;
use imgal::spatial::geometry::inside_polyhedron;
use imgal::spatial::halfspace::{hull_to_halfspace, inside_halfspace_interior};
use ndarray::{Array1, Array3, ArrayView1, ArrayView2, Axis, indices};
use rayon::prelude::*;

use crate::geometry::polyhedron::{golden_spiral, polyhedron_bbox, polyhedron_verts};

/// TODO
pub fn distance_polyhedron_to_label(
    polyhedron_dist: ArrayView2<f32>,
    polyhedron_pnts: ArrayView2<f32>,
    polyhedron_prob: ArrayView1<f32>,
    prob_threshold: f32,
    anisotropy: [f64; 3],
    shape: [usize; 3],
) -> Result<Array3<u64>, ImgalError> {
    let n_polys = polyhedron_dist.dim().0;
    let n_rays = polyhedron_dist.dim().1;
    let mut labels = Array3::<u64>::zeros(shape);
    let ids: Array1<u64> = (1..n_polys + 1).map(|i| i as u64).collect();
    let inds: Vec<usize> = polyhedron_prob
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v >= prob_threshold)
        .map(|(i, _)| i)
        .collect();
    if inds.is_empty() {
        return Ok(labels);
    }
    let poly_ax = Axis(0);
    let dist = polyhedron_dist.select(poly_ax, &inds);
    let pnts = polyhedron_pnts.select(poly_ax, &inds);
    let prob = polyhedron_prob.select(poly_ax, &inds);
    let ids = ids.select(poly_ax, &inds);
    let mut sorted_inds: Vec<usize> = (0..prob.len()).collect();
    sorted_inds.sort_by(|&a, &b| prob[b].partial_cmp(&prob[a]).unwrap());
    let dist = dist.select(poly_ax, &sorted_inds);
    let pnts = pnts.select(poly_ax, &sorted_inds);
    let ids = ids.select(poly_ax, &sorted_inds);
    // next we render each label inside its bounding box
    let (gs_verts, gs_faces) = golden_spiral(n_rays, Some(anisotropy))?;
    let n_polys = dist.dim().0;
    let [nz, ny, nx] = shape;
    (0..n_polys).try_for_each(|i| -> Result<(), ImgalError> {
        let cur_dist = dist.row(i);
        let cur_pnt = pnts.row(i);
        let bbox = polyhedron_bbox(cur_dist, cur_pnt, gs_verts.view());
        let z_min = bbox[0].max(0) as usize;
        let z_max = bbox[1].min(nz as i32 - 1) as usize;
        let y_min = bbox[2].max(0) as usize;
        let y_max = bbox[3].min(ny as i32 - 1) as usize;
        let x_min = bbox[4].max(0) as usize;
        let x_max = bbox[5].min(nx as i32 - 1) as usize;
        if z_min > z_max || y_min > y_max || x_min > x_max {
            return Ok(());
        }
        let cur_poly_verts = polyhedron_verts(cur_dist, cur_pnt, gs_verts.view());
        let cur_pnt = cur_pnt.mapv(|v| v as f32);
        let hull = quickhull_3d(&cur_poly_verts, None)?;
        let convex_hs = hull_to_halfspace(&hull.0, &hull.1, None)?;
        let kernel_hs = hull_to_halfspace(&cur_poly_verts, &gs_faces, None)?;
        let z_len = (z_max - z_min) + 1;
        let y_len = (y_max - y_min) + 1;
        let x_len = (x_max - x_min) + 1;
        let hits = indices((z_len, y_len, x_len)).into_iter().par_bridge().try_fold(
            || Vec::new(),
            |mut acc, (bz, by, bx)| -> Result<Vec<([usize; 3], u64)>, ImgalError> {
                let pnt = [bz + z_min, by + y_min, bx + x_min];
                if inside_halfspace_interior(&kernel_hs, &pnt, true, None)? {
                    acc.push((pnt, ids[i]));
                    return Ok(acc);
                }
                if inside_halfspace_interior(&convex_hs, &pnt, true, None)?
                    && inside_polyhedron(
                        &cur_poly_verts,
                        &gs_faces,
                        cur_pnt.view(),
                        ArrayView1::from(&[pnt[0] as f32, pnt[1] as f32, pnt[2] as f32]),
                        None,
                    )?
                {
                    acc.push((pnt, ids[i]));
                }
                Ok(acc)
            },
        )
        .try_reduce(
            || Vec::new(),
            |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            }
        )?;
        hits.iter().for_each(|&(p, i)| {
            labels[p] = i;
        });
        Ok(())
    })?;
    Ok(labels)
}
