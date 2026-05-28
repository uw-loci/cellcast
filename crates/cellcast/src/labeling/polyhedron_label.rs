use imgal::ImgalError;
use imgal::spatial::geometry::inside_polyhedron;
use ndarray::{Array1, Array3, ArrayView1, ArrayView2, Axis};

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
        let z_min = bbox[0].max(0);
        let z_max = bbox[1].min(nz as i32 - 1);
        let y_min = bbox[2].max(0);
        let y_max = bbox[3].min(ny as i32 - 1);
        let x_min = bbox[4].max(0);
        let x_max = bbox[5].min(nx as i32 - 1);
        if z_min > z_max || y_min > y_max || x_min > x_max {
            return Ok(());
        }
        let cur_poly_verts = polyhedron_verts(cur_dist, cur_pnt, gs_verts.view());
        let cur_pnt = cur_pnt.mapv(|v| v as f32);
        (z_min as usize..=z_max as usize).for_each(|z| {
            (y_min as usize..=y_max as usize).for_each(|y| {
                (x_min as usize..=x_max as usize).for_each(|x| {
                    let pnt = [z, y, x];
                    if inside_polyhedron(
                        &cur_poly_verts,
                        &gs_faces,
                        cur_pnt.view(),
                        ArrayView1::from(&[z as f32, y as f32, x as f32]),
                        false,
                    )
                    .unwrap()
                    {
                        labels[pnt] = ids[i];
                    }
                })
            })
        });
        Ok(())
    })?;
    Ok(labels)
}
