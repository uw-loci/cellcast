use imgal::spatial::geometry::tetrahedron_volume;
use ndarray::{ArrayView1, ArrayView2};

/// Compute the volume of a polyhedron.
///
/// # Description
///
/// todo
///
/// # Arguments
///
/// * `distances`: The polyhedron distances.
/// * `gs_vertices`: The "Golden Spiral" unit sphere vertices with shape
///   `(n_points, 3)`.
/// * `gs_faces`: The "Golden Spiral" unit sphere face indices with shape
///   `(n_triangles, 3)`.
///
/// # Returns
///
/// * `f32`: The volume of the polyhedron.
pub fn polyhedron_volume(
    distances: ArrayView1<f32>,
    gs_vertices: ArrayView2<f64>,
    gs_faces: ArrayView2<usize>,
) -> f32 {
    let origin = vec![0.0_f32; 3];
    let n_faces = gs_faces.dim().0;
    (0..n_faces).fold(0.0_f32, |acc, f| {
        let tri = gs_faces.row(f);
        let a: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[0];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let b: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[1];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let c: Vec<f32> = (0..3)
            .map(|i| {
                let ti = tri[2];
                distances[ti] * gs_vertices[[ti, i]] as f32
            })
            .collect();
        let v = tetrahedron_volume(&a, &b, &c, &origin) as f32;
        acc + v
    })
}
