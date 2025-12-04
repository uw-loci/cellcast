use ndarray::{ArrayViewMutD, Axis, Slice};

/// Clip the board of an n-dimensional boolean array.
///
/// # Description
///
/// Clip the board (_i.e._ set to `false`) the border of an n-dimensional array
/// symmetrically. This does not change the shape of the input array.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array.
/// * `size`: The number of elements to clip (_i.e._ set to false`) from the
///   input array.
pub fn clip_mask_border(data: &mut ArrayViewMutD<bool>, size: usize) {
    let shape = data.shape().to_vec();
    shape.iter().enumerate().for_each(|(i, &s)| {
        // slice the end of each axis
        let mut end_clip = data.slice_axis_mut(Axis(i), Slice::from((s - size)..s));
        end_clip.fill(false);
        // slice the start of each axis
        let mut start_clip = data.slice_axis_mut(Axis(i), Slice::from(0..size));
        start_clip.fill(false);
    });
}
