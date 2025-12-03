/// Get the pad value needed to make an axis divisiable.
///
/// # Description
///
/// Compute the needed pad value to make an axis divisible by the
/// `div` value.
///
/// # Arguments
///
/// * `axis_len`: The length of a given axis.
/// * `div`: The division value.
///
/// # Returns
///
/// * `usize`: The pad value needed to make `axis_len` divisable by `div`.
#[inline]
pub fn divisible_pad(axis_len: usize, div: usize) -> usize {
    (div - axis_len % div) % div
}
