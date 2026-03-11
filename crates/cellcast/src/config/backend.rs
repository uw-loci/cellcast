use burn::backend::{NdArray, Wgpu};

pub(crate) type CpuBackend<E, I> = NdArray<E, I>;
pub(crate) type GpuBackend<E, I> = Wgpu<E, I>;
