use burn::backend::{NdArray, Wgpu};

pub type CpuBackend<E, I> = NdArray<E, I>;
pub type GpuBackend<E, I> = Wgpu<E, I>;
