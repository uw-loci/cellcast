use burn::backend::{Flex, Wgpu};

pub(crate) type CpuBackend<E, I> = Flex<E, I>;
pub(crate) type GpuBackend<E, I> = Wgpu<E, I>;
