use std::sync::OnceLock;

use burn::tensor::Device;

pub(crate) static CPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();

/// Initialize or fetch the CPU device.
///
/// This function initializes or fetches thte CPU device.
#[inline(always)]
pub(crate) fn init_cpu() {
    CPU_DEVICE.get_or_init(|| {
        Device::flex()
    });
}

/// Initialize or fetch the GPU device.
///
/// This function initializes or fetches the GPU device.
#[inline(always)]
pub(crate) fn init_gpu() {
    GPU_DEVICE.get_or_init(|| {
        Device::wgpu(Default::default())
    });
}
