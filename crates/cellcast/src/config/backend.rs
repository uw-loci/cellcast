use std::sync::OnceLock;

use burn::tensor::Device;

pub(crate) static CPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) const CPU_INIT_FAIL_MSG: &str = "Failed to initialize the CPU.";
pub(crate) const GPU_INIT_FAIL_MSG: &str = "Failed to initialize the GPU.";
pub(crate) const CPU_RETRIEVE_FAIL_MSG: &str = "Failed to retrieve data from the CPU.";
pub(crate) const GPU_RETRIEVE_FAIL_MSG: &str = "Failed to retrieve data from the GPU.";

/// Initialize or fetch the CPU device.
///
/// This function initializes or fetches thte CPU device.
#[inline(always)]
pub(crate) fn init_cpu() {
    CPU_DEVICE.get_or_init(|| Device::flex());
}

/// Initialize or fetch the GPU device.
///
/// This function initializes or fetches the GPU device.
#[inline(always)]
pub(crate) fn init_gpu() {
    GPU_DEVICE.get_or_init(|| Device::wgpu(Default::default()));
}
