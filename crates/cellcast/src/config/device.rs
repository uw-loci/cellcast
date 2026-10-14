use std::sync::OnceLock;

use burn::Tensor;
use burn::tensor::Device;

pub(crate) static CPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) static GPU_DEVICE: OnceLock<Device> = OnceLock::new();
pub(crate) const CPU_INIT_FAIL_MSG: &str = "Failed to initialize the CPU.";
pub(crate) const GPU_INIT_FAIL_MSG: &str = "Failed to initialize the GPU.";
pub(crate) const DEVICE_RETRIEVE_FAIL_MSG: &str =
    "Failed to retrieve data from the initialized device (CPU or GPU).";

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

/// Read a Tensor and return the data as an owned Vec.
///
/// This function reads any Tensor and checks for out-of-memory (OOM) errors
/// exiting gracefully with less scary error message.
#[inline(always)]
pub(crate) fn read_tensor<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    match tensor.try_into_data() {
        Ok(data) => data
            .try_to_vec()
            .expect("Unable to copy raw tensor data into Vec."),
        Err(e) => {
            let reason = e.reason();
            let oom_err = reason
                .lines()
                .find(|l| l.contains("can't allocate buffer"))
                .map(|l| l.trim());
            match oom_err {
                Some(oe) => {
                    eprintln!("{DEVICE_RETRIEVE_FAIL_MSG} \nCause: {oe}");
                    std::process::exit(1);
                }
                None => {
                    eprintln!("{DEVICE_RETRIEVE_FAIL_MSG} \nCause: {reason}");
                    std::process::exit(1);
                }
            }
        }
    }
}
