// Including this hook allows toplevel builds on macOS -- see:
// https://pyo3.rs/v0.29.2/building-and-distribution.html#macos

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
