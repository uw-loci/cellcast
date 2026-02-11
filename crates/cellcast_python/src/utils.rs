use std::ffi::CString;

use pyo3::prelude::*;

/// Add a child module to Python's sys.modules dict.
///
/// # Description
///
/// This function manually adds a given module to Python's sys.modules
/// dict. This enables imports like `import cellcast.stardist_2d as star`.
///
/// # Arguments
///
/// * `module_name` - The name of the module to add to sys.modules.
pub fn py_import_module(module_name: &str) {
    let import_cmd = format!(
        "import sys; sys.modules['cellcast.{}'] = '{}'",
        module_name, module_name
    );
    let c_str_cmd =
        CString::new(import_cmd).expect("Failed to create 'CString' module import command.");
    Python::attach(|py| {
        py.run(c_str_cmd.as_c_str(), None, None).unwrap();
    });
}
