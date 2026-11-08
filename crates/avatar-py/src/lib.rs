//! avatar-py — Python bindings for the avatar pipeline.
//!
//! The `Engine` class (task 10) exposes `render(text, out)` to Python.
//! For now this is a placeholder module so the workspace builds.

use pyo3::prelude::*;

/// Placeholder function so the crate compiles before the Engine lands.
#[pyfunction]
fn version() -> String {
    avatar_core::placeholder::version().to_string()
}

#[pymodule]
fn live_avatar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
