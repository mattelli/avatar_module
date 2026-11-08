//! avatar-py — Python bindings for the avatar pipeline (report §3.8, plan task 10).
//!
//! ```python
//! import live_avatar
//! engine = live_avatar.Engine(
//!     voice_model="voices/en_US-amy-medium.onnx",
//!     picture="portrait.png",   # optional; synthetic face otherwise
//!     size=64,                  # 32 or 64 px
//! )
//! engine.render("Hello <happy> world", "out.mp4")
//! ```
//!
//! The render runs with the GIL released (PyO3 `allow_threads`), so Python
//! threads keep running while WebGPU compute + encoding happen.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper around [`avatar_core::pipeline::EngineConfig`] + [`render`].
#[pyclass]
struct Engine {
    model_path: std::path::PathBuf,
    picture: Option<std::path::PathBuf>,
    strength: f32,
    size: u32,
    breath_period_secs: f32,
}

#[pymethods]
impl Engine {
    /// Creates an avatar engine.
    ///
    /// - `voice_model`: path to a Piper `.onnx` voice model
    ///   (the matching `.onnx.json` must sit next to it).
    /// - `picture`: optional avatar picture (PNG/JPEG); a synthetic
    ///   gradient face is used when omitted.
    /// - `size`: output size in px, 32 (default) or 64.
    /// - `strength`: expression warp strength (photos ~0.5, cartoons ~1.0).
    /// - `breath_period_secs`: breathing cycle length (default 4 s).
    #[new]
    #[pyo3(signature = (voice_model, picture=None, size=32, strength=0.5, breath_period_secs=4.0))]
    fn new(
        voice_model: &str,
        picture: Option<&str>,
        size: u32,
        strength: f32,
        breath_period_secs: f32,
    ) -> PyResult<Self> {
        if size != 32 && size != 64 {
            return Err(PyValueError::new_err(format!(
                "size must be 32 or 64, got {size}"
            )));
        }
        let model_path = std::path::PathBuf::from(voice_model);
        if !model_path.exists() {
            return Err(PyValueError::new_err(format!(
                "voice model not found: {voice_model}"
            )));
        }
        Ok(Self {
            model_path,
            picture: picture.map(std::path::PathBuf::from),
            strength,
            size,
            breath_period_secs,
        })
    }

    /// Renders `text` (with optional `<happy>`/`<mad>`/... markers) to an
    /// MP4 file. The GIL is released for the duration of the render.
    ///
    /// Returns the number of video frames written.
    #[pyo3(name = "render")]
    fn render_py(&self, py: Python<'_>, text: &str, out: &str) -> PyResult<usize> {
        let cfg = avatar_core::pipeline::EngineConfig {
            model_path: self.model_path.clone(),
            picture: self.picture.clone(),
            strength: self.strength,
            size: self.size,
            breath_period_secs: self.breath_period_secs,
        };
        let out_path = std::path::PathBuf::from(out);
        // GIL released: the pipeline is pure native compute.
        let n = py
            .allow_threads(|| avatar_core::pipeline::render(text, out_path, &cfg))
            .map_err(|e| PyValueError::new_err(format!("render failed: {e:#}")))?;
        Ok(n)
    }

    /// Output size in px.
    #[getter]
    fn size(&self) -> u32 {
        self.size
    }
}

/// Returns the avatar-core version.
#[pyfunction]
fn version() -> String {
    avatar_core::placeholder::version().to_string()
}

#[pymodule]
fn live_avatar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<Engine>()?;
    Ok(())
}
