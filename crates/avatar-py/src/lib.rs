//! avatar-py — Python bindings for the avatar pipeline (report §3.8, plan task 10).
//!
//! ```python
//! import live_avatar
//! engine = live_avatar.Engine(
//!     voice_model="voices/en_US-amy-medium.onnx",
//!     picture="portrait.png",   # optional; synthetic face otherwise
//!     size=256,                 # 64 | 128 | 256 | 512
//! )
//! engine.render("Hello world", "out.mp4")
//! ```
//!
//! Expression markers (`<happy>`/`<sad>`/...) are stripped (Plan 5, task 7):
//! expression is audio-driven.
//!
//! The render runs with the GIL released (PyO3 `allow_threads`), so Python
//! threads keep running while the talking-head inference + encoding happen.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper around [`avatar_core::pipeline::EngineConfig`] + [`render`].
#[pyclass]
struct Engine {
    model_path: std::path::PathBuf,
    picture: Option<std::path::PathBuf>,
    strength: f32,
    size: u32,
    face_crop: bool,
    face_crop_margin: f32,
    backend: Option<avatar_core::pipeline::BackendPref>,
    speech_speed: f32,
    fps: f32,
}

#[pymethods]
impl Engine {
    /// Creates an avatar engine.
    ///
    /// - `voice_model`: path to a Piper `.onnx` voice model
    ///   (the matching `.onnx.json` must sit next to it).
    /// - `picture`: optional avatar picture (PNG/JPEG); a synthetic
    ///   gradient face is used when omitted.
    /// - `size`: output size in px: 64, 128, 256, or 512 (default 256).
    /// - `strength`: expression warp strength (photos ~0.5, cartoons ~1.0).
    /// - `face_crop`: detect the face and center-crop before warping
    ///   (default True; falls back to the whole image if no face is found).
    /// - `face_crop_margin`: crop margin as a fraction of the face-box size
    ///   (default 0.35).
    /// - `backend`: backend for the talking-head path: "auto" (default; GPU
    ///   when available, CPU otherwise), "gpu", or "cpu".
    /// - `speech_speed`: speech speed multiplier (default 1.0 = the voice
    ///   model's native rate; < 1.0 = slower, > 1.0 = faster).
    /// - `fps`: output video frame rate (default 25.0). Wav2Lip was trained at
    ///   25 fps; other rates change the frame count and the mel-window stride.
    #[new]
    #[pyo3(signature = (voice_model, picture=None, size=256, strength=0.5, face_crop=true, face_crop_margin=0.35, backend="auto", speech_speed=1.0, fps=25.0))]
    fn new(
        voice_model: &str,
        picture: Option<&str>,
        size: u32,
        strength: f32,
        face_crop: bool,
        face_crop_margin: f32,
        backend: &str,
        speech_speed: f32,
        fps: f32,
    ) -> PyResult<Self> {
        if size != 64 && size != 128 && size != 256 && size != 512 {
            return Err(PyValueError::new_err(format!(
                "size must be 64, 128, 256, or 512, got {size}"
            )));
        }
        if !(speech_speed > 0.0 && speech_speed.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "speech_speed must be a positive finite number, got {speech_speed}"
            )));
        }
        if !(fps > 0.0 && fps.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "fps must be a positive finite number, got {fps}"
            )));
        }
        let model_path = std::path::PathBuf::from(voice_model);
        if !model_path.exists() {
            return Err(PyValueError::new_err(format!(
                "voice model not found: {voice_model}"
            )));
        }
        let backend = match avatar_core::pipeline::BackendPref::parse(backend) {
            Some(b) => b,
            None => {
                return Err(PyValueError::new_err(format!(
                    "backend must be 'auto', 'gpu', or 'cpu', got {backend}"
                )))
            }
        };
        Ok(Self {
            model_path,
            picture: picture.map(std::path::PathBuf::from),
            strength,
            size,
            face_crop,
            face_crop_margin,
            backend,
            speech_speed,
            fps,
        })
    }

    /// Renders `text` to an MP4 file. Expression markers (`<happy>`/`<mad>`/...)
    /// are stripped — expression is audio-driven. The GIL is released for the
    /// duration of the render.
    ///
    /// Returns the number of video frames written.
    #[pyo3(name = "render")]
    fn render_py(&self, py: Python<'_>, text: &str, out: &str) -> PyResult<usize> {
        let cfg = avatar_core::pipeline::EngineConfig {
            model_path: self.model_path.clone(),
            picture: self.picture.clone(),
            strength: self.strength,
            size: self.size,
            face_crop: self.face_crop,
            face_crop_margin: self.face_crop_margin,
            backend: self.backend,
            speech_speed: self.speech_speed,
            fps: self.fps,
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
