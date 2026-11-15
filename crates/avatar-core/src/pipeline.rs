//! Pipeline glue (report §3, plan task 9).
//!
//! Wires the whole pipeline: text → markers → TTS → talking-head
//! (Wav2Lip + ExpNet) → encode → `.mp4`.
//!
//! Expression is audio-driven (Plan 5): markers are stripped (task 7). The
//! talking-head backend (Plan 5) is the only backend — the classic atlas/wgpu
//! path is removed (Plan 6).

use crate::encoder;
use crate::markers::{self, Span};
use crate::tts::{EmbeddedPiper, SynthResult, Synthesizer};
use anyhow::Context;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for the avatar engine (talking-head backend only).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Voice model path (`model.onnx` + `model.onnx.json`).
    pub model_path: PathBuf,
    /// Avatar picture (PNG/JPEG); a synthetic gradient face if `None`.
    pub picture: Option<PathBuf>,
    /// Warp strength (photos ~0.5, cartoons ~1.0).
    pub strength: f32,
    /// Output size: 64, 128, 256, or 512 px (default 256).
    pub size: u32,
    /// Crop to the detected face before warping (plan 4).
    pub face_crop: bool,
    /// Face-crop margin as a fraction of the face-box size (default 0.35).
    pub face_crop_margin: f32,
    /// Backend override for the talking-head path (GPU/CPU/Auto). `None` =
    /// auto-select (GPU when available, CPU otherwise).
    pub backend: Option<BackendPref>,
    /// Optional Audio2Pose latent seed (Plan 6). `None` = stochastic head
    /// motion (a fresh latent `z ~ N(0,1)` per render); `Some(seed)` =
    /// reproducible head motion.
    pub pose_seed: Option<u64>,
    /// Speech speed multiplier (default `1.0`). `< 1.0` = slower, `> 1.0` =
    /// faster. Mapped to Piper's `length_scale` as `1 / speech_speed`.
    pub speech_speed: f32,
    /// Output video frame rate (default 25.0). Wav2Lip was trained at 25 fps;
    /// other rates change the frame count and the mel-window stride.
    pub fps: f32,
}

/// Backend selection for the talking-head path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPref {
    /// Force the GPU (burn-wgpu).
    Gpu,
    /// Force the CPU (burn-flex).
    Cpu,
}

impl BackendPref {
    /// Parses `"auto"` -> None, `"gpu"`/`"cpu"` -> the preference.
    pub fn parse(s: &str) -> Option<Option<Self>> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(None),
            "gpu" | "cuda" | "vulkan" | "wgpu" => Some(Some(Self::Gpu)),
            "cpu" | "flex" | "ndarray" => Some(Some(Self::Cpu)),
            _ => None,
        }
    }

    /// Maps to the expnet backend preference.
    pub fn to_expnet(self) -> expnet::backend::BackendPreference {
        match self {
            Self::Gpu => expnet::backend::BackendPreference::Gpu,
            Self::Cpu => expnet::backend::BackendPreference::Cpu,
        }
    }
}

impl EngineConfig {
    /// Config from a voice model with defaults (256 px, synthetic face).
    pub fn voice(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            picture: None,
            strength: 0.5,
            size: crate::encoder::DEFAULT_SIZE,
            face_crop: true,
            face_crop_margin: crate::face_crop::DEFAULT_FACE_CROP_MARGIN,
            backend: None,
            pose_seed: None,
            speech_speed: 1.0,
            fps: crate::analysis::FPS,
        }
    }

    /// Uses an avatar picture instead of the synthetic face.
    pub fn with_picture(mut self, picture: impl Into<PathBuf>) -> Self {
        self.picture = Some(picture.into());
        self
    }

    /// Output size (64, 128, 256, or 512 px). Returns an error for other sizes.
    pub fn with_size(mut self, size: u32) -> anyhow::Result<Self> {
        validate_size(size)?;
        self.size = size;
        Ok(self)
    }

    /// Warp strength for the expression warp.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    /// Enables/disables face detection + center-crop (default enabled).
    pub fn with_face_crop(mut self, enabled: bool) -> Self {
        self.face_crop = enabled;
        self
    }

    /// Sets the face-crop margin as a fraction of the face-box size.
    pub fn with_face_crop_margin(mut self, margin: f32) -> Self {
        self.face_crop_margin = margin;
        self
    }

    /// Forces the backend for the talking-head path (GPU/CPU/Auto).
    pub fn with_backend(mut self, backend: BackendPref) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Sets the Audio2Pose latent seed (Plan 6). `None` = stochastic head
    /// motion; `Some(seed)` = reproducible head motion.
    pub fn with_pose_seed(mut self, seed: Option<u64>) -> Self {
        self.pose_seed = seed;
        self
    }

    /// Sets the speech speed multiplier: `< 1.0` = slower, `> 1.0` = faster,
    /// `1.0` = the voice model's native rate (default). Must be positive.
    pub fn with_speech_speed(mut self, speed: f32) -> Self {
        self.speech_speed = speed;
        self
    }

    /// Sets the output video frame rate (default 25.0). Wav2Lip was trained at
    /// 25 fps; other rates change the frame count and the mel-window stride.
    /// Must be positive.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }
}

/// One synthesized span.
#[derive(Debug, Clone)]
struct SpanSynth {
    result: SynthResult,
}

/// Synthesizes each span separately and concatenates them into a clip.
fn synthesize_spans(
    tts: &mut EmbeddedPiper,
    spans: &[Span],
) -> anyhow::Result<Vec<SpanSynth>> {
    let mut out = Vec::new();
    for span in spans {
        let result = tts
            .synthesize(&span.text)
            .with_context(|| format!("synthesize {:?}", span.text))?;
        out.push(SpanSynth { result });
    }
    Ok(out)
}

/// Concatenates the per-span samples into one clip.
fn concat_samples(synth: &[SpanSynth]) -> (Vec<f32>, u32) {
    let mut samples = Vec::new();
    let sr = synth
        .first()
        .map(|s| s.result.sample_rate)
        .unwrap_or(crate::analysis::FPS as u32);
    for s in synth {
        samples.extend_from_slice(&s.result.samples);
    }
    (samples, sr)
}

/// Validates an output size against [`crate::encoder::SUPPORTED_SIZES`].
pub fn validate_size(size: u32) -> anyhow::Result<()> {
    anyhow::ensure!(
        crate::encoder::SUPPORTED_SIZES.contains(&size),
        "unsupported output size {size}; supported sizes are {}",
        crate::encoder::SUPPORTED_SIZES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

/// Renders the full clip: text → markers → TTS → talking-head → encode → `.mp4`.
///
/// Returns the number of video frames written.
pub fn render(text: &str, out: impl AsRef<Path>, cfg: &EngineConfig) -> anyhow::Result<usize> {
    render_with_stats(text, out, cfg).map(|(n, _)| n)
}

/// As [`render`], but also returns wall-clock render time.
pub fn render_with_stats(
    text: &str,
    out: impl AsRef<Path>,
    cfg: &EngineConfig,
) -> anyhow::Result<(usize, Duration)> {
    let t0 = std::time::Instant::now();
    let out = out.as_ref();

    // 0. Validate the output size.
    validate_size(cfg.size)?;

    // 1. Markers (stripped; expression is audio-driven).
    let spans = markers::parse(text);
    anyhow::ensure!(!spans.is_empty(), "no text to render");

    // 2. TTS (per span). `speech_speed` maps to Piper's `length_scale`
    //    (inverse of speed): 1.0 = native, <1.0 = slower, >1.0 = faster.
    anyhow::ensure!(
        cfg.speech_speed > 0.0 && cfg.speech_speed.is_finite(),
        "speech_speed must be a positive finite number, got {}",
        cfg.speech_speed
    );
    let length_scale = if (cfg.speech_speed - 1.0).abs() < f32::EPSILON {
        None // model default
    } else {
        Some(1.0 / cfg.speech_speed)
    };
    let mut tts = EmbeddedPiper::new(&cfg.model_path)
        .with_context(|| format!("load voice {}", cfg.model_path.display()))?
        .with_length_scale(length_scale);
    let synth = synthesize_spans(&mut tts, &spans)?;
    let (samples, sr) = concat_samples(&synth);
    anyhow::ensure!(!samples.is_empty(), "no audio synthesized");

    // 3. Talking-head path (Plan 5, tasks 8-9): Wav2Lip + ExpNet fused
    //    frames at the configured `cfg.size` output.
    anyhow::ensure!(
        cfg.fps > 0.0 && cfg.fps.is_finite(),
        "fps must be a positive finite number, got {}",
        cfg.fps
    );
    let fcfg = crate::fusion::FusionConfig::default()
        .with_size(cfg.size)
        .with_strength(cfg.strength);
    let pref = cfg
        .backend
        .map(BackendPref::to_expnet)
        .unwrap_or(expnet::backend::BackendPreference::Auto);
    let clip = crate::talking_head::render_talking_head(
        &samples,
        sr,
        cfg.picture.as_deref(),
        &fcfg,
        cfg.pose_seed, // Audio2Pose latent seed (Plan 6, task 7)
        pref,
        cfg.fps,
    )
    .context("talking-head render")?;
    anyhow::ensure!(!clip.is_empty(), "talking-head produced no frames");

    // 4. Encode + mux.
    encoder::encode_and_mux(out, &clip, &samples, sr, 1, cfg.fps)?;

    Ok((clip.len(), t0.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolves `.res` by walking up from the crate root (cargo test runs
    /// with CWD = crates/avatar-core, where `.res` does not exist).
    fn res_dir() -> Option<PathBuf> {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let candidate = dir.join(".res");
            if candidate.is_dir() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    /// Path to the bundled test voice model, if present.
    fn test_model() -> Option<PathBuf> {
        let dir = res_dir()?;
        let p = dir.join("en_US-hfc_female-medium.onnx");
        p.exists().then_some(p)
    }

    /// T9: end-to-end — `render("Hello world", out)` → MP4 exists,
    /// duration ≈ audio duration ± 100 ms.
    #[test]
    fn end_to_end_render() {
        let Some(model) = test_model() else { return };
        let tmp = std::env::temp_dir().join("avatar_t9_test.mp4");
        let cfg = EngineConfig::voice(&model);
        let (frames, _) = render_with_stats("Hello world", &tmp, &cfg).unwrap();
        assert!(frames > 0);

        let f = std::fs::File::open(&tmp).unwrap();
        let size = f.metadata().unwrap().len();
        assert!(size > 1000, "mp4 too small: {size}");
        let mut reader = std::io::BufReader::new(f);
        let parsed = mp4::Mp4Reader::read_header(&mut reader, size).unwrap();
        assert_eq!(parsed.tracks().len(), 2, "expected video + audio tracks");

        // Capture track IDs first (the reader is moved below).
        let audio_id = parsed
            .tracks()
            .values()
            .find(|t| t.track_type().map(|tt| tt == mp4::TrackType::Audio).unwrap_or(false))
            .unwrap()
            .track_id();
        let video_id = parsed
            .tracks()
            .values()
            .find(|t| t.track_type().map(|tt| tt == mp4::TrackType::Video).unwrap_or(false))
            .unwrap()
            .track_id();

        // Video duration ≈ audio duration ± 100 ms.
        let audio_t = parsed.tracks().get(&audio_id).unwrap();
        let video_t = parsed.tracks().get(&video_id).unwrap();
        let audio_ms = audio_t.duration().as_millis() as f32;
        let video_ms = video_t.duration().as_millis() as f32;
        assert!(
            (video_ms - audio_ms).abs() <= 100.0,
            "video {video_ms} ms vs audio {audio_ms} ms"
        );

        // Regression (PLAN_FIX_RENDER_ISSUES): one MP4 video sample per
        // rendered frame. The old NAL-count heuristic produced misaligned
        // samples -> flickering video.
        assert_eq!(
            video_t.sample_count() as usize,
            frames,
            "video sample count must equal rendered frame count"
        );

        // Audio track must be non-silent when the AAC encoder is enabled.
        #[cfg(feature = "aac")]
        {
            let mut parsed = parsed;
            let n = parsed.sample_count(audio_id).unwrap();
            assert!(n > 0, "audio track has no samples");
            let mut any_nonzero = false;
            for i in 1..=n {
                if let Some(sample) = parsed.read_sample(audio_id, i).unwrap() {
                    if sample.bytes.iter().any(|&b| b != 0) {
                        any_nonzero = true;
                        break;
                    }
                }
            }
            assert!(any_nonzero, "audio track is silent (AAC feature on)");
        }

        let _ = std::fs::remove_file(&tmp);
    }

    /// T3: with `face_crop=true` and a detectable face, the atlas source is
    /// smaller than the input and centered on the face; with `face_crop=false`
    /// the whole image is used.
    #[test]
    fn face_crop_affects_atlas_source() {
        let Some(res) = res_dir() else { return };
        let photo = res.join("avatar.jpg");
        if !photo.exists() {
            return;
        }
        let img = crate::atlas::load_image(&photo).unwrap();
        let (iw, ih) = img.dimensions();

        // Enabled: crop to the face (smaller than input).
        let cropped = crate::face_crop::apply_face_crop(img.clone(), true, 0.35);
        let (cw, ch) = cropped.dimensions();
        assert!(
            cw < iw || ch < ih,
            "face_crop=true should shrink the source ({cw}x{ch} vs {iw}x{ih})"
        );

        // Disabled: whole image.
        let whole = crate::face_crop::apply_face_crop(img, false, 0.35);
        assert_eq!(whole.dimensions(), (iw, ih));
    }

    /// T7: marker parser now strips markers into a single neutral span.
    #[test]
    fn parser_integration() {
        let spans = markers::parse("<happy> hi <xyz> <sad> a <mad> b");
        assert_eq!(spans.len(), 1);
        // Each `<tag>` (with its leading space) is removed, leaving the
        // surrounding whitespace: " hi " + " a " + " b" with the markers'
        // spaces preserved.
        assert_eq!(spans[0].text, " hi   a  b");
    }

    /// Output sizes: 64/128/256/512 are valid; others are rejected.
    #[test]
    fn validate_size_accepts_supported() {
        for &s in &crate::encoder::SUPPORTED_SIZES {
            assert!(validate_size(s).is_ok(), "{s} should be supported");
        }
    }

    #[test]
    fn validate_size_rejects_unsupported() {
        for &s in &[0u32, 32, 100, 1280] {
            assert!(validate_size(s).is_err(), "{s} should be rejected");
        }
    }

    /// `with_size` returns an error for unsupported sizes.
    #[test]
    fn with_size_rejects_invalid() {
        let model = PathBuf::from("unused.onnx");
        assert!(EngineConfig::voice(&model).with_size(64).is_ok());
        assert!(EngineConfig::voice(&model).with_size(256).is_ok());
        assert!(EngineConfig::voice(&model).with_size(32).is_err());
    }
}
