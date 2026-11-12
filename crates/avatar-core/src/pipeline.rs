//! Pipeline glue (report §3, plan task 9).
//!
//! Wires the whole pipeline: text → markers → TTS → analysis → visemes →
//! render → encode → `.mp4`.
//!
//! Expression is audio-driven (Plan 5): markers are stripped (task 7), and
//! the renderer always uses the neutral atlas band — the old expression
//! timeline / band switching is removed.

use crate::analysis::{self, FPS};
use crate::encoder::{self, DEFAULT_SIZE};
use crate::markers::{self, Span};
use crate::tts::{EmbeddedPiper, SynthResult, Synthesizer};
use crate::visemes::viseme_frames;
use anyhow::Context;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Configuration for the avatar engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Voice model path (`model.onnx` + `model.onnx.json`).
    pub model_path: PathBuf,
    /// Avatar picture (PNG/JPEG); a synthetic gradient face if `None`.
    pub picture: Option<PathBuf>,
    /// Warp strength (photos ~0.5, cartoons ~1.0).
    pub strength: f32,
    /// Output size: 32, 64, or 128 px.
    pub size: u32,
    /// Breathing period in seconds.
    pub breath_period_secs: f32,
    /// Crop to the detected face before atlas warping (plan 4).
    pub face_crop: bool,
    /// Face-crop margin as a fraction of the face-box size (default 0.35).
    pub face_crop_margin: f32,
    /// Use the Wav2Lip + ExpNet talking-head backend (Plan 5, task 8).
    ///
    /// When false, the classic atlas/warp path is used (default). When true,
    /// requires the `talking-head` cargo feature and the model weights in
    /// `.res/models/{wav2lip,expnet}/*.bpk` to be present (fails otherwise).
    pub talking_head: bool,
}

impl EngineConfig {
    /// Config from a voice model with defaults (64 px, synthetic face).
    pub fn voice(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            picture: None,
            strength: 0.5,
            size: DEFAULT_SIZE,
            breath_period_secs: 4.0,
            face_crop: true,
            face_crop_margin: crate::face_crop::DEFAULT_FACE_CROP_MARGIN,
            talking_head: false,
        }
    }

    /// Uses an avatar picture instead of the synthetic face.
    pub fn with_picture(mut self, picture: impl Into<PathBuf>) -> Self {
        self.picture = Some(picture.into());
        self
    }

    /// Output size (32, 64, or 128 px).
    pub fn with_size(mut self, size: u32) -> Self {
        self.size = size;
        self
    }

    /// Warp strength for the expression atlas.
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

    /// Enables/disables the Wav2Lip + ExpNet talking-head backend
    /// (requires the `talking-head` feature + model weights).
    pub fn with_talking_head(mut self, enabled: bool) -> Self {
        self.talking_head = enabled;
        self
    }
}

/// One synthesized span.
#[derive(Debug, Clone)]
struct SpanSynth {
    /// Start time of this span in the final clip (seconds).
    start: f32,
    result: SynthResult,
}

/// Synthesizes each span separately and offsets its timings into the clip.
fn synthesize_spans(
    tts: &mut EmbeddedPiper,
    spans: &[Span],
) -> anyhow::Result<Vec<SpanSynth>> {
    let mut out = Vec::new();
    let mut t = 0.0f32;
    for span in spans {
        let result = tts
            .synthesize(&span.text)
            .with_context(|| format!("synthesize {:?}", span.text))?;
        let dur = result.duration();
        out.push(SpanSynth {
            start: t,
            result,
        });
        t += dur;
    }
    Ok(out)
}

/// Concatenates the per-span samples into one clip.
fn concat_samples(synth: &[SpanSynth]) -> (Vec<f32>, u32) {
    let mut samples = Vec::new();
    let sr = synth
        .first()
        .map(|s| s.result.sample_rate)
        .unwrap_or(FPS as u32);
    for s in synth {
        samples.extend_from_slice(&s.result.samples);
    }
    (samples, sr)
}

/// Renders the full clip: text → markers → TTS → analysis → visemes →
/// render → encode → `.mp4`.
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

    // 1. Markers.
    let spans = markers::parse(text);
    anyhow::ensure!(!spans.is_empty(), "no text to render");

    // 2. TTS (per span).
    let mut tts = EmbeddedPiper::new(&cfg.model_path)
        .with_context(|| format!("load voice {}", cfg.model_path.display()))?;
    let synth = synthesize_spans(&mut tts, &spans)?;
    let (samples, sr) = concat_samples(&synth);
    anyhow::ensure!(!samples.is_empty(), "no audio synthesized");

    // 2b. Talking-head path (Plan 5, tasks 8-9): Wav2Lip + ExpNet fused
    //    frames at the fixed high-res 256×256 output (user decision G7).
    #[cfg(feature = "talking-head")]
    if cfg.talking_head {
        let fcfg = crate::fusion::FusionConfig::default()
            .with_size(crate::talking_head::TALKING_HEAD_SIZE)
            .with_strength(cfg.strength);
        let clip = crate::talking_head::render_talking_head(
            &samples,
            sr,
            cfg.picture.as_deref(),
            &fcfg,
            expnet::backend::BackendPreference::Auto,
        )
        .context("talking-head render")?;
        anyhow::ensure!(!clip.is_empty(), "talking-head produced no frames");
        encoder::encode_and_mux(out, &clip, &samples, sr, 1)?;
        return Ok((clip.len(), t0.elapsed()));
    }
    #[cfg(not(feature = "talking-head"))]
    if cfg.talking_head {
        anyhow::bail!(
            "talking_head=true requires the `talking-head` cargo feature \
             (wav2lip + expnet burn crates)"
        );
    }

    // 3. Analysis (head bob + breathing).
    let analysis = analysis::analyze(&samples, sr, cfg.breath_period_secs);
    let audio_frames = analysis.frame_count();

    // 4. Visemes: concatenate per-span timings, offset into the clip.
    let mut timings = Vec::new();
    for s in &synth {
        for t in &s.result.phonemes {
            timings.push(crate::tts::PhonemeTiming {
                phoneme: t.phoneme.clone(),
                start: s.start + t.start,
                end: s.start + t.end,
            });
        }
    }
    let frame_count = audio_frames.max(1);
    let visemes = viseme_frames(&timings, frame_count);

    // 5. Atlas (CPU warping; the GPU atlas builder is exercised by task 6).
    //    Face-crop first (plan 4): detect the face and center-crop so the
    //    hardcoded atlas control points align with real features. Falls back
    //    to the whole image when disabled, the model is missing, or no face
    //    is found. Then center-crop to a square (the renderer requires a
    //    square atlas band) and downscale so the warp stays cheap: the atlas
    //    is built at source resolution but only ever sampled at 32/64 px.
    let src = match &cfg.picture {
        Some(p) => {
            let img = crate::atlas::load_image(p)
                .with_context(|| format!("load avatar {}", p.display()))?;
            let img = crate::face_crop::apply_face_crop(img, cfg.face_crop, cfg.face_crop_margin);
            let img = crate::atlas::center_crop_square(&img);
            crate::atlas::downscale(&img)
        }
        None => crate::atlas::test_face_cpu(cfg.size, cfg.size),
    };
    let atlas = crate::atlas::atlas_cpu(&src, cfg.strength);

    // 6. Render (wgpu compute; tokio runtime blocks on the async device).
    let rt = tokio::runtime::Runtime::new().context("tokio runtime")?;
    let renderer = rt
        .block_on(crate::renderer::FrameRenderer::new())
        .context("GPU device")?;
    // Expression is audio-driven (task 7): every frame uses the neutral atlas
    // band (0) with no cross-fade.
    let mut frames: Vec<([f32; 12], f32, f32, u32, u32, f32)> = Vec::with_capacity(frame_count);
    for (frame, row) in visemes.iter().enumerate() {
        let mut w = [0.0f32; 12];
        w.copy_from_slice(row);
        let f = &analysis.frames[frame.min(audio_frames.saturating_sub(1))];
        frames.push((w, f.head_bob, f.breathing, 0, 0, 0.0));
    }
    let clip = renderer.render_clip(&atlas, &frames, cfg.size)?;

    // 7. Encode + mux.
    encoder::encode_and_mux(out, &clip, &samples, sr, 1)?;

    Ok((clip.len(), t0.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::PhonemeTiming;

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

    /// T9: end-to-end — `render("Hello <happy> world", out)` → MP4 exists,
    /// duration ≈ audio duration ± 100 ms, size 32 or 64 px.
    #[test]
    fn end_to_end_render() {
        let Some(model) = test_model() else { return };
        let tmp = std::env::temp_dir().join("avatar_t9_test.mp4");
        let cfg = EngineConfig::voice(&model).with_size(32);
        let (frames, _) = render_with_stats("Hello <happy> world", &tmp, &cfg).unwrap();
        assert!(frames > 0);
        assert!(cfg.size == 32 || cfg.size == 64);

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
        assert_eq!(spans[0].mode, crate::markers::Mode::Neutral);
        // Each `<tag>` (with its leading space) is removed, leaving the
        // surrounding whitespace: " hi " + " a " + " b" with the markers'
        // spaces preserved.
        assert_eq!(spans[0].text, " hi   a  b");
    }

    /// Per-span phoneme timings are offset into the clip timeline.
    #[test]
    fn concat_timings_offset() {
        let timing = |p: &str, start: f32, end: f32| PhonemeTiming {
            phoneme: p.to_string(),
            start,
            end,
        };
        let s = SpanSynth {
            start: 0.5,
            result: SynthResult {
                samples: vec![0.0; 16],
                sample_rate: 32,
                phonemes: vec![timing("h", 0.0, 0.2), timing("ə", 0.2, 0.5)],
            },
        };
        let mut timings = Vec::new();
        for t in &s.result.phonemes {
            timings.push(PhonemeTiming {
                phoneme: t.phoneme.clone(),
                start: s.start + t.start,
                end: s.start + t.end,
            });
        }
        assert_eq!(timings[0].start, 0.5);
        assert_eq!(timings[1].end, 0.5 + 0.5);
    }
}
