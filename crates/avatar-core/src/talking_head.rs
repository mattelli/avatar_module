//! Wav2Lip + ExpNet talking-head render path (Plan 5, task 8).
//!
//! Replaces the atlas/warp renderer with a photorealistic talking head:
//!
//! 1. Piper TTS audio is resampled to 16 kHz (the models' input rate).
//! 2. [`expnet::ExpNet`] maps the audio to per-frame expression coefficients.
//! 3. [`wav2lip::Wav2Lip`] maps audio windows + the (face-cropped) source to
//!    precise 96×96 mouth patches.
//! 4. [`crate::fusion`] combines them per frame: expression-warp the source,
//!    then alpha-blend the mouth patch in.
//!
//! Returns RGBA frames at `FusionConfig::size`, which the caller encodes to
//! MP4 with the existing encoder/muxer (audio keeps the original TTS rate).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use expnet::backend::{Backend, BackendPreference};
use image::{ImageBuffer, Rgba};

use crate::analysis::FPS;
use crate::fusion::{FusionConfig, fuse_frame};

/// Sample rate the Wav2Lip / ExpNet models expect.
pub const MODEL_SAMPLE_RATE: u32 = 16_000;

/// Default output resolution of the talking-head clip (px). Supported sizes
/// are 64, 128, 256, or 512; this is the default used when the caller does not
/// override `EngineConfig::size`.
pub const TALKING_HEAD_SIZE: u32 = 256;

/// Path to the Wav2Lip `.bpk` weights.
pub const WAV2LIP_WEIGHTS: &str = "wav2lip_gan.bpk";
/// Path to the ExpNet `.bpk` weights.
pub const EXPNET_WEIGHTS: &str = "expnet.bpk";

/// Resolves the `.res/models` directory (same walk-up as `face_crop`).
fn models_dir() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let candidate = dir.join(".res").join("models");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Linear-interpolation resampler. `samples` at `from_rate` Hz → `to_rate` Hz.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio; // in input samples
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let frac = (pos - i0 as f64) as f32;
        out.push(samples[i0] * (1.0 - frac) + samples[i1] * frac);
    }
    out
}

/// Renders the talking-head frames for a clip.
///
/// - `samples` / `sr`: the TTS audio (at Piper's rate; resampled internally).
/// - `picture`: the avatar source image (optional; a synthetic face is used).
/// - `cfg`: the fusion configuration (output size, warp strength, mask).
/// - `pose`: optional flat per-frame Audio2Pose head-pose sequence
///   (`video_frames × 6`; yaw/pitch/roll + tx/ty/tz per frame). `None`
///   applies no head-pose warp.
/// - `backend_pref`: optional backend override (defaults to Auto).
///
/// Returns the RGBA frames, one per 25 fps video frame, at `cfg.size`.
pub fn render_talking_head(
    samples: &[f32],
    sr: u32,
    picture: Option<&Path>,
    cfg: &FusionConfig,
    pose: Option<&[f32]>,
    backend_pref: BackendPreference,
) -> Result<Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>> {
    // 1. Resample to the model rate.
    let samples16k = resample(samples, sr, MODEL_SAMPLE_RATE);
    anyhow::ensure!(!samples16k.is_empty(), "no audio to render");

    // 2. Mel (shared by ExpNet + Wav2Lip).
    let mel = expnet::mel::melspectrogram(&samples16k);
    anyhow::ensure!(mel.n_frames > 0, "empty mel spectrogram");

    // Video frame count (25 fps).
    let video_frames = (samples16k.len() as f32 / MODEL_SAMPLE_RATE as f32 * FPS) as usize;
    let video_frames = video_frames.max(1);

    // 3. Load model weights.
    let dir = models_dir().context("no .res/models directory found")?;
    let expnet_weights = dir.join("expnet").join(EXPNET_WEIGHTS);
    let wav2lip_weights = dir.join("wav2lip").join(WAV2LIP_WEIGHTS);
    anyhow::ensure!(
        expnet_weights.exists(),
        "expnet weights not found: {}",
        expnet_weights.display()
    );
    anyhow::ensure!(
        wav2lip_weights.exists(),
        "wav2lip weights not found: {}",
        wav2lip_weights.display()
    );

    let backend = Backend::select(backend_pref).context("backend selection failed")?;
    let expnet = expnet::ExpNet::with_backend(&expnet_weights, &backend)
        .context("load expnet model")?;
    let wav2lip = wav2lip::Wav2Lip::with_backend(&wav2lip_weights, &backend)
        .context("load wav2lip model")?;

    // 4. Expression coefficients (ExpNet) — one 64-d vector per video frame.
    let coeffs = expnet.predict(&mel, video_frames).context("expnet predict")?;

    // 5. Mel chunks + mouth patches (Wav2Lip).
    let chunks = wav2lip::mel_chunks(&mel);
    anyhow::ensure!(!chunks.is_empty(), "no wav2lip mel chunks");
    let chunk_frames = chunks.len();

    // 6. Source face: detect + crop (or synthetic face).
    let src = load_source_face(picture, cfg.size)?;

    // The Wav2Lip input face is the same source crop every frame (static
    // image), resized to 96x96.
    let face = wav2lip::face_input(&src);
    let faces = vec![face; chunk_frames];
    let mouth_patches = wav2lip
        .predict(&chunks, &faces)
        .context("wav2lip predict")?;

    // 7. Fuse: one frame per video_frames. The mouth patches cover the mel
    //    chunks (chunk_frames >= video_frames); index clamped. The optional
    //    pose sequence is `video_frames × 6`; index clamped to its length.
    let mut frames = Vec::with_capacity(video_frames);
    for i in 0..video_frames {
        let coeff_row = &coeffs[i * 64..(i + 1) * 64];
        let chunk_idx = i.min(chunk_frames.saturating_sub(1));
        let mouth = &mouth_patches[chunk_idx];
        let pose_row = pose.and_then(|p| {
            let n = p.len() / 6;
            if n == 0 {
                None
            } else {
                let idx = i.min(n - 1);
                Some(&p[idx * 6..(idx + 1) * 6])
            }
        });
        let frame = fuse_frame(&src, coeff_row, pose_row, mouth, cfg);
        frames.push(frame);
    }

    Ok(frames)
}

/// Loads the source face for fusion: face-crop the picture (if any) or use a
/// synthetic face, ensuring it is square and at least `size` on each side.
fn load_source_face(
    picture: Option<&Path>,
    size: u32,
) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    match picture {
        Some(p) => {
            let img = crate::atlas::load_image(p)
                .with_context(|| format!("load avatar {}", p.display()))?;
            // Face-crop, center-crop to square, then resize to `size`.
            let img =
                crate::face_crop::apply_face_crop(img, true, crate::face_crop::DEFAULT_FACE_CROP_MARGIN);
            let img = crate::atlas::center_crop_square(&img);
            Ok(image::imageops::resize(
                &img,
                size,
                size,
                image::imageops::FilterType::Triangle,
            ))
        }
        None => Ok(crate::atlas::test_face_cpu(size, size)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default output size is 256×256 (one of the supported sizes).
    #[test]
    fn output_size_is_256() {
        assert_eq!(TALKING_HEAD_SIZE, 256);
    }

    #[test]
    fn resample_identity() {
        let x = vec![0.0f32, 0.5, 1.0, 0.5];
        assert_eq!(resample(&x, 22050, 22050), x);
    }

    #[test]
    fn resample_upsample_length() {
        // 22050 -> 16000 shrinks the length by ~0.725.
        let x = vec![0.0f32; 2205]; // 0.1 s
        let y = resample(&x, 22050, 16000);
        assert_eq!(y.len(), 1600);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn resample_interpolates() {
        // Linear ramp: resampling preserves the endpoints and is within range.
        let x: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let y = resample(&x, 100, 50);
        assert!((y[0] - 0.0).abs() < 0.05);
        assert!((y[y.len() - 1] - 1.0).abs() < 0.05);
        for v in &y {
            assert!((0.0..=1.0).contains(v));
        }
    }

    fn models_present() -> bool {
        let Some(dir) = models_dir() else { return false };
        dir.join("expnet").join(EXPNET_WEIGHTS).exists()
            && dir.join("wav2lip").join(WAV2LIP_WEIGHTS).exists()
    }

    /// T8: the full talking-head path produces frames at the configured size
    /// for a short synthetic audio clip, on the auto-selected backend (GPU on
    /// this machine — fast; the test skips when weights are absent).
    #[test]
    fn talking_head_renders_short_clip() {
        if !models_present() {
            return;
        }
        let sr = MODEL_SAMPLE_RATE;
        let n = (0.2 * sr as f32) as usize; // 5 frames
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let cfg = FusionConfig::default().with_size(128);
        let frames = render_talking_head(&samples, sr, None, &cfg, None, BackendPreference::Auto).unwrap();
        assert_eq!(frames.len(), 5);
        for f in &frames {
            assert_eq!(f.dimensions(), (cfg.size, cfg.size));
            // Frames are (essentially) opaque — the bilinear warp / Triangle
            // resize can attenuate edge alpha slightly.
            assert!(f.pixels().all(|p| p[3] >= 200), "frame alpha too low");
        }
    }

    /// T8: talking-head frames reflect the audio (non-constant output).
    #[test]
    fn talking_head_audio_drives_frames() {
        if !models_present() {
            return;
        }
        let sr = MODEL_SAMPLE_RATE;
        let n = (0.2 * sr as f32) as usize;
        let sine = |freq: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.5
                })
                .collect()
        };
        let cfg = FusionConfig::default().with_size(128);

        let a = render_talking_head(&sine(440.0), sr, None, &cfg, None, BackendPreference::Auto).unwrap();
        let b = render_talking_head(&sine(880.0), sr, None, &cfg, None, BackendPreference::Auto).unwrap();
        assert_eq!(a.len(), b.len());

        // Different audio -> different frames (mouth/expression differ).
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(fa, fb)| {
                fa.pixels()
                    .zip(fb.pixels())
                    .map(|(pa, pb)| {
                        (pa[0] as f64 - pb[0] as f64).abs()
                            + (pa[1] as f64 - pb[1] as f64).abs()
                            + (pa[2] as f64 - pb[2] as f64).abs()
                    })
                    .sum::<f64>()
            })
            .sum();
        assert!(diff > 1e-3, "different audio should produce different frames");
    }

    /// T11: end-to-end smoke — render a short talking-head clip, encode +
    /// mux to MP4, and verify 2 tracks with video sample count == frame count.
    /// Runs on the auto-selected backend (GPU on this machine).
    #[test]
    fn talking_head_mp4_smoke() {
        if !models_present() {
            return;
        }
        let sr = MODEL_SAMPLE_RATE;
        let n = (0.12 * sr as f32) as usize; // 3 frames
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let cfg = FusionConfig::default().with_size(256);
        let frames = render_talking_head(&samples, sr, None, &cfg, None, BackendPreference::Auto).unwrap();
        assert_eq!(frames.len(), 3);

        let out = std::env::temp_dir().join("avatar_t11_talking_head.mp4");
        crate::encoder::encode_and_mux(&out, &frames, &samples, sr, 1).unwrap();
        let size = std::fs::metadata(&out).unwrap().len();
        assert!(size > 1000, "mp4 too small: {size}");

        let f = std::fs::File::open(&out).unwrap();
        let reader = std::io::BufReader::new(f);
        let parsed = mp4::Mp4Reader::read_header(reader, size).unwrap();
        assert_eq!(parsed.tracks().len(), 2, "expected video + audio tracks");
        let video_id = parsed
            .tracks()
            .values()
            .find(|t| t.track_type().map(|tt| tt == mp4::TrackType::Video).unwrap_or(false))
            .unwrap()
            .track_id();
        let audio_id = parsed
            .tracks()
            .values()
            .find(|t| t.track_type().map(|tt| tt == mp4::TrackType::Audio).unwrap_or(false))
            .unwrap()
            .track_id();
        assert_eq!(
            parsed.sample_count(video_id).unwrap(),
            frames.len() as u32,
            "video sample count must equal rendered frame count"
        );
        assert!(parsed.sample_count(audio_id).unwrap() > 0, "audio track empty");

        let _ = std::fs::remove_file(&out);
    }
}
