//! Wav2Lip + ExpNet talking-head render path (Plan 5, task 8; reordered
//! pipeline Plan 7, task 5).
//!
//! Replaces the atlas/warp renderer with a photorealistic talking head. The
//! order is the Plan 7 reorder (see `steps-analysis.md`):
//!
//! 1. Piper TTS audio is resampled to 16 kHz (the models' input rate).
//! 2. [`expnet::ExpNet`] maps the audio to per-frame expression coefficients.
//! 3. The source picture is made square at its **native** resolution (N×N, no
//!    face crop — the full frame is kept for the later in-paint).
//! 4. The detected face bbox is cropped with the Wav2Lip pads (M×M → 96×96)
//!    and [`wav2lip::Wav2Lip`] re-renders the **whole** face (clean input).
//! 5. The 96×96 face is resized to 112×112 and the **expression warp** runs on
//!    it, driven by **PFLD-detected** control points (Plan 7, tasks 2–3).
//! 6. The 112×112 face is resized back to M×M and **in-painted** into the N×N
//!    square at the bbox rect (2 px feather) — no sub-region mask, no seam.
//! 7. The frame is resized to the requested output size.
//!
//! Returns RGBA frames at `FusionConfig::size`, which the caller encodes to
//! MP4 with the existing encoder/muxer (audio keeps the original TTS rate).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use expnet::backend::{Backend, BackendPreference};
use image::{ImageBuffer, Rgba};

use crate::analysis::FPS;
use crate::fusion::{FusionConfig, patch_to_image, warp_from_landmarks};
use crate::inpaint::{DEFAULT_FEATHER_PX, FaceRect, crop_face_padded, paste_face};

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
/// Path to the PFLD `.bpk` weights (Plan 7).
pub const PFLD_WEIGHTS: &str = "pfld.bpk";

/// Resolution the expression warp (step 5) runs at — PFLD's native input.
const EXPR_SIZE: u32 = 112;

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

/// Number of video frames for `n_samples` of audio at `sample_rate`, rendered
/// at `fps`. Always at least 1 (a non-empty clip yields at least one frame).
fn frame_count(n_samples: usize, sample_rate: u32, fps: f32) -> usize {
    let secs = n_samples as f32 / sample_rate as f32;
    ((secs * fps) as usize).max(1)
}

/// Renders the talking-head frames for a clip.
///
/// - `samples` / `sr`: the TTS audio (at Piper's rate; resampled internally).
/// - `picture`: the avatar source image (optional; a synthetic face is used).
/// - `cfg`: the fusion configuration (output size, warp strength, mask).
/// - `backend_pref`: optional backend override (defaults to Auto).
/// - `fps`: output video frame rate. Wav2Lip was trained at 25 fps; other
///   rates change the frame count and the mel-window stride (see
///   [`wav2lip::mel_chunks`]). Values `<= 0` fall back to 25.
///
/// Returns the RGBA frames, one per `fps` video frame, at `cfg.size`.
pub fn render_talking_head(
    samples: &[f32],
    sr: u32,
    picture: Option<&Path>,
    cfg: &FusionConfig,
    backend_pref: BackendPreference,
    fps: f32,
) -> Result<Vec<ImageBuffer<Rgba<u8>, Vec<u8>>>> {
    let fps = if fps > 0.0 { fps } else { FPS };

    // 1. Resample to the model rate.
    let samples16k = resample(samples, sr, MODEL_SAMPLE_RATE);
    anyhow::ensure!(!samples16k.is_empty(), "no audio to render");

    // 2. Mel (shared by ExpNet + Wav2Lip).
    let mel = expnet::mel::melspectrogram(&samples16k);
    anyhow::ensure!(mel.n_frames > 0, "empty mel spectrogram");

    // Video frame count at the requested rate.
    let video_frames = frame_count(samples16k.len(), MODEL_SAMPLE_RATE, fps);

    // 3. Load model weights.
    let dir = models_dir().context("no .res/models directory found")?;
    let expnet_weights = dir.join("expnet").join(EXPNET_WEIGHTS);
    let wav2lip_weights = dir.join("wav2lip").join(WAV2LIP_WEIGHTS);
    let pfld_weights = dir.join("pfld").join(PFLD_WEIGHTS);
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
    anyhow::ensure!(
        pfld_weights.exists(),
        "pfld weights not found: {}",
        pfld_weights.display()
    );

    let backend = Backend::select(backend_pref).context("backend selection failed")?;
    let expnet = expnet::ExpNet::with_backend(&expnet_weights, &backend)
        .context("load expnet model")?;
    let wav2lip = wav2lip::Wav2Lip::with_backend(&wav2lip_weights, &backend)
        .context("load wav2lip model")?;
    let pfld = pfld::PFLD::with_backend(&pfld_weights, &backend)
        .context("load pfld model")?;

    // 4. Expression coefficients (ExpNet) — one 64-d vector per video frame.
    let coeffs = expnet.predict(&mel, video_frames).context("expnet predict")?;

    // 5. Mel chunks + mouth patches (Wav2Lip).
    let chunks = wav2lip::mel_chunks(&mel, fps);
    anyhow::ensure!(!chunks.is_empty(), "no wav2lip mel chunks");
    let chunk_frames = chunks.len();

    // 6. Source square (step 0): native N×N, no face crop (full frame kept).
    let square = load_source_square(picture)?;
    let (n, _) = square.dimensions();

    // 7. Step 1: detect the face on the square and crop it with the Wav2Lip
    //    pads (M×M). Falls back to the whole square when no face is found.
    let (face_crop, rect) = match face_bbox(&square) {
        Some(bbox) => crop_face_padded(&square, &bbox),
        None => {
            let rect = FaceRect { x: 0, y: 0, w: n, h: n };
            (square.clone(), rect)
        }
    };

    // 8. Step 2: Wav2Lip on the clean cropped face (same face every frame,
    //    reused across chunks). `face_input` resizes the crop to 96×96.
    let face = wav2lip::face_input(&face_crop);
    let faces = vec![face; chunk_frames];
    let patches = wav2lip
        .predict(&chunks, &faces)
        .context("wav2lip predict")?;

    // 9. Steps 3–5: one frame per video frame. Each frame uses **its own**
    //    Wav2Lip patch (one per mel chunk) — the mouth must follow the audio
    //    window for that frame, not a single reused face. Per frame:
    //      step 3: 96×96 patch → 112×112, PFLD landmarks → expression warp;
    //      step 4: 112×112 → M×M, in-paint into the N×N square;
    //      step 5: resize to the requested output size.
    let mut frames = Vec::with_capacity(video_frames);
    for i in 0..video_frames {
        // The Wav2Lip patch for this frame's audio window (chunks ≈ frames;
        // index clamped). Using `patches[0]` for every frame would freeze the
        // mouth — the per-chunk patches carry the lip motion.
        let chunk_idx = i.min(chunk_frames.saturating_sub(1));
        let face96 = patch_to_image(&patches[chunk_idx]);
        let face112 = image::imageops::resize(
            &face96,
            EXPR_SIZE,
            EXPR_SIZE,
            image::imageops::FilterType::Lanczos3,
        );

        // Step 3: PFLD landmarks on this (clean, unwarped) Wav2Lip face, then
        // the expression warp driven by the ExpNet coefficients.
        let landmarks9 = detect_control_points(&pfld, &face112)?;
        let coeff_row = &coeffs[i * 64..(i + 1) * 64];
        let expr_face = warp_from_landmarks(&face112, coeff_row, &landmarks9, cfg.strength);

        // Step 4: resize 112×112 → M×M and paste into the N×N square.
        let face_m = image::imageops::resize(
            &expr_face,
            rect.w,
            rect.h,
            image::imageops::FilterType::Lanczos3,
        );
        let square2 = paste_face(&square, &face_m, rect, DEFAULT_FEATHER_PX);

        // Step 5: resize to the requested output size.
        let frame = if square2.dimensions() == (cfg.size, cfg.size) {
            square2
        } else {
            image::imageops::resize(
                &square2,
                cfg.size,
                cfg.size,
                image::imageops::FilterType::Lanczos3,
            )
        };
        frames.push(frame);
    }

    Ok(frames)
}

/// Step 0: loads the source picture and makes it square at its **native**
/// resolution (no face crop, no resize). Falls back to a synthetic face when
/// no picture is given.
fn load_source_square(picture: Option<&Path>) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    match picture {
        Some(p) => {
            let img = crate::atlas::load_image(p)
                .with_context(|| format!("load avatar {}", p.display()))?;
            Ok(crate::atlas::center_crop_square(&img))
        }
        None => Ok(crate::atlas::test_face_cpu(256, 256)),
    }
}

/// Step 1 helper: detects the highest-scoring face on the square, using the
/// bundled SeetaFace model. Returns `None` when the model is missing or no
/// face is found (the caller falls back to the whole square).
fn face_bbox(square: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Option<rustface::Rectangle> {
    let model = crate::face_crop::face_model_path()?;
    crate::face_crop::detect_face_box(square, &model)
}

/// Step 3 helper: runs PFLD on the 112×112 face and maps the 98 landmarks to
/// the 9 expression-warp control points.
fn detect_control_points(
    pfld: &pfld::PFLD,
    face112: &ImageBuffer<Rgba<u8>, Vec<u8>>,
) -> Result<[[f32; 2]; pfld::NUM_CONTROL_POINTS]> {
    // RGBA → flat RGB (HWC, 0..1) for PFLD.
    let (w, h) = face112.dimensions();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = face112.get_pixel(x, y);
            rgb.push(p[0] as f32 / 255.0);
            rgb.push(p[1] as f32 / 255.0);
            rgb.push(p[2] as f32 / 255.0);
        }
    }
    let landmarks = pfld.predict(&rgb).context("pfld predict")?;
    pfld::landmarks_to_control_points(&landmarks).context("map landmarks to control points")
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

    /// `frame_count` scales linearly with fps: a 2 s clip at 16 kHz gives
    /// `2 s × fps` frames (and at least 1 for a non-empty clip).
    #[test]
    fn frame_count_scales_with_fps() {
        let sr = MODEL_SAMPLE_RATE;
        let two_secs = (2.0 * sr as f32) as usize;
        assert_eq!(frame_count(two_secs, sr, 25.0), 50);
        assert_eq!(frame_count(two_secs, sr, 16.0), 32);
        assert_eq!(frame_count(two_secs, sr, 10.0), 20);
        // Non-empty audio always yields at least one frame.
        assert_eq!(frame_count(1, sr, 16.0), 1);
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
            && dir.join("pfld").join(PFLD_WEIGHTS).exists()
    }

    /// Resolves the `.res` asset directory (for the avatar picture).
    fn res_dir() -> Option<PathBuf> {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let c = dir.join(".res");
            if c.is_dir() {
                return Some(c);
            }
            if !dir.pop() {
                return None;
            }
        }
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
        let frames = render_talking_head(&samples, sr, None, &cfg, BackendPreference::Auto, 25.0).unwrap();
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

        let a = render_talking_head(&sine(440.0), sr, None, &cfg, BackendPreference::Auto, 25.0).unwrap();
        let b = render_talking_head(&sine(880.0), sr, None, &cfg, BackendPreference::Auto, 25.0).unwrap();
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

    /// Regression (frozen mouth): each frame must use **its own** Wav2Lip patch
    /// (one per mel chunk), not a single reused face. With a speech-like clip
    /// the mouth region must change materially from frame to frame. The old bug
    /// (`patch_to_image(&patches[0])` for every frame) left the mouth nearly
    /// static. Skips when weights are absent.
    #[test]
    fn talking_head_mouth_moves_across_frames() {
        if !models_present() {
            return;
        }
        let sr = MODEL_SAMPLE_RATE;
        let n = (0.5 * sr as f32) as usize; // 12 frames
        // Amplitude-modulated tone: the mouth should move a lot.
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
                (2.0 * std::f32::consts::PI * 180.0 * t).sin() * env * 0.5
            })
            .collect();

        let cfg = FusionConfig::default().with_size(256);
        let frames =
            render_talking_head(&samples, sr, None, &cfg, BackendPreference::Auto, 25.0).unwrap();
        assert!(frames.len() >= 4);

        // Mean absolute difference between consecutive frames over the whole
        // frame. The frozen-mouth bug gave ~0.017/255; the fix gives ~0.36/255.
        let mean_abs = |a: &ImageBuffer<Rgba<u8>, Vec<u8>>,
                        b: &ImageBuffer<Rgba<u8>, Vec<u8>>| -> f64 {
            let mut s = 0.0f64;
            let mut c = 0.0f64;
            for (p, q) in a.pixels().zip(b.pixels()) {
                for ch in 0..3 {
                    s += (p[ch] as f64 - q[ch] as f64).abs();
                    c += 1.0;
                }
            }
            s / c
        };
        let avg: f64 = frames
            .windows(2)
            .map(|w| mean_abs(&w[0], &w[1]))
            .sum::<f64>()
            / (frames.len() - 1) as f64;

        assert!(
            avg > 0.1,
            "frames are nearly static (avg consecutive diff {avg:.4}/255) — \
             the per-frame Wav2Lip patch is probably not being used"
        );
    }

    /// Plan 7, task 6: the reordered pipeline has **no in-paint seam**. Renders
    /// a real frame from `avatar.jpg` and checks that the horizontal gradient at
    /// the paste boundary is not a spike vs. the interior median. Skips when the
    /// avatar picture is absent.
    #[test]
    fn talking_head_has_no_inpaint_seam() {
        if !models_present() {
            return;
        }
        let Some(res) = res_dir() else { return };
        let avatar = res.join("avatar.jpg");
        if !avatar.exists() {
            return;
        }

        let sr = MODEL_SAMPLE_RATE;
        let n = (0.2 * sr as f32) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();

        let cfg = FusionConfig::default().with_size(256);
        let frames = render_talking_head(
            &samples,
            sr,
            Some(&avatar),
            &cfg,
            BackendPreference::Auto,
            25.0,
        )
        .unwrap();
        let frame = &frames[frames.len() / 2];

        // Locate the paste boundary from the detector bbox on the N×N square
        // (avatar.jpg is 740×740; bbox x=219 w=352 -> right edge 571).
        let (n_side, _) = (740.0f32, 740.0f32);
        let scale = cfg.size as f32 / n_side;
        let x_right = (571.0 * scale).round() as u32;
        let y0 = (135.0 * scale).round() as u32 + 5;
        let y1 = (497.0 * scale).round() as u32 - 5;

        // Mean absolute horizontal gradient at column x over rows y0..y1.
        let col_grad = |x: u32| -> f64 {
            let mut s = 0.0;
            let mut c = 0.0;
            for y in y0..y1 {
                let a = frame.get_pixel(x, y);
                let b = frame.get_pixel(x + 1, y);
                for ch in 0..3 {
                    s += (a[ch] as f64 - b[ch] as f64).abs();
                }
                c += 1.0;
            }
            s / (c * 3.0)
        };

        let mut interior: Vec<f64> = (x_right - 60..x_right - 3).map(col_grad).collect();
        interior.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = interior[interior.len() / 2].max(1e-6);
        let boundary = col_grad(x_right);

        // A seam would show as a large spike; require it to stay modest.
        assert!(
            boundary / median < 3.0,
            "in-paint boundary gradient {boundary:.2} is {:.2}x the interior \
             median {median:.2} — visible seam",
            boundary / median
        );
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
        let frames = render_talking_head(&samples, sr, None, &cfg, BackendPreference::Auto, 25.0).unwrap();
        assert_eq!(frames.len(), 3);

        let out = std::env::temp_dir().join("avatar_t11_talking_head.mp4");
        crate::encoder::encode_and_mux(&out, &frames, &samples, sr, 1, 25.0).unwrap();
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

    /// The render is deterministic: identical inputs produce byte-identical
    /// frames (no stochastic latent — the pose warp was removed).
    #[test]
    fn talking_head_render_is_deterministic() {
        if !models_present() {
            return;
        }
        let sr = MODEL_SAMPLE_RATE;
        let n = (0.2 * sr as f32) as usize; // 5 frames
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let cfg = FusionConfig::default().with_size(128);

        let a = render_talking_head(&samples, sr, None, &cfg, BackendPreference::Auto, 25.0)
            .unwrap();
        let b = render_talking_head(&samples, sr, None, &cfg, BackendPreference::Auto, 25.0)
            .unwrap();
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.iter().zip(b.iter()) {
            assert_eq!(fa.as_raw(), fb.as_raw(), "render should be deterministic");
        }
    }
}
