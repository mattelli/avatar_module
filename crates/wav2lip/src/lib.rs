//! wav2lip — Wav2Lip lip-sync GAN inference (Plan 5, task 4).
//!
//! Wav2Lip (Rudrabha et al.) takes an audio window + a face frame and
//! re-renders the mouth to match the speech. This crate runs the generator
//! as inference-only Burn code on `burn-wgpu` (GPU) or `burn-flex` (CPU),
//! using weights imported from ONNX (`model/wav2lip.rs` + `.bpk`, task 2).
//!
//! # Pipeline (mirrors `inference.py`)
//!
//! 1. [`mel`] (reused from `expnet`) computes the SadTalker/librosa mel.
//! 2. [`mel_chunks`] slices 16-column windows at `80/25` per video frame.
//! 3. [`face_input`] detects the face (rustface), crops + resizes to 96×96,
//!    and builds the 6-channel input: `[masked_face (lower half zeroed),
//!    full_face]`, normalized to 0..1.
//! 4. [`Wav2Lip::predict`] runs the generator per chunk and returns the
//!    96×96×3 mouth patches (0..1).

pub mod model;

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use image::{ImageBuffer, Rgba};
use rustface::Rectangle;

use model::Model;

/// Tile size for the Wav2Lip face window.
pub const IMG_SIZE: u32 = 96;
/// Mel window width (columns) per video frame.
pub const MEL_STEP: usize = 16;
/// Audio-embedding dimension (the Wav2Lip audio encoder output, consumed by
/// Audio2Pose — Plan 6).
pub const AUDIO_EMB_DIM: usize = 512;
/// `mel_idx_multiplier = 80 / fps` (80 mel frames/sec at 16 kHz, 25 fps video).
const MEL_IDX_MULTIPLIER: f32 = 80.0 / 25.0;
/// Face pad: Wav2Lip default `pads = [0, 10, 0, 0]` -> (pady1, pady2, padx1, padx2).
const PADS: (i64, i64, i64, i64) = (0, 10, 0, 0);

/// Wav2Lip inference wrapper.
///
/// Loads the converted `.bpk` weights and runs audio + face frames → lip-
/// synced mouth patches on the selected Burn device.
pub struct Wav2Lip {
    model: Model,
    device: Device,
}

impl Wav2Lip {
    /// Loads the Wav2Lip model from a `.bpk` weight file, auto-selecting the
    /// backend (GPU via burn-wgpu when available, CPU otherwise).
    pub fn new(weights: impl AsRef<Path>) -> Result<Self> {
        let backend = expnet::backend::Backend::auto()?;
        Self::with_backend(weights, &backend)
    }

    /// Loads the Wav2Lip model on an explicitly chosen [`expnet::backend::Backend`].
    pub fn with_backend(
        weights: impl AsRef<Path>,
        backend: &expnet::backend::Backend,
    ) -> Result<Self> {
        let device = backend.device().clone();
        let model = Model::from_file(weights.as_ref(), &device);
        Ok(Self { model, device })
    }

    /// Runs only the Wav2Lip audio encoder on mel chunks and returns the
    /// 512-d audio embeddings, one per chunk (Plan 6 — Audio2Pose input).
    ///
    /// - `mel_chunks`: `[n, 80, 16]` normalized mel windows (from
    ///   [`mel_chunks`]).
    ///
    /// Returns a flat `[n, 512]` row-major embedding sequence. Chunks are
    /// processed in batches of `batch_size` to amortize per-call overhead.
    pub fn audio_embeddings(
        &self,
        mel_chunks: &[MelChunk],
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(!mel_chunks.is_empty(), "no input chunks");
        let n = mel_chunks.len();
        let batch_size = batch_size.max(1);
        let mut out = Vec::with_capacity(n * AUDIO_EMB_DIM);

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let b = end - start;

            // mel: [b, 1, 80, 16]
            let mut mel_data = Vec::with_capacity(b * 80 * 16);
            for chunk in &mel_chunks[start..end] {
                mel_data.extend_from_slice(&chunk.mel);
            }
            let mel_t: Tensor<4> =
                Tensor::from_data(TensorData::new(mel_data, [b, 1, 80, 16]), &self.device);

            let out_t = self.model.audio_embedding(mel_t); // [b, 512, 1, 1]
            let data = out_t.into_data();
            let arr = data.as_slice::<f32>().context("wav2lip embedding not f32")?;
            // Flatten [b, 512, 1, 1] -> [b, 512].
            out.extend_from_slice(arr);
        }

        Ok(out)
    }

    /// Runs Wav2Lip on mel chunks + face inputs.
    ///
    /// - `mel_chunks`: `[n, 80, 16]` normalized mel windows (from
    ///   [`mel_chunks`]).
    /// - `faces`: `[n, 6, 96, 96]` 6-channel face tensors (from
    ///   [`face_input`]); must have the same length as `mel_chunks`.
    ///
    /// Returns `[n, 3, 96, 96]` mouth patches (values 0..1).
    ///
    /// Chunks are processed in batches of `batch_size` (default 8) to
    /// amortize per-call overhead on the GPU while keeping memory bounded.
    pub fn predict(&self, mel_chunks: &[MelChunk], faces: &[FaceInput]) -> Result<Vec<Vec<f32>>> {
        self.predict_batched(mel_chunks, faces, 8)
    }

    /// As [`predict`], with an explicit batch size.
    pub fn predict_batched(
        &self,
        mel_chunks: &[MelChunk],
        faces: &[FaceInput],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            mel_chunks.len() == faces.len(),
            "mel chunks ({}) and faces ({}) length mismatch",
            mel_chunks.len(),
            faces.len()
        );
        anyhow::ensure!(!mel_chunks.is_empty(), "no input chunks");

        let n = mel_chunks.len();
        let batch_size = batch_size.max(1);
        let mut out = Vec::with_capacity(n);

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let b = end - start;

            // mel: [b, 1, 80, 16]
            let mut mel_data = Vec::with_capacity(b * 80 * 16);
            for chunk in &mel_chunks[start..end] {
                mel_data.extend_from_slice(&chunk.mel);
            }
            let mel_t: Tensor<4> =
                Tensor::from_data(TensorData::new(mel_data, [b, 1, 80, 16]), &self.device);

            // face: [b, 6, 96, 96]
            let mut face_data = Vec::with_capacity(b * 6 * 96 * 96);
            for i in start..end {
                face_data.extend_from_slice(&faces[i].data);
            }
            let face_t: Tensor<4> = Tensor::from_data(
                TensorData::new(
                    face_data,
                    [b, 6, IMG_SIZE as usize, IMG_SIZE as usize],
                ),
                &self.device,
            );

            let out_t = self.model.forward(mel_t, face_t); // [b, 3, 96, 96]
            let data = out_t.into_data();
            let arr = data.as_slice::<f32>().context("wav2lip output not f32")?;
            // Split the batch into per-chunk patches.
            let per = 3 * 96 * 96;
            for c in 0..b {
                out.push(arr[c * per..(c + 1) * per].to_vec());
            }
        }

        Ok(out)
    }
}

/// One 16-column mel window (`[80, 16]`, row-major).
#[derive(Debug, Clone)]
pub struct MelChunk {
    /// `[80, 16]` row-major.
    pub mel: Vec<f32>,
}

/// One 6-channel face tensor (`[6, 96, 96]`), values 0..1.
#[derive(Debug, Clone)]
pub struct FaceInput {
    /// `[6, 96, 96]` row-major (channels first).
    pub data: Vec<f32>,
}

/// Slices the mel spectrogram into 16-column chunks, one per video frame.
///
/// Mirrors `inference.py`: `start_idx = int(i * 80/25)`; the last chunk uses
/// the final 16 columns when it would run past the end. `mel` is the
/// SadTalker mel matrix (`n_frames × 80`, from `expnet::mel::melspectrogram`).
pub fn mel_chunks(mel: &expnet::mel::MelSpectrogram) -> Vec<MelChunk> {
    let n_cols = mel.n_frames;
    let mut chunks = Vec::new();
    let mut i = 0usize;
    loop {
        let start = (i as f32 * MEL_IDX_MULTIPLIER) as usize;
        if start + MEL_STEP > n_cols {
            // Last chunk: use the final 16 columns.
            let last_start = n_cols.saturating_sub(MEL_STEP);
            chunks.push(slice_mel(mel, last_start));
            break;
        }
        chunks.push(slice_mel(mel, start));
        i += 1;
    }
    chunks
}

fn slice_mel(mel: &expnet::mel::MelSpectrogram, start: usize) -> MelChunk {
    let n_cols = mel.n_frames;
    let mut out = Vec::with_capacity(80 * MEL_STEP);
    for c in 0..MEL_STEP {
        let col = (start + c).min(n_cols - 1);
        out.extend_from_slice(mel.frame(col)); // 80 values
    }
    MelChunk { mel: out }
}

/// Detects the face in `img` (RGBA) and returns its bounding box.
///
/// Uses the SeetaFace detector via `rustface`. Returns `None` when the model
/// file is missing or no face is found (caller decides fallback).
pub fn detect_face_box(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    model_path: &Path,
) -> Option<Rectangle> {
    let (w, h) = img.dimensions();
    if w < 20 || h < 20 {
        return None;
    }
    let Ok(mut detector) = rustface::create_detector(&model_path.to_string_lossy()) else {
        return None;
    };
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let gray = image::imageops::grayscale(img);
    let data = gray.as_raw();
    let (gw, gh) = gray.dimensions();
    let image_data = rustface::ImageData::new(data, gw, gh);
    let mut faces: Vec<(rustface::Rectangle, f64)> = detector
        .detect(&image_data)
        .into_iter()
        .map(|f| (*f.bbox(), f.score()))
        .collect();
    faces.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    faces.into_iter().next().map(|(bbox, _)| bbox)
}

/// Builds the 6-channel Wav2Lip input for one 96×96 face crop.
///
/// Mirrors `inference.py datagen`: the face is resized to 96×96, normalized
/// to 0..1, and stacked as `[masked_face (rows 48..96 zeroed), full_face]` →
/// `[6, 96, 96]`.
pub fn face_input(face: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> FaceInput {
    let resized = image::imageops::resize(
        face,
        IMG_SIZE,
        IMG_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let mut data = vec![0.0f32; 6 * 96 * 96];
    let hw = (IMG_SIZE / 2) as usize; // 48

    for y in 0..IMG_SIZE as usize {
        for x in 0..IMG_SIZE as usize {
            let p = resized.get_pixel(x as u32, y as u32);
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            // Channels 0-2: masked (lower half zeroed).
            let masked = y >= hw;
            data[(0) * 96 * 96 + y * 96 + x] = if masked { 0.0 } else { r };
            data[(1) * 96 * 96 + y * 96 + x] = if masked { 0.0 } else { g };
            data[(2) * 96 * 96 + y * 96 + x] = if masked { 0.0 } else { b };
            // Channels 3-5: full face.
            data[(3) * 96 * 96 + y * 96 + x] = r;
            data[(4) * 96 * 96 + y * 96 + x] = g;
            data[(5) * 96 * 96 + y * 96 + x] = b;
        }
    }
    FaceInput { data }
}

/// Crops the source image to the detected face box with Wav2Lip pads, then
/// resizes to 96×96.
pub fn crop_face(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    bbox: &Rectangle,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (iw, ih) = img.dimensions();
    let (pady1, pady2, padx1, padx2) = PADS;

    let y1 = (bbox.y() as i64 - pady1).max(0) as u32;
    let y2 = (bbox.y() as i64 + bbox.height() as i64 + pady2).min(ih as i64) as u32;
    let x1 = (bbox.x() as i64 - padx1).max(0) as u32;
    let x2 = (bbox.x() as i64 + bbox.width() as i64 + padx2).min(iw as i64) as u32;
    let w = (x2 - x1).max(1);
    let h = (y2 - y1).max(1);

    let crop = image::imageops::crop_imm(img, x1, y1, w, h).to_image();
    image::imageops::resize(&crop, IMG_SIZE, IMG_SIZE, image::imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res_dir() -> Option<std::path::PathBuf> {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
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

    fn weights_path() -> Option<std::path::PathBuf> {
        let p = res_dir()?
            .join("models")
            .join("wav2lip")
            .join("wav2lip_gan.bpk");
        p.exists().then_some(p)
    }

    fn sine(freq: f32) -> Vec<f32> {
        let sr = expnet::mel::SAMPLE_RATE as usize;
        (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect()
    }

    /// T4: mel chunk count matches the Wav2Lip reference for 1 s audio.
    #[test]
    fn mel_chunks_match_reference() {
        let mel = expnet::mel::melspectrogram(&sine(440.0));
        let chunks = mel_chunks(&mel);
        // 1 s audio -> 81 mel frames; the reference `inference.py` loop
        // terminates when int(i*3.2)+16 > 81, i.e. at i=21 (start=67), giving
        // 22 chunks. Starts: [0,3,6,...,64,67].
        assert_eq!(chunks.len(), 22, "expected 22 chunks for 1 s audio");
        assert_eq!(
            chunks[0].mel.len(),
            80 * MEL_STEP,
            "chunk width must be 80x16"
        );
    }

    /// T4: face_input builds a 6-channel 96×96 tensor with the lower half of
    /// channels 0-2 zeroed and channels 3-5 equal to the full face.
    #[test]
    fn face_input_structure() {
        // Plain white image.
        let img = ImageBuffer::from_pixel(100, 100, Rgba([255u8, 255, 255, 255]));
        let fi = face_input(&img);
        assert_eq!(fi.data.len(), 6 * 96 * 96);
        // Upper half, channel 0 = 1.0 (white / 255).
        assert!((fi.data[0 * 96 * 96 + 0 * 96 + 10] - 1.0).abs() < 1e-3);
        // Lower half, channel 0 = 0.0 (masked).
        assert!(fi.data[0 * 96 * 96 + 50 * 96 + 10].abs() < 1e-3);
        // Channel 3 (full face) has the value everywhere.
        assert!((fi.data[3 * 96 * 96 + 50 * 96 + 10] - 1.0).abs() < 1e-3);
    }

    /// Builds a mel chunk directly (no model audio) from a sine — returns the
    /// FIRST mel chunk so tests run 1 forward pass (~16 s on CPU flex).
    fn first_chunk(freq: f32) -> MelChunk {
        let mel = expnet::mel::melspectrogram(&sine(freq));
        mel_chunks(&mel).remove(0)
    }

    /// T4: the Wav2Lip model runs on synthetic audio + a synthetic face on
    /// the CPU fallback backend, producing a 96×96×3 mouth patch in 0..1.
    #[test]
    fn wav2lip_runs_on_cpu() {
        let Some(weights) = weights_path() else { return };
        let backend = expnet::backend::Backend::select(expnet::backend::BackendPreference::Cpu)
            .unwrap();
        let model = Wav2Lip::with_backend(&weights, &backend).unwrap();

        let chunk = first_chunk(440.0);
        let img = ImageBuffer::from_pixel(96, 96, Rgba([128u8, 128, 128, 255]));
        let face = face_input(&img);

        let patches = model.predict(&[chunk], &[face]).unwrap();
        assert_eq!(patches.len(), 1);
        let p = &patches[0];
        assert_eq!(p.len(), 3 * 96 * 96);
        for v in p {
            assert!(v.is_finite(), "wav2lip patch not finite: {v}");
            assert!((0.0..=1.0).contains(v), "wav2lip patch out of range: {v}");
        }
    }

    /// The mouth region changes when the audio changes (lip-sync signal).
    #[test]
    fn mouth_changes_with_audio() {
        let Some(weights) = weights_path() else { return };
        let backend = expnet::backend::Backend::select(expnet::backend::BackendPreference::Cpu)
            .unwrap();
        let model = Wav2Lip::with_backend(&weights, &backend).unwrap();

        let chunk_a = first_chunk(440.0);
        let chunk_b = first_chunk(880.0);

        let img = ImageBuffer::from_pixel(96, 96, Rgba([128u8, 128, 128, 255]));
        let face_a = face_input(&img);
        let face_b = face_input(&img);

        let out_a = model.predict(&[chunk_a], &[face_a]).unwrap();
        let out_b = model.predict(&[chunk_b], &[face_b]).unwrap();

        // The mouth patches differ (same face, different audio).
        let diff: f32 = out_a[0]
            .iter()
            .zip(out_b[0].iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1e-3, "mouth should differ with audio, diff={diff}");
    }

    /// Plan 6: the audio encoder exposes a 512-d embedding per mel chunk, and
    /// different audio yields different embeddings (Audio2Pose input).
    #[test]
    fn audio_embeddings_shape_and_vary() {
        let Some(weights) = weights_path() else { return };
        let backend = expnet::backend::Backend::select(expnet::backend::BackendPreference::Cpu)
            .unwrap();
        let model = Wav2Lip::with_backend(&weights, &backend).unwrap();

        let chunk_a = first_chunk(440.0);
        let chunk_b = first_chunk(880.0);

        let emb_a = model.audio_embeddings(&[chunk_a], 8).unwrap();
        let emb_b = model.audio_embeddings(&[chunk_b], 8).unwrap();

        assert_eq!(emb_a.len(), AUDIO_EMB_DIM);
        assert_eq!(emb_b.len(), AUDIO_EMB_DIM);
        for v in emb_a.iter().chain(emb_b.iter()) {
            assert!(v.is_finite(), "embedding not finite: {v}");
        }
        // Different audio -> different embeddings.
        let diff: f32 = emb_a.iter().zip(emb_b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-3, "embeddings should differ with audio, diff={diff}");
    }
}
