//! expnet — ExpNet audio-to-expression inference (Plan 5, task 3).
//!
//! ExpNet (from SadTalker) maps an audio window to 3DMM expression
//! coefficients — the non-lip facial motion (eyebrows, cheeks, subtle
//! expression changes) driven by the audio. This crate runs it as
//! inference-only Burn code on `burn-wgpu` (GPU) or `burn-flex` (CPU),
//! using weights imported from ONNX (`model/expnet.rs` + `.bpk`, task 2).
//!
//! # Pipeline
//!
//! 1. [`mel::melspectrogram`] converts mono 16 kHz PCM to the SadTalker mel
//!    matrix (80 bins × frames).
//! 2. [`ExpNet::predict`] slices 16-frame mel windows, runs the network per
//!    frame, and returns a `(frames, 64)` expression-coefficient sequence.
//!
//! The `ref_coeff` (source-face 3DMM expression) defaults to zeros — see
//! [`ExpNet::with_ref_coeff`].

pub mod backend;
pub mod mel;
pub mod model;

pub use backend::{Backend, BackendPreference};

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use mel::{MelSpectrogram, N_MELS};

use model::Model;

/// Frames-per-second of the video (25 fps) — the mel window stride.
pub const FPS: f32 = 25.0;
/// Number of mel columns per audio window (SadTalker `syncnet_mel_step_size`).
pub const MEL_STEP: usize = 16;
/// Expression coefficient vector length (3DMM, first 64 of 70).
pub const EXP_DIM: usize = 64;
/// Chunk size used by SadTalker's `Audio2Exp.test` (processes 10 frames at a
/// time); we process frame-by-frame, so kept for reference.
pub const CHUNK: usize = 10;

/// ExpNet inference wrapper.
///
/// Loads the converted `.bpk` weights and runs audio → expression
/// coefficients on the selected Burn device.
pub struct ExpNet {
    model: Model,
    /// Source-face expression coefficients `[1, 64]` (zeros by default).
    ref_coeff: [f32; EXP_DIM],
    device: Device,
}

impl ExpNet {
    /// Loads the ExpNet model from a `.bpk` weight file, auto-selecting the
    /// backend (GPU via burn-wgpu when available, CPU otherwise).
    pub fn new(weights: impl AsRef<Path>) -> Result<Self> {
        let backend = Backend::auto()?;
        Self::with_backend(weights, &backend)
    }

    /// Loads the ExpNet model on an explicitly chosen [`Backend`].
    pub fn with_backend(weights: impl AsRef<Path>, backend: &Backend) -> Result<Self> {
        let device = backend.device().clone();
        let model = Model::from_file(weights.as_ref(), &device);
        let ref_coeff = [0.0f32; EXP_DIM];
        Ok(Self {
            model,
            ref_coeff,
            device,
        })
    }

    /// Sets the source-face 3DMM expression coefficients (length 64).
    ///
    /// Defaults to neutral zeros. In full SadTalker these come from BFM
    /// fitting on the input photo; the ExpNet audio→expression mapping still
    /// produces meaningful motion with zeros (the residual add is disabled).
    pub fn with_ref_coeff(mut self, coeff: [f32; EXP_DIM]) -> Self {
        self.ref_coeff = coeff;
        self
    }

    /// Runs ExpNet on a full mel spectrogram.
    ///
    /// `video_frames` is the number of 25 fps video frames (one expression
    /// vector per frame); it is derived from the audio length in SadTalker's
    /// `parse_audio_length` (audio_len * 25 / 16000). Returns a
    /// `(video_frames, EXP_DIM)` row-major expression-coefficient sequence.
    ///
    /// Frames are processed in batches of `batch_size` (default 32) to
    /// amortize per-call overhead on the GPU.
    pub fn predict(&self, mel: &MelSpectrogram, video_frames: usize) -> Result<Vec<f32>> {
        self.predict_batched(mel, video_frames, 32)
    }

    /// As [`predict`], with an explicit batch size.
    pub fn predict_batched(
        &self,
        mel: &MelSpectrogram,
        video_frames: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(mel.n_frames > 0, "empty mel spectrogram");
        anyhow::ensure!(video_frames > 0, "no video frames");
        let n_frames = video_frames;
        let batch_size = batch_size.max(1);

        let mut out = Vec::with_capacity(n_frames * EXP_DIM);

        for start in (0..n_frames).step_by(batch_size) {
            let end = (start + batch_size).min(n_frames);
            let b = end - start;

            // Build [b, 1, 80, 16] mel windows.
            let mut mel_data = Vec::with_capacity(b * N_MELS * MEL_STEP);
            for i in start..end {
                // SadTalker generate_batch: start_idx = (i-2) * 80/25, clamped.
                let start_idx = ((i as i64 - 2) * 80 / 25).max(0) as usize;
                for c in 0..MEL_STEP {
                    let src = (start_idx + c).min(mel.n_frames - 1);
                    mel_data.extend_from_slice(mel.frame(src));
                }
            }
            let mel_t: Tensor<4> = Tensor::from_data(
                TensorData::new(mel_data, [b, 1, N_MELS, MEL_STEP]),
                &self.device,
            );
            // ref/ratio broadcast over the batch (the model concats them with
            // the [b, 512] mel embedding, so they must be [b, ...]).
            let ref_t: Tensor<2> = Tensor::from_data(
                TensorData::new(self.ref_coeff.repeat(b), [b, EXP_DIM]),
                &self.device,
            );
            let ratio_t: Tensor<2> =
                Tensor::from_data(TensorData::new(vec![0.0f32; b], [b, 1]), &self.device);

            let out_t = self.model.forward(mel_t, ref_t, ratio_t); // [b, 64]
            let data = out_t.into_data();
            let arr = data.as_slice::<f32>().context("expnet output not f32")?;
            out.extend_from_slice(arr);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res_weights() -> Option<std::path::PathBuf> {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let candidate = dir
                .join(".res")
                .join("models")
                .join("expnet")
                .join("expnet.bpk");
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    /// T3: on synthetic audio + neutral ref, ExpNet produces a valid
    /// expression-coefficient sequence (finite, correct shape) on the CPU
    /// fallback backend.
    #[test]
    fn expnet_runs_on_cpu() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = ExpNet::with_backend(&weights, &backend).unwrap();

        // 1 s of 440 Hz sine at 16 kHz.
        let sr = mel::SAMPLE_RATE as usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let mel = mel::melspectrogram(&samples);
        // 1 s audio -> 25 video frames.
        let video_frames = samples.len() * super::FPS as usize / mel::SAMPLE_RATE as usize;
        let coeff = model.predict(&mel, video_frames).unwrap();

        assert_eq!(coeff.len(), video_frames * EXP_DIM);
        for v in &coeff {
            assert!(v.is_finite(), "expnet coeff not finite: {v}");
        }
    }

    /// Predict on silence still returns the correct shape.
    #[test]
    fn expnet_silence_shape() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = ExpNet::with_backend(&weights, &backend).unwrap();
        let samples = vec![0.0f32; mel::SAMPLE_RATE as usize];
        let mel = mel::melspectrogram(&samples);
        let video_frames = samples.len() * super::FPS as usize / mel::SAMPLE_RATE as usize;
        let coeff = model.predict(&mel, video_frames).unwrap();
        assert_eq!(coeff.len(), video_frames * EXP_DIM);
    }
}
