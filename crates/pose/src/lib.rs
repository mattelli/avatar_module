//! pose — Audio2Pose CVAE decoder inference (Plan 6, task 5).
//!
//! Audio2Pose (from SadTalker) maps an audio window to 3DMM head-pose
//! motion — yaw/pitch/roll + translation — driven by the audio. This crate
//! runs the CVAE *decoder* as inference-only Burn code on `burn-wgpu` (GPU)
//! or `burn-flex` (CPU), using weights imported from ONNX
//! (`model/pose.rs` + `.bpk`, task 4).
//!
//! # Pipeline
//!
//! 1. The audio is embedded with the Wav2Lip audio encoder (produced
//!    upstream by the `wav2lip` crate) into 512-d embeddings, one per
//!    16-column mel window.
//! 2. [`Audio2Pose::decode`] slices the embeddings into 32-frame sequences,
//!    runs the decoder per sequence, and returns a `(frames, 6)` pose-motion
//!    sequence.
//!
//! The `ref` (source-face pose) defaults to zeros (neutral start) and the
//! `class` input is fixed to a single default class (0), per the plan. The
//! latent `z` is seeded optionally: `None` = sample `z ~ N(0,1)` per render,
//! `Some(seed)` = fixed reproducible latent.

pub mod model;

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use expnet::backend::Backend;

use model::Model;

/// Pose coefficient vector length (yaw/pitch/roll + tx/ty/tz).
pub const POSE_DIM: usize = 6;
/// Audio2Pose sequence length (frames processed per decoder call).
pub const SEQ_LEN: usize = 32;
/// Audio-embedding dimension (Wav2Lip audio encoder output).
pub const AUDIO_EMB_DIM: usize = 512;
/// Latent dimension of the CVAE decoder.
pub const LATENT_SIZE: usize = 64;

/// Audio2Pose inference wrapper.
///
/// Loads the converted `.bpk` weights and runs audio embeddings → head-pose
/// motion on the selected Burn device.
pub struct Audio2Pose {
    model: Model,
    /// Optional latent seed: `None` = sample `z ~ N(0,1)` per render,
    /// `Some(seed)` = fixed reproducible latent.
    seed: Option<u64>,
    device: Device,
}

impl Audio2Pose {
    /// Loads the Audio2Pose model from a `.bpk` weight file, auto-selecting
    /// the backend (GPU via burn-wgpu when available, CPU otherwise).
    pub fn new(weights: impl AsRef<Path>) -> Result<Self> {
        let backend = Backend::auto()?;
        Self::with_backend(weights, &backend)
    }

    /// Loads the Audio2Pose model on an explicitly chosen [`Backend`].
    pub fn with_backend(weights: impl AsRef<Path>, backend: &Backend) -> Result<Self> {
        let device = backend.device().clone();
        let model = Model::from_file(weights.as_ref(), &device);
        Ok(Self {
            model,
            seed: None,
            device,
        })
    }

    /// Sets the latent seed.
    ///
    /// `None` = sample `z ~ N(0,1)` per render (stochastic); `Some(n)` = a
    /// fixed, reproducible latent drawn from a seeded RNG.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the configured latent seed.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Runs the decoder on audio embeddings, one 32-frame sequence at a time.
    ///
    /// `audio_emb` is a flat `[n_frames, 512]` row-major sequence of Wav2Lip
    /// audio embeddings (one per video frame). Returns a flat
    /// `[n_frames, 6]` pose-motion sequence (yaw/pitch/roll + tx/ty/tz per
    /// frame).
    ///
    /// The last partial sequence is padded by repeating the first embedding
    /// of that sequence, then truncated back to the exact frame count.
    pub fn decode(&self, audio_emb: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(!audio_emb.is_empty(), "empty audio embeddings");
        anyhow::ensure!(
            audio_emb.len() % AUDIO_EMB_DIM == 0,
            "audio_emb length must be a multiple of {AUDIO_EMB_DIM}"
        );
        let n_frames = audio_emb.len() / AUDIO_EMB_DIM;

        // ref: zeros (neutral start), class fixed to 0.
        let ref_coeff = [0.0f32; POSE_DIM];

        let mut out = Vec::with_capacity(n_frames * POSE_DIM);

        for seq_start in (0..n_frames).step_by(SEQ_LEN) {
            let seq_len = (n_frames - seq_start).min(SEQ_LEN);
            // Build [1, SEQ_LEN, 512] audio embeddings, padding the last
            // partial sequence by repeating the first embedding of it.
            let mut emb_data = Vec::with_capacity(SEQ_LEN * AUDIO_EMB_DIM);
            for i in 0..SEQ_LEN {
                let src = seq_start + (i % seq_len);
                let base = src * AUDIO_EMB_DIM;
                emb_data.extend_from_slice(&audio_emb[base..base + AUDIO_EMB_DIM]);
            }
            let audio_t: Tensor<3> = Tensor::from_data(
                TensorData::new(emb_data, [1, SEQ_LEN, AUDIO_EMB_DIM]),
                &self.device,
            );
            let ref_t: Tensor<2> = Tensor::from_data(
                TensorData::new(ref_coeff.to_vec(), [1, POSE_DIM]),
                &self.device,
            );
            let z_t: Tensor<2> = Tensor::from_data(
                TensorData::new(self.sample_z(), [1, LATENT_SIZE]),
                &self.device,
            );

            let out_t = self.model.forward(ref_t, audio_t, z_t); // [1, 32, 6]
            let data = out_t.into_data();
            let arr = data.as_slice::<f32>().context("pose output not f32")?;
            // Take only the real frames of this sequence.
            for i in 0..seq_len {
                out.extend_from_slice(&arr[i * POSE_DIM..(i + 1) * POSE_DIM]);
            }
        }

        Ok(out)
    }

    /// Draws a latent `z` vector of length [`LATENT_SIZE`].
    ///
    /// `None` seed → `rand::random` (stochastic); `Some(seed)` → a fixed
    /// `rand::rngs::StdRng` seeded with the configured seed (reproducible).
    fn sample_z(&self) -> Vec<f32> {
        use rand::Rng;
        use rand::SeedableRng;
        match self.seed {
            None => (0..LATENT_SIZE).map(|_| rand::random::<f32>()).collect(),
            Some(seed) => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                (0..LATENT_SIZE).map(|_| rng.gen::<f32>()).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expnet::backend::BackendPreference;

    fn res_weights() -> Option<std::path::PathBuf> {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let candidate = dir.join(".res").join("models").join("pose").join("pose.bpk");
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    fn dummy_audio(n_frames: usize) -> Vec<f32> {
        // A slowly-varying deterministic embedding so different frames differ.
        (0..n_frames * AUDIO_EMB_DIM)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect()
    }

    /// T5: a single 32-frame sequence decodes to the correct shape and is
    /// finite on the CPU fallback backend.
    #[test]
    fn pose_runs_on_cpu() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = Audio2Pose::with_backend(&weights, &backend).unwrap();

        let emb = dummy_audio(SEQ_LEN);
        let pose = model.decode(&emb).unwrap();

        assert_eq!(pose.len(), SEQ_LEN * POSE_DIM);
        for v in &pose {
            assert!(v.is_finite(), "pose value not finite: {v}");
        }
    }

    /// T5: a partial sequence (e.g. 10 frames) decodes to exactly 10×6.
    #[test]
    fn pose_partial_sequence_shape() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = Audio2Pose::with_backend(&weights, &backend).unwrap();

        let n = 10;
        let emb = dummy_audio(n);
        let pose = model.decode(&emb).unwrap();

        assert_eq!(pose.len(), n * POSE_DIM);
        for v in &pose {
            assert!(v.is_finite(), "pose value not finite: {v}");
        }
    }

    /// T5: with a fixed seed, the same input gives byte-identical output.
    #[test]
    fn pose_deterministic_with_seed() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let a = Audio2Pose::with_backend(&weights, &backend)
            .unwrap()
            .with_seed(Some(42));
        let b = Audio2Pose::with_backend(&weights, &backend)
            .unwrap()
            .with_seed(Some(42));

        let emb = dummy_audio(SEQ_LEN);
        let pa = a.decode(&emb).unwrap();
        let pb = b.decode(&emb).unwrap();

        assert_eq!(pa, pb, "seeded poses should be identical");
    }

    /// T5: without a seed, two runs differ (stochastic latent).
    #[test]
    fn pose_stochastic_without_seed() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let a = Audio2Pose::with_backend(&weights, &backend).unwrap();
        let b = Audio2Pose::with_backend(&weights, &backend).unwrap();

        let emb = dummy_audio(SEQ_LEN);
        let pa = a.decode(&emb).unwrap();
        let pb = b.decode(&emb).unwrap();

        assert_ne!(pa, pb, "unseeded poses should differ (stochastic latent)");
    }
}
