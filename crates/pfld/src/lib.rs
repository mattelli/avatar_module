//! pfld — PFLD 98-landmark detector inference (Plan 7, task 2).
//!
//! PFLD (Practical Facial Landmark Detector) maps a 112×112 face image to 98
//! facial landmarks (x, y), normalized to 0..1. This crate runs the backbone
//! as inference-only Burn code on `burn-wgpu` (GPU) or `burn-flex` (CPU),
//! using weights imported from ONNX (`model/pfld.rs` + `.bpk`, task 1).
//!
//! The landmarks are used to place the **expression-warp control points** on
//! the actual face (replacing the hardcoded canonical positions). The backbone
//! also returns an auxiliary feature map (`out1`) used only during training;
//! inference uses only the `landmarks` output.
//!
//! # Pipeline
//!
//! 1. The caller supplies a 112×112 RGB face (already cropped/centred).
//! 2. [`PFLD::predict`] runs the backbone and returns 98 `(x, y)` points in
//!    0..1 (normalized to the input image).
//! 3. [`landmarks_to_control_points`] maps the 98 points to the 9 control
//!    points used by the expression warp (WFLW layout).

pub mod model;

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use expnet::backend::Backend;

use model::Model;

/// PFLD input resolution (square, px).
pub const INPUT_SIZE: usize = 112;
/// Number of landmarks returned by PFLD.
pub const NUM_LANDMARKS: usize = 98;
/// Number of control points used by the expression warp.
pub const NUM_CONTROL_POINTS: usize = 9;

/// PFLD inference wrapper.
///
/// Loads the converted `.bpk` weights and runs 112×112 face → 98 landmarks on
/// the selected Burn device.
pub struct PFLD {
    model: Model,
    device: Device,
}

impl PFLD {
    /// Loads the PFLD model from a `.bpk` weight file, auto-selecting the
    /// backend (GPU via burn-wgpu when available, CPU otherwise).
    pub fn new(weights: impl AsRef<Path>) -> Result<Self> {
        let backend = Backend::auto()?;
        Self::with_backend(weights, &backend)
    }

    /// Loads the PFLD model on an explicitly chosen [`Backend`].
    pub fn with_backend(weights: impl AsRef<Path>, backend: &Backend) -> Result<Self> {
        let device = backend.device().clone();
        let model = Model::from_file(weights.as_ref(), &device);
        Ok(Self { model, device })
    }

    /// Runs PFLD on a 112×112 RGB image.
    ///
    /// `rgb` is a flat `[112, 112, 3]` row-major buffer with values in `0..1`
    /// (HWC). Returns 98 `(x, y)` landmarks, each normalized to `0..1`.
    pub fn predict(&self, rgb: &[f32]) -> Result<Vec<[f32; 2]>> {
        anyhow::ensure!(
            rgb.len() == INPUT_SIZE * INPUT_SIZE * 3,
            "expected {} RGB values, got {}",
            INPUT_SIZE * INPUT_SIZE * 3,
            rgb.len()
        );

        // HWC -> CHW [1, 3, 112, 112].
        let mut chw = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let hwc = (y * INPUT_SIZE + x) * 3;
                let plane = y * INPUT_SIZE + x;
                chw[plane] = rgb[hwc];
                chw[INPUT_SIZE * INPUT_SIZE + plane] = rgb[hwc + 1];
                chw[2 * INPUT_SIZE * INPUT_SIZE + plane] = rgb[hwc + 2];
            }
        }

        let input: Tensor<4> = Tensor::from_data(
            TensorData::new(chw, [1, 3, INPUT_SIZE, INPUT_SIZE]),
            &self.device,
        );

        let (_out1, landmarks) = self.model.forward(input);
        let data = landmarks.into_data();
        let arr = data.as_slice::<f32>().context("pfld landmarks not f32")?;
        anyhow::ensure!(
            arr.len() == NUM_LANDMARKS * 2,
            "expected {} landmark values, got {}",
            NUM_LANDMARKS * 2,
            arr.len()
        );

        Ok((0..NUM_LANDMARKS)
            .map(|i| [arr[2 * i], arr[2 * i + 1]])
            .collect())
    }
}

/// WFLW 98-point layout → the 9 expression-warp control points.
///
/// The WFLW layout (used by PFLD) groups the 98 points as (verified against
/// the released checkpoint's output):
///
/// - `0..33`  face contour (33; chin is the lowest point, index 15)
/// - `33..42` left eyebrow (9)
/// - `42..51` right eyebrow (9)
/// - `51..60` nose (9)
/// - `60..68` left eye (8)
/// - `68..76` right eye (8)
/// - `76..89` outer mouth (13; corners 76 left, 82 right)
/// - `89..96` inner mouth (7)
/// - `96..98` pupils (left 96, right 97)
///
/// Returns the 9 control points in the order used by the expression warp:
/// `[mouth center, mouth left, mouth right, brow left, brow right,
///   eye left, eye right, nose, chin]`.
pub fn landmarks_to_control_points(landmarks: &[[f32; 2]]) -> Result<[[f32; 2]; NUM_CONTROL_POINTS]> {
    anyhow::ensure!(
        landmarks.len() == NUM_LANDMARKS,
        "expected {NUM_LANDMARKS} landmarks, got {}",
        landmarks.len()
    );

    let mean = |idx: &[usize]| -> [f32; 2] {
        let n = idx.len() as f32;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for &i in idx {
            sx += landmarks[i][0];
            sy += landmarks[i][1];
        }
        [sx / n, sy / n]
    };

    // Inner mouth 89..96 → mouth center.
    let mouth_center = mean(&(89..96).collect::<Vec<_>>());
    // Outer mouth corners (76 left, 82 right).
    let mouth_left = landmarks[76];
    let mouth_right = landmarks[82];
    // Brows.
    let brow_left = mean(&(33..42).collect::<Vec<_>>());
    let brow_right = mean(&(42..51).collect::<Vec<_>>());
    // Eyes: pupils (96, 97).
    let eye_left = landmarks[96];
    let eye_right = landmarks[97];
    // Nose: lowest-centre point of the nose group (51..60).
    let nose = landmarks[57];
    // Chin: lowest contour point.
    let chin = landmarks[15];

    Ok([
        mouth_center,
        mouth_left,
        mouth_right,
        brow_left,
        brow_right,
        eye_left,
        eye_right,
        nose,
        chin,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use expnet::backend::BackendPreference;

    fn res_weights() -> Option<std::path::PathBuf> {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let candidate = dir.join(".res").join("models").join("pfld").join("pfld.bpk");
            if candidate.exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    /// A deterministic 112×112 RGB image (a smooth gradient) in 0..1.
    fn dummy_image() -> Vec<f32> {
        (0..INPUT_SIZE * INPUT_SIZE * 3)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.5 + 0.5)
            .collect()
    }

    /// T2: PFLD runs on the CPU fallback and returns 98 finite, in-range
    /// landmarks.
    #[test]
    fn pfld_runs_on_cpu() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = PFLD::with_backend(&weights, &backend).unwrap();

        let lm = model.predict(&dummy_image()).unwrap();
        assert_eq!(lm.len(), NUM_LANDMARKS);
        for (i, p) in lm.iter().enumerate() {
            assert!(p[0].is_finite() && p[1].is_finite(), "landmark {i} not finite");
            assert!(
                (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]),
                "landmark {i} out of range: {p:?}"
            );
        }
    }

    /// T2: prediction is deterministic (no randomness).
    #[test]
    fn pfld_deterministic() {
        let Some(weights) = res_weights() else { return };
        let backend = Backend::select(BackendPreference::Cpu).unwrap();
        let model = PFLD::with_backend(&weights, &backend).unwrap();

        let img = dummy_image();
        let a = model.predict(&img).unwrap();
        let b = model.predict(&img).unwrap();
        assert_eq!(a, b, "PFLD output should be deterministic");
    }

    /// T2: the 98 → 9 mapping picks the expected points.
    #[test]
    fn control_point_mapping() {
        // Distinct, recognizable coordinates per index.
        let mut lm = [[0.0f32; 2]; NUM_LANDMARKS];
        for (i, p) in lm.iter_mut().enumerate() {
            p[0] = i as f32 / 100.0;
            p[1] = (i as f32 / 100.0) + 0.5;
        }

        let cp = landmarks_to_control_points(&lm).unwrap();
        assert_eq!(cp.len(), NUM_CONTROL_POINTS);

        // mouth left = index 76, mouth right = 82.
        assert_eq!(cp[1], lm[76]);
        assert_eq!(cp[2], lm[82]);
        // eyes = pupils 96, 97.
        assert_eq!(cp[5], lm[96]);
        assert_eq!(cp[6], lm[97]);
        // nose = 57, chin = 15.
        assert_eq!(cp[7], lm[57]);
        assert_eq!(cp[8], lm[15]);
        // mouth center = mean of 89..96.
        let exp_x: f32 = (89..96).map(|i| lm[i][0]).sum::<f32>() / 7.0;
        assert!((cp[0][0] - exp_x).abs() < 1e-6);
    }

    /// T2: the mapping rejects a wrong-length landmark list.
    #[test]
    fn control_point_mapping_rejects_bad_len() {
        let lm = vec![[0.0f32; 2]; 10];
        assert!(landmarks_to_control_points(&lm).is_err());
    }
}
