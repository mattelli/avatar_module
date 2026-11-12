//! wav2lip — Wav2Lip lip-sync GAN inference (Plan 5).
//!
//! Wav2Lip (Rudrabha et al.) takes an audio window + a face frame and
//! re-renders the mouth to match the speech. This crate runs the generator
//! as inference-only Burn code on the `burn-wgpu` backend (GPU, with
//! `burn-flex` CPU fallback), using weights imported from ONNX via
//! `burn-onnx` (see the `scripts/convert_weights.py` pipeline, task 2).
//!
//! Status: scaffolding only (task 1). The model module, weight loading, and
//! preprocessing land in task 4.

/// Placeholder so the crate compiles before the model lands (task 4).
pub fn placeholder() -> &'static str {
    "wav2lip: not implemented yet (Plan 5, task 4)"
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_present() {
        assert!(super::placeholder().contains("wav2lip"));
    }
}
