//! expnet — ExpNet audio-to-expression inference (Plan 5).
//!
//! ExpNet (from SadTalker) maps an audio window to 3DMM expression
//! coefficients — the non-lip facial motion (eyebrows, cheeks, subtle
//! expression changes) driven by the audio. This crate runs it as
//! inference-only Burn code on `burn-wgpu` (GPU, `burn-flex` CPU fallback),
//! using weights imported from ONNX via `burn-onnx` (task 2).
//!
//! Status: scaffolding only (task 1). The model module, weight loading, and
//! preprocessing land in task 3.

/// Placeholder so the crate compiles before the model lands (task 3).
pub fn placeholder() -> &'static str {
    "expnet: not implemented yet (Plan 5, task 3)"
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_present() {
        assert!(super::placeholder().contains("expnet"));
    }
}
