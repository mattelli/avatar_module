//! avatar-core — pipeline for text -> MP4 avatar video.
//!
//! Components (implemented by subsequent tasks):
//! - marker parser (task 2)
//! - Piper TTS integration (task 3)
//! - audio analysis / RMS envelope (task 4)
//! - deterministic phoneme -> viseme mapping (task 5)
//! - expression atlas (task 6)
//! - wgpu renderer (task 7)
//! - H.264 + AAC + MP4 mux (task 8)
//! - pipeline glue (task 9)

pub mod analysis;
pub mod markers;
pub mod tts;
pub mod visemes;

/// Placeholder module so the crate compiles before the first real component lands.
pub mod placeholder {
    /// Returns the crate version.
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::placeholder::version().is_empty());
    }
}
