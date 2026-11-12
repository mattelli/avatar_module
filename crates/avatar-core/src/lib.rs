pub mod analysis;
pub mod atlas;
pub mod encoder;
pub mod face_crop;
pub mod markers;
pub mod pipeline;
pub mod renderer;
pub mod tts;
pub mod visemes;

// Plan 5, task 6 — Wav2Lip + ExpNet fusion (only when the Burn crates are
// enabled via the `talking-head` feature).
#[cfg(feature = "talking-head")]
pub mod fusion;

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
