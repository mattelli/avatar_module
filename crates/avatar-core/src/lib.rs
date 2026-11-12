pub mod analysis;
pub mod atlas;
pub mod encoder;
pub mod face_crop;
pub mod markers;
pub mod pipeline;
pub mod renderer;
pub mod tts;
pub mod visemes;

// Plan 5 — Wav2Lip + ExpNet fusion (task 6) and the talking-head render path
// (task 8), only when the Burn crates are enabled via `talking-head`.
#[cfg(feature = "talking-head")]
pub mod fusion;
#[cfg(feature = "talking-head")]
pub mod talking_head;

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
