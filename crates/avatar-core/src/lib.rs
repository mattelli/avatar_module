pub mod analysis;
pub mod atlas;
pub mod encoder;
pub mod face_crop;
pub mod markers;
pub mod pipeline;
pub mod tts;

// Plan 5 — Wav2Lip + ExpNet fusion (task 6) and the talking-head render path
// (task 8). The talking-head backend is the only backend (classic removed).
pub mod fusion;
pub mod talking_head;

// Re-export the ExpNet backend types so downstream crates (avatar-py) can
// name them without depending on expnet directly.
pub use expnet::backend as expnet_backend;

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