//! lipsync-train — placeholder for Plan 2 (`PLAN_LIPSYNC_TRAINING.md`).
//!
//! Will implement: dataset preparation (GRID/TIMIT), Burn model training on
//! burn-wgpu, evaluation, and export of `visemes.burn` weights consumed by
//! `avatar-core` as a drop-in replacement for the deterministic phoneme→viseme
//! mapping.

/// Placeholder so the crate compiles before Plan 2 lands.
pub fn placeholder() -> &'static str {
    "lipsync-train: not implemented yet (Plan 2)"
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_present() {
        assert!(super::placeholder().contains("lipsync-train"));
    }
}
