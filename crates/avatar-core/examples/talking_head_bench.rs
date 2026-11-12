//! Benchmarks the talking-head render on GPU (burn-wgpu) vs CPU (burn-flex)
//! for a short clip, with per-stage timing. Run:
//!   cargo run -p avatar-core --features talking-head --example talking_head_bench

use std::time::Instant;

use avatar_core::{
    atlas::test_face_cpu,
    fusion::FusionConfig,
    talking_head::{MODEL_SAMPLE_RATE, render_talking_head},
};
use expnet::backend::BackendPreference;
use image::{ImageBuffer, Rgba};

fn bench(pref: BackendPreference, label: &str, secs: f32) {
    let sr = MODEL_SAMPLE_RATE;
    let n = (secs * sr as f32) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
        .collect();

    let face: ImageBuffer<Rgba<u8>, Vec<u8>> = test_face_cpu(128, 128);
    let pic_path = std::env::temp_dir().join("avatar_bench_face.png");
    avatar_core::atlas::save_image(&face, &pic_path).unwrap();

    let cfg = FusionConfig::default().with_size(256);
    let t0 = Instant::now();
    let frames = render_talking_head(&samples, sr, Some(&pic_path), &cfg, pref).unwrap();
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() / frames.len().max(1) as f64;
    println!(
        "{label:<12} {secs:.1}s audio -> {} frames in {dt:.2?} ({per:.3}s/frame)",
        frames.len()
    );
    let _ = std::fs::remove_file(&pic_path);
}

fn main() {
    bench(BackendPreference::Auto, "GPU", 1.0);
    bench(BackendPreference::Cpu, "CPU", 0.2); // CPU is slow; keep it short
}
