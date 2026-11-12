//! Full public render() smoke: Piper TTS -> talking-head -> MP4 (256x256).
//! Requires the `talking-head` feature + model weights + a voice model.
//!   cargo run -p avatar-core --features talking-head --example render_talking_head_api

use std::path::PathBuf;

use avatar_core::pipeline::{EngineConfig, render};

fn res_dir() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let c = dir.join(".res");
        if c.is_dir() {
            return Some(c);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn main() {
    let Some(res) = res_dir() else {
        println!("no .res dir");
        return;
    };
    let model = res.join("en_US-hfc_female-medium.onnx");
    if !model.exists() {
        println!("no voice model");
        return;
    }
    let out = std::env::temp_dir().join("avatar_render_talking_head_api.mp4");
    let cfg = EngineConfig::voice(&model)
        .with_talking_head(true)
        .with_backend(avatar_core::pipeline::BackendPref::Gpu);
    let n = render("Hello world.", &out, &cfg).unwrap();
    let size = std::fs::metadata(&out).unwrap().len();
    println!("rendered {n} frames -> {} ({size} bytes)", out.display());
    let _ = std::fs::remove_file(&out);
}
