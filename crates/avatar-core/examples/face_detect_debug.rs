//! Debug tool: prints face-detection results for any image.
//!
//! Shows exactly what the pipeline's face-crop step sees: whether a face is
//! detected, its bounding box and score, and the margin-expanded crop rect
//! that would be fed to the atlas. If no face is detected, the pipeline uses
//! the whole image (the fallback), which is why the output may look
//! off-center.
//!
//! Usage (from the workspace root):
//!     cargo run -p avatar-core --example face_detect_debug -- <image> [margin]
//!
//! Examples:
//!     cargo run -p avatar-core --example face_detect_debug -- .res/avatar.jpg
//!     cargo run -p avatar-core --example face_detect_debug -- my_photo.png 0.35
//!
//! Optionally writes `<image>.detect.png` with the detected boxes drawn in
//! red and the crop rect in green (only when a face is found).

use avatar_core::atlas::load_image;
use avatar_core::face_crop::{detect_faces, expand_margin, face_model_path, DEFAULT_FACE_CROP_MARGIN};
use image::{ImageBuffer, Rgba};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <image> [margin]\n  margin: crop margin as a fraction of the face-box size (default {DEFAULT_FACE_CROP_MARGIN})",
            args[0]
        );
        std::process::exit(2);
    }
    let img_path = Path::new(&args[1]);
    let margin: f32 = args
        .get(2)
        .map(|s| s.parse().unwrap_or(DEFAULT_FACE_CROP_MARGIN))
        .unwrap_or(DEFAULT_FACE_CROP_MARGIN);

    let img = load_image(img_path)?;
    let (iw, ih) = img.dimensions();
    println!("image: {} ({}x{})", img_path.display(), iw, ih);

    let Some(model) = face_model_path() else {
        println!("RESULT: model file not found -> pipeline uses the WHOLE image (fallback)");
        println!("  looked for {} next to the crate's .res dir (or $AVATAR_RES_DIR)", avatar_core::face_crop::FACE_MODEL_FILE);
        return Ok(());
    };
    println!("model: {}", model.display());

    let faces = detect_faces(&img, &model);
    if faces.is_empty() {
        println!("RESULT: NO FACE DETECTED -> pipeline uses the WHOLE image (fallback)");
        println!("  the output will NOT be centered on a face");
        return Ok(());
    }

    println!("RESULT: {} face(s) detected", faces.len());
    for (i, (bbox, score)) in faces.iter().enumerate() {
        println!(
            "  face {}: x={} y={} w={} h={} score={:.3}",
            i + 1,
            bbox.x(),
            bbox.y(),
            bbox.width(),
            bbox.height(),
            score
        );
    }

    // The pipeline uses the highest-scoring face.
    let (best, score) = &faces[0];
    let (x, y, w, h) = expand_margin(best, iw, ih, margin);
    println!("best face (score {score:.3}):");
    println!("  bbox:  x={} y={} w={} h={}", best.x(), best.y(), best.width(), best.height());
    println!("  crop (margin {margin}): x={x} y={y} w={w} h={h}");
    println!(
        "  crop is {}% of the image area",
        (w as f64 * h as f64 / (iw as f64 * ih as f64) * 100.0).round() as i64
    );

    // Optional debug overlay: red boxes = detections, green box = crop.
    let mut overlay = img.clone();
    for (bbox, _) in &faces {
        draw_rect(&mut overlay, bbox.x(), bbox.y(), bbox.width(), bbox.height(), [255, 0, 0, 255]);
    }
    draw_rect(&mut overlay, x as i32, y as i32, w, h, [0, 255, 0, 255]);
    let out_path = img_path.with_extension("detect.png");
    overlay.save(&out_path)?;
    println!("debug overlay written to {}", out_path.display());

    Ok(())
}

/// Draws a 2 px hollow rectangle (clamped to the image bounds).
fn draw_rect(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, x: i32, y: i32, w: u32, h: u32, color: [u8; 4]) {
    let (iw, ih) = img.dimensions();
    let x0 = x.max(0) as u32;
    let y0 = y.max(0) as u32;
    let x1 = (x + w as i32).min(iw as i32).max(0) as u32;
    let y1 = (y + h as i32).min(ih as i32).max(0) as u32;
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for px in x0..x1 {
        for (yy, c) in [(y0, color), ((y0 + 1).min(ih - 1), color), (y1.saturating_sub(1), color), (y1.saturating_sub(2).min(ih - 1), color)] {
            if yy < ih {
                img.put_pixel(px, yy, Rgba(c));
            }
        }
    }
    for py in y0..y1 {
        for (xx, c) in [(x0, color), ((x0 + 1).min(iw - 1), color), (x1.saturating_sub(1), color), (x1.saturating_sub(2).min(iw - 1), color)] {
            if xx < iw {
                img.put_pixel(xx, py, Rgba(c));
            }
        }
    }
}
