//! Face detection + center-crop (plan 4).
//!
//! The expression atlas and renderer assume a centered, frame-filling face
//! (hardcoded normalized control points / mouth coordinates). When the face
//! is off-center or small in the source photo, the warp and lip-sync land on
//! the wrong pixels. This module detects the face bounding box in an RGBA
//! image with the pure-Rust SeetaFace detector (`rustface`) and returns a
//! margin-expanded, clamped crop rectangle so the face fills the working
//! frame before atlas warping.
//!
//! Fallback policy (user-confirmed): if the model file is missing, detection
//! fails, or no face is found, the caller uses the whole image — rendering
//! never fails.

use crate::atlas::load_image;
use image::ImageBuffer;
use image::Rgba;
use std::path::{Path, PathBuf};

/// Default margin (fraction of the face-box size) added on each side so the
/// forehead/hair/chin are not cut off.
pub const DEFAULT_FACE_CROP_MARGIN: f32 = 0.35;

/// Name of the bundled SeetaFace frontal detector model.
pub const FACE_MODEL_FILE: &str = "seeta_fd_frontal_v1.0.bin";

/// Resolves the `.res` asset directory by walking up from the crate manifest
/// dir (works for `cargo test`, the repo layout, and the installed wheel).
/// An explicit `AVATAR_RES_DIR` environment variable wins when set.
pub fn res_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("AVATAR_RES_DIR") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let candidate = dir.join(".res");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Path to the bundled face-detection model, if present.
pub fn face_model_path() -> Option<PathBuf> {
    let p = res_dir()?.join(FACE_MODEL_FILE);
    p.exists().then_some(p)
}

/// Runs the SeetaFace detector on a grayscale conversion of `img` and
/// returns every detection as `(bbox, score)`, sorted by descending score.
///
/// Returns an empty vec when the model cannot be loaded or the image is too
/// small for the detector. Never panics on the input image: a blank or tiny
/// image yields an empty vec rather than a detector panic.
pub fn detect_faces(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    model_path: &Path,
) -> Vec<(rustface::Rectangle, f64)> {
    let (w, h) = img.dimensions();
    // The detector's sliding window is at least 20 px; anything smaller
    // cannot contain a detectable face.
    if w < 20 || h < 20 {
        return Vec::new();
    }

    let Ok(mut detector) = rustface::create_detector(&model_path.to_string_lossy()) else {
        return Vec::new();
    };
    // Tuned like the rustface demo: lower threshold for recall, 0.8 pyramid
    // step, 4 px slide for a good speed/accuracy trade-off.
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    // rustface expects a single-channel grayscale image.
    let gray = image::imageops::grayscale(img);
    let data = gray.as_raw();
    let image_data = rustface::ImageData::new(data, w, h);
    let mut faces: Vec<(rustface::Rectangle, f64)> = detector
        .detect(&image_data)
        .into_iter()
        .map(|f| (*f.bbox(), f.score()))
        .collect();
    // Highest score first; ties broken by largest area.
    faces.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let area_a = a.0.width() * a.0.height();
                let area_b = b.0.width() * b.0.height();
                area_b.cmp(&area_a)
            })
    });
    faces
}

/// Detects the highest-scoring face in an RGBA image and returns its
/// bounding box in pixel coordinates.
///
/// Returns `None` when the model cannot be loaded, the image is too small for
/// the detector, or no face is found. Never panics on the input image: a
/// blank or tiny image yields `None` rather than a detector panic.
pub fn detect_face_box(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    model_path: &Path,
) -> Option<rustface::Rectangle> {
    detect_faces(img, model_path)
        .into_iter()
        .next()
        .map(|(bbox, _)| bbox)
}

/// Expands `bbox` by `margin` (a fraction of the box size) on each side and
/// clamps the result to the image bounds `(img_w, img_h)`.
///
/// Returns `(x, y, w, h)` with `w`/`h` at least 1. If the expanded box
/// overflows the image on both sides the result is clamped to the full image.
pub fn expand_margin(
    bbox: &rustface::Rectangle,
    img_w: u32,
    img_h: u32,
    margin: f32,
) -> (u32, u32, u32, u32) {
    let margin = margin.max(0.0);
    let bw = bbox.width() as i64;
    let bh = bbox.height() as i64;
    let dx = (bw as f32 * margin) as i64;
    let dy = (bh as f32 * margin) as i64;

    let x0 = (bbox.x() as i64 - dx).max(0) as u32;
    let y0 = (bbox.y() as i64 - dy).max(0) as u32;
    let x1 = (bbox.x() as i64 + bw + dx).min(img_w as i64) as u32;
    let y1 = (bbox.y() as i64 + bh + dy).min(img_h as i64) as u32;

    let w = (x1 - x0).max(1);
    let h = (y1 - y0).max(1);
    (x0, y0, w, h)
}

/// Crops `img` to the rectangle `(x, y, w, h)`.
pub fn crop(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    image::imageops::crop_imm(img, x, y, w, h).to_image()
}

/// Convenience: loads `path`, detects the face, and returns the margin-
/// expanded crop of the image. Returns the whole image unchanged when the
/// model is missing, detection fails, or no face is found.
pub fn face_crop(
    path: &Path,
    model_path: &Path,
    margin: f32,
) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let img = load_image(path)?;
    Ok(match detect_face_box(&img, model_path) {
        Some(bbox) => {
            let (w, h) = img.dimensions();
            let (x, y, cw, ch) = expand_margin(&bbox, w, h, margin);
            crop(&img, x, y, cw, ch)
        }
        None => img,
    })
}

/// Applies face-crop to a loaded image using the auto-resolved bundled model.
///
/// Returns the whole image unchanged when face-crop is disabled, the model is
/// missing, detection fails, or no face is found — rendering never fails.
pub fn apply_face_crop(
    img: ImageBuffer<Rgba<u8>, Vec<u8>>,
    enabled: bool,
    margin: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    if !enabled {
        return img;
    }
    let Some(model) = face_model_path() else {
        return img;
    };
    match detect_face_box(&img, &model) {
        Some(bbox) => {
            let (w, h) = img.dimensions();
            let (x, y, cw, ch) = expand_margin(&bbox, w, h, margin);
            crop(&img, x, y, cw, ch)
        }
        None => img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> Option<PathBuf> {
        face_model_path()
    }

    /// Blank image: no face → `None` (and no detector panic).
    #[test]
    fn blank_image_returns_none() {
        let Some(model) = model_path() else { return };
        let img = ImageBuffer::from_pixel(64, 64, Rgba([0, 0, 0, 255]));
        assert!(detect_face_box(&img, &model).is_none());
    }

    /// Tiny image (< 20 px) short-circuits to `None` without loading the model.
    #[test]
    fn tiny_image_returns_none() {
        let img = ImageBuffer::from_pixel(10, 10, Rgba([255, 255, 255, 255]));
        assert!(detect_face_box(&img, Path::new("does_not_exist.bin")).is_none());
    }

    /// A real photo with a face (`.res/avatar.jpg`) is detected and the
    /// margin-expanded crop is strictly smaller than the input and centered
    /// on the detected box.
    #[test]
    fn detects_face_in_photo() {
        let Some(model) = model_path() else { return };
        let Some(res) = res_dir() else { return };
        let photo = res.join("avatar.jpg");
        if !photo.exists() {
            return;
        }
        let img = load_image(&photo).unwrap();
        let (iw, ih) = img.dimensions();
        let bbox = detect_face_box(&img, &model).expect("face detected in avatar.jpg");
        assert!(bbox.width() > 0 && bbox.height() > 0);
        assert!(bbox.x() >= 0 && bbox.y() >= 0);
        assert!((bbox.x() as u32) + bbox.width() <= iw);
        assert!((bbox.y() as u32) + bbox.height() <= ih);

        let (x, y, w, h) = expand_margin(&bbox, iw, ih, DEFAULT_FACE_CROP_MARGIN);
        assert!(w > 0 && h > 0);
        assert!(w <= iw && h <= ih);
        // The crop must be smaller than the input (the face does not fill it).
        assert!(w < iw || h < ih, "crop should be smaller than input");
        // The crop contains the face box.
        assert!(x <= bbox.x() as u32);
        assert!(y <= bbox.y() as u32);
        assert!(x + w >= (bbox.x() as u32) + bbox.width());
        assert!(y + h >= (bbox.y() as u32) + bbox.height());
    }

    /// Margin expansion clamps to image bounds: a face at the corner grows
    /// only as far as the edges allow.
    #[test]
    fn expand_margin_clamps_to_bounds() {
        // Face in the top-left corner of a 100x100 image.
        let bbox = rustface::Rectangle::new(0, 0, 40, 40);
        let (x, y, w, h) = expand_margin(&bbox, 100, 100, 0.5);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
        assert_eq!(w, 60); // 40 + 20 margin, clamped to right edge
        assert_eq!(h, 60);

        // Face in the bottom-right corner.
        let bbox = rustface::Rectangle::new(60, 60, 40, 40);
        let (x, y, w, h) = expand_margin(&bbox, 100, 100, 0.5);
        assert_eq!(x, 40);
        assert_eq!(y, 40);
        assert_eq!(w, 60);
        assert_eq!(h, 60);
    }

    /// Margin expansion grows the box by the requested fraction.
    #[test]
    fn expand_margin_grows_by_fraction() {
        let bbox = rustface::Rectangle::new(20, 20, 40, 40);
        let (x, y, w, h) = expand_margin(&bbox, 200, 200, 0.25);
        assert_eq!(x, 10); // 20 - 40*0.25
        assert_eq!(y, 10);
        assert_eq!(w, 60); // 40 + 2*40*0.25
        assert_eq!(h, 60);
    }

    /// Zero/negative margin keeps the box as-is (clamped to bounds).
    #[test]
    fn expand_margin_zero_is_identity() {
        let bbox = rustface::Rectangle::new(10, 10, 30, 30);
        let (x, y, w, h) = expand_margin(&bbox, 100, 100, 0.0);
        assert_eq!((x, y, w, h), (10, 10, 30, 30));
    }

    /// A margin so large the box would overflow both sides clamps to the
    /// full image.
    #[test]
    fn expand_margin_huge_clamps_to_full_image() {
        let bbox = rustface::Rectangle::new(40, 40, 20, 20);
        let (x, y, w, h) = expand_margin(&bbox, 100, 100, 10.0);
        assert_eq!((x, y, w, h), (0, 0, 100, 100));
    }

    /// `apply_face_crop` with face-crop enabled crops a photo with a face to
    /// a region smaller than the input (T3/T4).
    #[test]
    fn apply_face_crop_enabled_crops_photo() {
        let Some(res) = res_dir() else { return };
        let photo = res.join("avatar.jpg");
        if !photo.exists() {
            return;
        }
        let img = load_image(&photo).unwrap();
        let (iw, ih) = img.dimensions();
        let cropped = apply_face_crop(img, true, DEFAULT_FACE_CROP_MARGIN);
        let (cw, ch) = cropped.dimensions();
        assert!(
            cw < iw || ch < ih,
            "crop should be smaller than input ({cw}x{ch} vs {iw}x{ih})"
        );
        assert!(cw > 0 && ch > 0);
    }

    /// `apply_face_crop` with face-crop disabled returns the whole image
    /// unchanged (T3).
    #[test]
    fn apply_face_crop_disabled_keeps_whole_image() {
        let Some(res) = res_dir() else { return };
        let photo = res.join("avatar.jpg");
        if !photo.exists() {
            return;
        }
        let img = load_image(&photo).unwrap();
        let (iw, ih) = img.dimensions();
        let out = apply_face_crop(img, false, DEFAULT_FACE_CROP_MARGIN);
        assert_eq!(out.dimensions(), (iw, ih));
    }

    /// `apply_face_crop` falls back to the whole image when the model file is
    /// missing (Q5): point AVATAR_RES_DIR at an empty dir.
    #[test]
    fn apply_face_crop_missing_model_keeps_whole_image() {
        let tmp = std::env::temp_dir().join("avatar_face_crop_empty_res");
        std::fs::create_dir_all(&tmp).unwrap();
        std::env::set_var("AVATAR_RES_DIR", &tmp);
        let img = crate::atlas::test_face_cpu(64, 64);
        let out = apply_face_crop(img, true, DEFAULT_FACE_CROP_MARGIN);
        std::env::remove_var("AVATAR_RES_DIR");
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(out.dimensions(), (64, 64));
    }
}
