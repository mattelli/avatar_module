//! Wav2Lip + ExpNet fusion stage (Plan 5, task 6).
//!
//! Combines, per video frame, the two talking-head signals:
//!
//! - **ExpNet** supplies 3DMM-style expression coefficients (64 per frame)
//!   that drive the *non-lip* facial motion (brows, eyes, mouth corners,
//!   cheeks) — the audio-driven expression deformation.
//! - **Wav2Lip** supplies a precise 96×96 mouth patch re-rendered to match
//!   the speech — the *lip articulation*.
//!
//! The fusion is:
//!
//! 1. **Expression warp**: the coefficients are mapped to displacements on
//!    the existing atlas control points ([`crate::atlas`]), then the source
//!    face is warped with `atlas::warp_cpu`. The residual / low-frequency
//!    coefficients drive the coarse expression; we use a small projection to
//!    the 7-parameter mouth/brow/eye/chin motion.
//! 2. **Mouth blend**: the Wav2Lip mouth patch is alpha-blended into the
//!    lower-face region of the warped frame through a smooth elliptical
//!    mask, so the precise lips replace the atlas's deterministic mouth.
//!
//! The output is the photorealistic talking-head frame at the configured
//! resolution (task 9 sets it to the source-image size; 256×256 per the
//! user decision).
//!
//! [`crate::atlas`]: crate::atlas

use image::{ImageBuffer, Rgba};

use crate::atlas::{ControlPoint, WarpParams, warp_cpu};

/// Control-point index for the mouth center.
const MOUTH_CENTER: usize = 0;
/// Control-point index for the mouth left corner.
const MOUTH_LEFT: usize = 1;
/// Control-point index for the mouth right corner.
const MOUTH_RIGHT: usize = 2;
/// Control-point index for the left brow.
const BROW_LEFT: usize = 3;
/// Control-point index for the right brow.
const BROW_RIGHT: usize = 4;
/// Control-point index for the left eye.
const EYE_LEFT: usize = 5;
/// Control-point index for the right eye.
const EYE_RIGHT: usize = 6;
/// Control-point index for the chin.
const CHIN: usize = 8;

/// Configuration for the fusion stage.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Output frame size (square). Defaults to 256 (user decision).
    pub size: u32,
    /// Overall warp strength for the expression deformation.
    pub strength: f32,
    /// Vertical fraction of the frame the mouth mask covers (from the bottom).
    pub mouth_mask_height: f32,
    /// Horizontal fraction of the frame the mouth mask covers (centered).
    pub mouth_mask_width: f32,
    /// Feather (smooth edge) of the mouth mask, as a fraction of the mask.
    pub mask_feather: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            size: 256,
            strength: 0.5,
            mouth_mask_height: 0.35,
            mouth_mask_width: 0.55,
            mask_feather: 0.3,
        }
    }
}

impl FusionConfig {
    /// Output size (square) — task 9 sets this to the source-image size.
    pub fn with_size(mut self, size: u32) -> Self {
        self.size = size;
        self
    }

    /// Expression warp strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }
}

/// The fused per-frame output.
pub struct FusedFrame {
    /// RGBA frame at `FusionConfig::size × size`.
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

/// Maps 64 ExpNet expression coefficients to a set of control-point
/// displacements, then warps `src` (the atlas face) into an expression frame.
///
/// The mapping is deliberately coarse: 3DMM expression coefficients are
/// abstract (they need a 3DMM renderer for full fidelity). Here we project
/// them onto the atlas's 9 control points:
///
/// - the first few coefficients drive the mouth (open / corners),
/// - the next drive the brows (raise / furrow),
/// - the next drive the eyes (squint),
/// - the tail drives the chin (jaw).
///
/// `strength` scales the displacement; `exp_coeffs` must be 64 long.
pub fn warp_from_coeffs(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    exp_coeffs: &[f32],
    strength: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut pts = face_points_neutral();
    // Mouth: coeff[0] opens (vertical), coeff[1]/[2] pull the corners.
    let mouth_open = exp_coeffs.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let mouth_l = exp_coeffs.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let mouth_r = exp_coeffs.get(2).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    // Brows: coeff[3]/[4] raise the brows (negative = furrow).
    let brow_l = exp_coeffs.get(3).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let brow_r = exp_coeffs.get(4).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    // Eyes: coeff[5]/[6] squint (negative = open).
    let eye_l = exp_coeffs.get(5).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let eye_r = exp_coeffs.get(6).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    // Chin: coeff[7] opens the jaw (down).
    let chin = exp_coeffs.get(7).copied().unwrap_or(0.0).clamp(-1.0, 1.0);

    // Mouth center down/up with the opening.
    pts[MOUTH_CENTER].disp = [0.0, -mouth_open * 0.02];
    // Mouth corners: spread out and slightly up when open.
    pts[MOUTH_LEFT].disp = [-mouth_open * 0.012, -mouth_l * 0.02];
    pts[MOUTH_RIGHT].disp = [mouth_open * 0.012, -mouth_r * 0.02];
    // Brows: raise (negative coeff => up). Map positive brow activation to up.
    pts[BROW_LEFT].disp = [0.0, -brow_l * 0.015];
    pts[BROW_RIGHT].disp = [0.0, -brow_r * 0.015];
    // Eyes: squint pulls the upper eyelid down (positive => down).
    pts[EYE_LEFT].disp = [0.0, eye_l * 0.008];
    pts[EYE_RIGHT].disp = [0.0, eye_r * 0.008];
    // Chin: down (open jaw).
    pts[CHIN].disp = [0.0, chin * 0.02];

    let params = WarpParams {
        strength,
        points: pts,
    };
    warp_cpu(src, &params)
}

/// Neutral control points (all zero displacement) — the `face_points` list.
fn face_points_neutral() -> Vec<ControlPoint> {
    vec![
        ControlPoint { pos: [0.5, 0.72], disp: [0.0, 0.0] }, // mouth center
        ControlPoint { pos: [0.42, 0.70], disp: [0.0, 0.0] }, // mouth left
        ControlPoint { pos: [0.58, 0.70], disp: [0.0, 0.0] }, // mouth right
        ControlPoint { pos: [0.35, 0.45], disp: [0.0, 0.0] }, // brow left
        ControlPoint { pos: [0.65, 0.45], disp: [0.0, 0.0] }, // brow right
        ControlPoint { pos: [0.35, 0.52], disp: [0.0, 0.0] }, // eye left
        ControlPoint { pos: [0.65, 0.52], disp: [0.0, 0.0] }, // eye right
        ControlPoint { pos: [0.5, 0.60], disp: [0.0, 0.0] },  // nose
        ControlPoint { pos: [0.5, 0.88], disp: [0.0, 0.0] },  // chin
    ]
}

/// Blends a Wav2Lip mouth patch into the lower-face region of `frame`.
///
/// The patch (`96×96×3`, values 0..1) is resized to the mask region of the
/// frame and alpha-blended through a smooth elliptical mask centered on the
/// mouth. Returns a new frame.
pub fn blend_mouth(
    frame: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    mouth_patch: &[f32],
    cfg: &FusionConfig,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (fw, fh) = frame.dimensions();
    let mask_w = (fw as f32 * cfg.mouth_mask_width) as u32;
    let mask_h = (fh as f32 * cfg.mouth_mask_height) as u32;
    let mask_w = mask_w.max(1);
    let mask_h = mask_h.max(1);
    // Centered horizontally, anchored at the bottom (mouth region).
    let x0 = (fw - mask_w) / 2;
    let y0 = fh - mask_h;

    // Resize the 96×96 patch to the mask region.
    let patch_img = patch_to_image(mouth_patch); // 96x96 Rgba
    let resized = image::imageops::resize(
        &patch_img,
        mask_w,
        mask_h,
        image::imageops::FilterType::Triangle,
    );

    let mut out = frame.clone();
    let feather = cfg.mask_feather.clamp(0.0, 1.0);
    let cx = x0 as f32 + mask_w as f32 / 2.0;
    let cy = y0 as f32 + mask_h as f32 / 2.0;
    let rx = mask_w as f32 / 2.0;
    let ry = mask_h as f32 / 2.0;

    for y in 0..mask_h {
        for x in 0..mask_w {
            let fx = x0 + x;
            let fy = y0 + y;
            // Normalized distance inside the ellipse (0 center, 1 edge).
            let nx = (fx as f32 - cx) / rx;
            let ny = (fy as f32 - cy) / ry;
            let d = (nx * nx + ny * ny).sqrt();
            if d > 1.0 {
                continue;
            }
            // Smooth alpha: 1 in the core, feather to 0 near the edge.
            let edge = (1.0 - feather).max(0.0);
            let alpha = if d <= edge {
                1.0
            } else {
                // Feather band: edge..1 -> alpha 1..0.
                let t = (d - edge) / (1.0 - edge).max(1e-6);
                1.0 - t
            };
            let p = resized.get_pixel(x, y);
            let dst = out.get_pixel_mut(fx, fy);
            let a = alpha;
            for c in 0..3 {
                let blend = p[c] as f32 * a + dst[c] as f32 * (1.0 - a);
                dst[c] = blend.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Converts a flat 96×96×3 mouth patch (0..1) into an RGBA image.
fn patch_to_image(patch: &[f32]) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_pixel(96, 96, Rgba([0u8, 0, 0, 255]));
    for y in 0..96usize {
        for x in 0..96usize {
            let i = (y * 96 + x) * 3;
            img.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    (patch[i].clamp(0.0, 1.0) * 255.0) as u8,
                    (patch[i + 1].clamp(0.0, 1.0) * 255.0) as u8,
                    (patch[i + 2].clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ]),
            );
        }
    }
    img
}

/// Full per-frame fusion: expression-warp the source face, then blend the
/// Wav2Lip mouth patch in.
///
/// - `src`: the avatar source face (already face-cropped).
/// - `exp_coeffs`: 64 ExpNet expression coefficients for this frame.
/// - `mouth_patch`: the 96×96×3 Wav2Lip mouth patch (values 0..1).
///
/// The frame is resized to `cfg.size` at the end.
pub fn fuse_frame(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    exp_coeffs: &[f32],
    mouth_patch: &[f32],
    cfg: &FusionConfig,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let warped = warp_from_coeffs(src, exp_coeffs, cfg.strength);
    let blended = blend_mouth(&warped, mouth_patch, cfg);
    if blended.dimensions() == (cfg.size, cfg.size) {
        blended
    } else {
        image::imageops::resize(
            &blended,
            cfg.size,
            cfg.size,
            image::imageops::FilterType::Triangle,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::test_face_cpu;

    /// T6: the fused output contains BOTH signals —
    /// 1. the ExpNet expression deformation (a nonzero coefficient changes the
    ///    warp vs. neutral), and
    /// 2. the Wav2Lip mouth patch (a bright patch changes the lower-face).
    #[test]
    fn fusion_contains_both_signals() {
        let src = test_face_cpu(128, 128);
        let cfg = FusionConfig::default().with_size(128);

        // Neutral coefficients (all zero) + a dark patch.
        let neutral = [0.0f32; 64];
        let dark_patch = vec![0.0f32; 96 * 96 * 3];
        let base = fuse_frame(&src, &neutral, &dark_patch, &cfg);

        // 1. ExpNet signal: a mouth-open coefficient (index 0 = 1.0) warps the
        //    mouth, so the fused frame differs from neutral.
        let mut coeffs = [0.0f32; 64];
        coeffs[0] = 1.0;
        let exp_warped = fuse_frame(&src, &coeffs, &dark_patch, &cfg);
        let exp_diff = pixel_diff(&base, &exp_warped);
        assert!(exp_diff > 0.0, "ExpNet signal should change the frame");

        // 2. Wav2Lip signal: a bright mouth patch changes the lower-face.
        let bright_patch = vec![1.0f32; 96 * 96 * 3];
        let lip_blended = fuse_frame(&src, &neutral, &bright_patch, &cfg);
        let lip_diff = pixel_diff(&base, &lip_blended);
        assert!(lip_diff > 0.0, "Wav2Lip signal should change the frame");
    }

    /// A nonzero brow coefficient (index 3) moves the brows (expression).
    #[test]
    fn brow_coeff_moves_brows() {
        let src = test_face_cpu(128, 128);
        let cfg = FusionConfig::default().with_size(128);
        let dark = vec![0.0f32; 96 * 96 * 3];

        let c0 = [0.0f32; 64];
        let mut c1 = [0.0f32; 64];
        c1[3] = 1.0; // brow raise
        let a = fuse_frame(&src, &c0, &dark, &cfg);
        let b = fuse_frame(&src, &c1, &dark, &cfg);
        assert!(pixel_diff(&a, &b) > 0.0);
    }

    /// Neutral coeffs + a black mouth patch darken the mouth mask region.
    #[test]
    fn black_patch_darkens_mouth_region() {
        let src = ImageBuffer::from_pixel(128, 128, Rgba([255u8, 255, 255, 255]));
        let cfg = FusionConfig::default().with_size(128);
        let neutral = [0.0f32; 64];
        let black = vec![0.0f32; 96 * 96 * 3];

        let out = fuse_frame(&src, &neutral, &black, &cfg);
        assert_eq!(out.dimensions(), (128, 128));

        // The mouth region (bottom-center) should be dark from the black patch.
        let (fw, fh) = out.dimensions();
        let mouth_y = (fh as f32 * 0.85) as u32;
        let mouth_x = fw / 2;
        let p = out.get_pixel(mouth_x, mouth_y);
        assert!(
            p[0] < 100,
            "mouth region should be darkened, got {}",
            p[0]
        );
        // The top of the frame stays white (mask only covers the lower face).
        let top = out.get_pixel(fw / 2, 5);
        assert!(top[0] > 200, "top of frame should stay white, got {}", top[0]);
    }

    fn pixel_diff(a: &ImageBuffer<Rgba<u8>, Vec<u8>>, b: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> f64 {
        let mut sum = 0.0f64;
        for (pa, pb) in a.pixels().zip(b.pixels()) {
            for c in 0..3 {
                sum += (pa[c] as f64 - pb[c] as f64).abs();
            }
        }
        sum
    }
}
