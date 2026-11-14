//! Wav2Lip + ExpNet fusion stage (Plan 5, task 6).
//!
//! Combines, per video frame, the two talking-head signals:
//!
//! - **ExpNet** supplies 3DMM-style expression coefficients (64 per frame)
//!   that drive the *non-lip* facial motion (brows, eyes, mouth corners,
//!   cheeks) — the audio-driven expression deformation.
//! - **Wav2Lip** supplies a re-rendered 96×96 face whose mouth matches the
//!   speech — the *lip articulation*. (Wav2Lip outputs a full face; only its
//!   mouth sub-rectangle is used.)
//!
//! The fusion is:
//!
//! 1. **Expression warp**: the coefficients are mapped to displacements on
//!    the existing atlas control points ([`crate::atlas`]), then the source
//!    face is warped with `atlas::warp_cpu`. The residual / low-frequency
//!    coefficients drive the coarse expression; we use a small projection to
//!    the 7-parameter mouth/brow/eye/chin motion.
//! 2. **Pose warp** (Plan 6, task 6): the Audio2Pose head-pose coefficients
//!    (yaw/pitch/roll + tx/ty/tz) are mapped to a 2D affine warp (rotate +
//!    slight scale/translate) applied to the expression-warped frame.
//! 3. **Mouth blend**: the mouth sub-rectangle of the Wav2Lip face is
//!    alpha-blended into the lower-face region of the warped frame through a
//!    smooth elliptical mask ([`RegionBlend`]), so the precise lips replace
//!    the atlas's deterministic mouth while the source keeps its expression
//!    and head pose.
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

/// A source→destination region blend: take a sub-rectangle of a source patch
/// and alpha-blend it (through a feathered ellipse) into a rectangle of the
/// frame. Both rectangles are normalized (0..1) so they scale with the frame.
///
/// This is the general primitive behind [`blend_mouth`]; other facial regions
/// (e.g. eyes for blinking) can be added by defining further regions once a
/// model produces imagery for them.
#[derive(Debug, Clone, Copy)]
pub struct RegionBlend {
    /// Source sub-rectangle `[x0, y0, x1, y1]` (normalized) of the patch.
    pub src: [f32; 4],
    /// Destination rectangle `[x0, y0, x1, y1]` (normalized) of the frame.
    pub dst: [f32; 4],
    /// Feather of the elliptical mask (0 = hard edge, 1 = fully soft).
    pub feather: f32,
}

impl RegionBlend {
    /// The mouth region: the lower-center of the Wav2Lip 96×96 face blended
    /// into the bottom-center of the frame.
    ///
    /// Wav2Lip outputs a **full face**, not a mouth-only patch, so we take the
    /// mouth sub-rectangle of that face (the lower-center third) and blend only
    /// that in — the rest of the source frame (with its expression/pose warp)
    /// is kept.
    pub fn mouth(cfg: &FusionConfig) -> Self {
        let w = cfg.mouth_mask_width.clamp(0.0, 1.0);
        let h = cfg.mouth_mask_height.clamp(0.0, 1.0);
        let x0 = (1.0 - w) / 2.0;
        Self {
            // Mouth sub-rectangle of the 96×96 Wav2Lip face (lower-center).
            src: [0.25, 0.60, 0.75, 1.0],
            // Bottom-center of the frame (centered horizontally, anchored down).
            dst: [x0, 1.0 - h, x0 + w, 1.0],
            feather: cfg.mask_feather,
        }
    }
}

/// Blends a sub-rectangle of `patch_img` into `frame` through a feathered
/// ellipse, per `region`. Returns a new frame.
pub fn blend_region(
    frame: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    patch_img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    region: &RegionBlend,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (fw, fh) = frame.dimensions();
    let (pw, ph) = patch_img.dimensions();

    // Source crop (clamped to the patch).
    let sx0 = (region.src[0].clamp(0.0, 1.0) * pw as f32) as u32;
    let sy0 = (region.src[1].clamp(0.0, 1.0) * ph as f32) as u32;
    let sx1 = ((region.src[2].clamp(0.0, 1.0) * pw as f32).ceil() as u32).min(pw);
    let sy1 = ((region.src[3].clamp(0.0, 1.0) * ph as f32).ceil() as u32).min(ph);
    let sx1 = sx1.max(sx0 + 1).min(pw);
    let sy1 = sy1.max(sy0 + 1).min(ph);
    let crop = image::imageops::crop_imm(patch_img, sx0, sy0, sx1 - sx0, sy1 - sy0).to_image();

    // Destination rectangle (clamped to the frame).
    let dx0 = (region.dst[0].clamp(0.0, 1.0) * fw as f32) as u32;
    let dy0 = (region.dst[1].clamp(0.0, 1.0) * fh as f32) as u32;
    let dx1 = ((region.dst[2].clamp(0.0, 1.0) * fw as f32).ceil() as u32).min(fw);
    let dy1 = ((region.dst[3].clamp(0.0, 1.0) * fh as f32).ceil() as u32).min(fh);
    let dw = dx1.saturating_sub(dx0).max(1);
    let dh = dy1.saturating_sub(dy0).max(1);

    let resized = image::imageops::resize(&crop, dw, dh, image::imageops::FilterType::Triangle);

    let mut out = frame.clone();
    let feather = region.feather.clamp(0.0, 1.0);
    let cx = dx0 as f32 + dw as f32 / 2.0;
    let cy = dy0 as f32 + dh as f32 / 2.0;
    let rx = dw as f32 / 2.0;
    let ry = dh as f32 / 2.0;

    for y in 0..dh {
        for x in 0..dw {
            let fx = dx0 + x;
            let fy = dy0 + y;
            if fx >= fw || fy >= fh {
                continue;
            }
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
            for c in 0..3 {
                let blend = p[c] as f32 * alpha + dst[c] as f32 * (1.0 - alpha);
                dst[c] = blend.clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Blends the Wav2Lip mouth patch into the lower-face region of `frame`.
///
/// The patch (`[3, 96, 96]` channel-major, values 0..1) is a **full face**; its
/// mouth sub-rectangle is cropped, resized to the mouth mask region of the
/// frame, and alpha-blended through a smooth elliptical mask. Returns a new
/// frame.
pub fn blend_mouth(
    frame: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    mouth_patch: &[f32],
    cfg: &FusionConfig,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let patch_img = patch_to_image(mouth_patch); // 96x96 Rgba (full face)
    blend_region(frame, &patch_img, &RegionBlend::mouth(cfg))
}

/// Converts a flat `[3, 96, 96]` (channel-major, CHW) mouth patch with values
/// in 0..1 into an RGBA image.
///
/// The Wav2Lip model returns channel-first tensors (`[b, 3, 96, 96]`), so the
/// buffer is laid out as three consecutive 96×96 planes (R, G, B) — **not**
/// interleaved RGB. Indexing it as HWC interleaves the planes and produces a
/// tiled/garbled face.
fn patch_to_image(patch: &[f32]) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    const PLANE: usize = 96 * 96;
    let mut img = ImageBuffer::from_pixel(96, 96, Rgba([0u8, 0, 0, 255]));
    for y in 0..96usize {
        for x in 0..96usize {
            let base = y * 96 + x;
            img.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    (patch[base].clamp(0.0, 1.0) * 255.0) as u8,
                    (patch[PLANE + base].clamp(0.0, 1.0) * 255.0) as u8,
                    (patch[2 * PLANE + base].clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ]),
            );
        }
    }
    img
}

/// Applies a 2D affine head-pose warp to `src`.
///
/// The Audio2Pose coefficients are `[yaw, pitch, roll, tx, ty, tz]`:
///
/// - `yaw` (about the vertical axis) → horizontal shear
/// - `pitch` (about the horizontal axis) → vertical shear
/// - `roll` (about the view axis) → rotation
/// - `tx` / `ty` → 2D translation (offset)
/// - `tz` → scale (toward / away from the camera)
///
/// The transform is applied about the frame center so the head rotates
/// around its own centre rather than drifting. The output is the same size
/// as `src`; out-of-bounds samples clamp to the edge (bounded).
///
/// `pose` must be 6 values long; `strength` scales the whole effect.
pub fn pose_warp(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    pose: &[f32],
    strength: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = (w as f32 / 2.0, h as f32 / 2.0);

    // Clamp each pose component to keep the warp bounded.
    let yaw = pose.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let pitch = pose.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let roll = pose.get(2).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let tx = pose.get(3).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let ty = pose.get(4).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
    let tz = pose.get(5).copied().unwrap_or(0.0).clamp(-1.0, 1.0);

    // Rotation (roll) about the centre.
    let ang = roll * 0.15 * strength;
    let (sin_a, cos_a) = ang.sin_cos();
    // Shear from yaw/pitch (small, bounded).
    let shear_x = yaw * 0.20 * strength;
    let shear_y = pitch * 0.20 * strength;
    // Translation (offset) and scale (tz).
    let off_x = tx * 0.10 * strength * w as f32;
    let off_y = ty * 0.10 * strength * h as f32;
    let scale = 1.0 + tz * 0.06 * strength;

    let mut out = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            // Normalized coords about the centre.
            let (nx, ny) = (x as f32 - cw, y as f32 - ch);
            // Inverse transform: apply rotation, shear, scale, then translate.
            // We sample the source at the inverse-mapped position.
            let (rx, ry) = (nx * cos_a + ny * sin_a, -nx * sin_a + ny * cos_a);
            let (sx, sy) = (rx / scale, ry / scale);
            let (px, py) = (sx - shear_x * sy, sy - shear_y * sx);
            // Source pixel (clamped to edges -> bounded).
            let sx = (px + cw - off_x).clamp(0.0, w as f32 - 1.0);
            let sy = (py + ch - off_y).clamp(0.0, h as f32 - 1.0);
            let x0 = (sx.floor() as usize).min(w - 1);
            let y0 = (sy.floor() as usize).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let p00 = src.get_pixel(x0 as u32, y0 as u32);
            let p10 = src.get_pixel(x1 as u32, y0 as u32);
            let p01 = src.get_pixel(x0 as u32, y1 as u32);
            let p11 = src.get_pixel(x1 as u32, y1 as u32);
            let mut px = [0u8; 4];
            for c in 0..4 {
                let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                px[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x as u32, y as u32, Rgba(px));
        }
    }
    out
}

/// Full per-frame fusion: expression-warp the source face, apply the
/// optional head-pose warp, then blend the Wav2Lip mouth patch in.
///
/// - `src`: the avatar source face (already face-cropped).
/// - `exp_coeffs`: 64 ExpNet expression coefficients for this frame.
/// - `pose`: optional 6 Audio2Pose head-pose coefficients (yaw/pitch/roll +
///   tx/ty/tz); `None` applies no pose warp.
/// - `mouth_patch`: the `[3, 96, 96]` (channel-major) Wav2Lip mouth patch
///   (values 0..1).
///
/// The frame is resized to `cfg.size` at the end.
pub fn fuse_frame(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    exp_coeffs: &[f32],
    pose: Option<&[f32]>,
    mouth_patch: &[f32],
    cfg: &FusionConfig,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut warped = warp_from_coeffs(src, exp_coeffs, cfg.strength);
    if let Some(p) = pose {
        warped = pose_warp(&warped, p, cfg.strength);
    }
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
        let base = fuse_frame(&src, &neutral, None, &dark_patch, &cfg);

        // 1. ExpNet signal: a mouth-open coefficient (index 0 = 1.0) warps the
        //    mouth, so the fused frame differs from neutral.
        let mut coeffs = [0.0f32; 64];
        coeffs[0] = 1.0;
        let exp_warped = fuse_frame(&src, &coeffs, None, &dark_patch, &cfg);
        let exp_diff = pixel_diff(&base, &exp_warped);
        assert!(exp_diff > 0.0, "ExpNet signal should change the frame");

        // 2. Wav2Lip signal: a bright mouth patch changes the lower-face.
        let bright_patch = vec![1.0f32; 96 * 96 * 3];
        let lip_blended = fuse_frame(&src, &neutral, None, &bright_patch, &cfg);
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
        let a = fuse_frame(&src, &c0, None, &dark, &cfg);
        let b = fuse_frame(&src, &c1, None, &dark, &cfg);
        assert!(pixel_diff(&a, &b) > 0.0);
    }

    /// Neutral coeffs + a black mouth patch darken the mouth mask region.
    #[test]
    fn black_patch_darkens_mouth_region() {
        let src = ImageBuffer::from_pixel(128, 128, Rgba([255u8, 255, 255, 255]));
        let cfg = FusionConfig::default().with_size(128);
        let neutral = [0.0f32; 64];
        let black = vec![0.0f32; 96 * 96 * 3];

        let out = fuse_frame(&src, &neutral, None, &black, &cfg);
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

    /// T6: a nonzero pose changes the fused frame vs. no pose.
    #[test]
    fn pose_changes_frame() {
        let src = test_face_cpu(128, 128);
        let cfg = FusionConfig::default().with_size(128);
        let neutral = [0.0f32; 64];
        let dark = vec![0.0f32; 96 * 96 * 3];

        let base = fuse_frame(&src, &neutral, None, &dark, &cfg);

        // A roll (rotation) pose should visibly change the frame.
        let pose = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let posed = fuse_frame(&src, &neutral, Some(&pose), &dark, &cfg);
        let diff = pixel_diff(&base, &posed);
        assert!(diff > 0.0, "nonzero pose should change the frame");

        // A translation (tx) pose should also change the frame.
        let pose_tx = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let posed_tx = fuse_frame(&src, &neutral, Some(&pose_tx), &dark, &cfg);
        let diff_tx = pixel_diff(&base, &posed_tx);
        assert!(diff_tx > 0.0, "nonzero tx pose should change the frame");
    }

    /// T6: the pose warp is bounded — a large pose does not produce NaN /
    /// garbage and keeps the same dimensions.
    #[test]
    fn pose_warp_is_bounded() {
        let src = test_face_cpu(64, 64);
        // Extreme pose values (clamped internally).
        let pose = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = pose_warp(&src, &pose, 1.0);
        assert_eq!(out.dimensions(), src.dimensions());

        // The warp must not panic / produce garbage: same size, all pixels valid.
        assert_eq!(out.dimensions(), src.dimensions());

        // A zero pose is the identity (no change).
        let zero = [0.0f32; 6];
        let identity = pose_warp(&src, &zero, 1.0);
        assert_eq!(pixel_diff(&src, &identity), 0.0, "zero pose should be identity");
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

    /// Regression: the Wav2Lip patch is channel-major (`[3, 96, 96]`), so a
    /// patch with a bright R plane and dark G/B planes must render a *red*
    /// mouth region — not a garbled/tiled or gray one. Indexing the buffer as
    /// interleaved HWC (the old bug) interleaves the planes and breaks this.
    #[test]
    fn patch_channels_are_planar_not_interleaved() {
        let src = ImageBuffer::from_pixel(128, 128, Rgba([0u8, 0, 0, 255]));
        let cfg = FusionConfig::default().with_size(128);
        let neutral = [0.0f32; 64];

        // CHW: plane 0 (R) = 1.0, planes 1 (G) and 2 (B) = 0.0.
        const PLANE: usize = 96 * 96;
        let mut patch = vec![0.0f32; 3 * PLANE];
        patch[..PLANE].fill(1.0);

        let out = fuse_frame(&src, &neutral, None, &patch, &cfg);

        // Sample the center of the mouth mask (bottom-center of the frame).
        let (fw, fh) = out.dimensions();
        let p = out.get_pixel(fw / 2, (fh as f32 * 0.85) as u32);
        assert!(
            p[0] > 200 && p[1] < 60 && p[2] < 60,
            "R plane should map to the red channel (got {:?}); \
             interleaved indexing would mix the planes",
            p
        );
    }

    /// Option B: only the mouth sub-rectangle of the Wav2Lip face is blended.
    /// A patch whose *upper* region (face/eyes) is bright red but whose *mouth*
    /// region is dark must NOT paint the frame red outside the mouth mask —
    /// i.e. the whole face is not squashed into the mouth region.
    #[test]
    fn only_mouth_subrect_of_patch_is_blended() {
        // White source frame so any red bleed is obvious.
        let src = ImageBuffer::from_pixel(128, 128, Rgba([255u8, 255, 255, 255]));
        let cfg = FusionConfig::default().with_size(128);
        let neutral = [0.0f32; 64];

        const PLANE: usize = 96 * 96;
        let mut patch = vec![0.0f32; 3 * PLANE];
        // Upper 60% of the face (rows 0..57) = bright red in the R plane.
        for y in 0..57usize {
            for x in 0..96usize {
                patch[y * 96 + x] = 1.0;
            }
        }

        let out = fuse_frame(&src, &neutral, None, &patch, &cfg);
        let (fw, fh) = out.dimensions();

        // The mouth region (bottom-center) takes the patch's *mouth* sub-rect,
        // which is dark here -> the frame goes dark there (not red).
        let mouth = out.get_pixel(fw / 2, (fh as f32 * 0.9) as u32);
        assert!(
            mouth[0] < 60,
            "mouth region should take the patch's dark mouth sub-rect, got {:?}",
            mouth
        );

        // The upper face (eyes/brows) must be untouched white — the patch's
        // upper red region must not leak in.
        let eyes = out.get_pixel(fw / 2, (fh as f32 * 0.4) as u32);
        assert!(
            eyes[0] > 200 && eyes[1] > 200 && eyes[2] > 200,
            "upper face should be untouched, got {:?}",
            eyes
        );
    }
}
