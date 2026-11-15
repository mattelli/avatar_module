//! Wav2Lip + ExpNet fusion stage (Plan 5, task 6; reordered Plan 7, task 5).
//!
//! Per-frame building blocks for the reordered talking-head pipeline:
//!
//! - **Expression warp** ([`warp_from_landmarks`]): the 64 ExpNet expression
//!   coefficients are mapped to displacements on **9 control points** whose
//!   *positions* come from **PFLD-detected** landmarks (Plan 7, tasks 2–3).
//!   The warp is `atlas::warp_cpu` (inverse-distance, bilinear).
//! - **Pose warp** ([`pose_warp`], Plan 6, task 6): the Audio2Pose head-pose
//!   coefficients (yaw/pitch/roll + tx/ty/tz) are mapped to a 2D affine warp
//!   (rotate + slight scale/shear/translate).
//! - **Patch conversion** ([`patch_to_image`]): the Wav2Lip `[3,96,96]`
//!   channel-major output → RGBA, for the step-4 in-paint.
//!
//! The per-frame orchestration (crop → Wav2Lip → expression warp → in-paint →
//! pose warp → resize) lives in [`crate::talking_head::render_talking_head`];
//! the in-paint itself is in [`crate::inpaint`]. The old fixed-ellipse mouth
//! blend (`RegionBlend` / `blend_mouth` / `fuse_frame`) was removed in Plan 7,
//! task 6 — it is superseded by the whole-face in-paint, which has no seam.
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

/// Number of expression-warp control points (matches
/// [`crate::pfld::NUM_CONTROL_POINTS`]).
pub const NUM_CONTROL_POINTS: usize = 9;

/// Configuration for the fusion stage.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Output frame size (square). Defaults to 256 (user decision).
    pub size: u32,
    /// Overall warp strength for the expression deformation.
    pub strength: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            size: 256,
            strength: 0.5,
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

/// Expression warp driven by **detected** control-point positions.
///
/// `landmarks9` are the 9 control points in the order used by
/// [`crate::pfld::landmarks_to_control_points`] — `[mouth center, mouth left,
/// mouth right, brow left, brow right, eye left, eye right, nose, chin]` —
/// normalized to `0..1` on the clean face. The landmark **positions** place the
/// control points on the actual face; the **displacements** still come from the
/// (hand-tuned) ExpNet coefficient mapping in [`warp_from_positions`].
///
/// This replaced the old canonical-position `warp_from_coeffs`: the 9 points
/// come from PFLD, so the warp follows the real face geometry.
pub fn warp_from_landmarks(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    exp_coeffs: &[f32],
    landmarks9: &[[f32; 2]],
    strength: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    debug_assert_eq!(
        landmarks9.len(),
        NUM_CONTROL_POINTS,
        "expected {NUM_CONTROL_POINTS} control points, got {}",
        landmarks9.len()
    );
    warp_from_positions(src, exp_coeffs, landmarks9, strength)
}

/// Shared expression-warp core: build the 9 control points from the given
/// **positions** (PFLD-detected) and the ExpNet coefficient **displacements**,
/// then warp `src`.
///
/// The displacement mapping — the hand-tuned ExpNet → motion projection — is
/// the same for every caller; only the positions vary.
fn warp_from_positions(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    exp_coeffs: &[f32],
    positions: &[[f32; 2]],
    strength: f32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut pts: Vec<ControlPoint> = positions
        .iter()
        .map(|&pos| ControlPoint { pos, disp: [0.0, 0.0] })
        .collect();

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

/// Converts a flat `[3, 96, 96]` (channel-major, CHW) Wav2Lip face patch with
/// values in 0..1 into an RGBA image.
///
/// The Wav2Lip model returns channel-first tensors (`[b, 3, 96, 96]`), so the
/// buffer is laid out as three consecutive 96×96 planes (R, G, B) — **not**
/// interleaved RGB. Indexing it as HWC interleaves the planes and produces a
/// tiled/garbled face.
pub fn patch_to_image(patch: &[f32]) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::test_face_cpu;

    /// The 9 detected control points (a plausible face layout).
    fn sample_landmarks9() -> [[f32; 2]; NUM_CONTROL_POINTS] {
        [
            [0.50, 0.72], // mouth center
            [0.42, 0.70], // mouth left
            [0.58, 0.70], // mouth right
            [0.35, 0.45], // brow left
            [0.65, 0.45], // brow right
            [0.35, 0.52], // eye left
            [0.65, 0.52], // eye right
            [0.50, 0.60], // nose
            [0.50, 0.88], // chin
        ]
    }

    /// T3: a nonzero coefficient changes the frame when the warp is driven by
    /// detected landmarks (the landmark-driven path is live, not a no-op).
    #[test]
    fn landmark_warp_nonzero_coeff_changes_frame() {
        let src = test_face_cpu(128, 128);
        let lm = sample_landmarks9();

        let neutral = [0.0f32; 64];
        let mut coeffs = [0.0f32; 64];
        coeffs[0] = 1.0; // mouth open

        let a = warp_from_landmarks(&src, &neutral, &lm, 0.5);
        let b = warp_from_landmarks(&src, &coeffs, &lm, 0.5);
        assert!(
            pixel_diff(&a, &b) > 0.0,
            "nonzero coefficient should change the landmark-driven frame"
        );
    }

    /// T3: zero coefficients are the identity (no displacement => no change).
    #[test]
    fn landmark_warp_neutral_is_identity() {
        let src = test_face_cpu(128, 128);
        let lm = sample_landmarks9();
        let neutral = [0.0f32; 64];
        let out = warp_from_landmarks(&src, &neutral, &lm, 0.5);
        assert_eq!(out.dimensions(), src.dimensions());
        assert_eq!(
            pixel_diff(&src, &out),
            0.0,
            "zero coefficients should leave the frame unchanged"
        );
    }

    /// T3: the warp is bounded — extreme coefficients and extreme landmark
    /// positions still return a same-sized frame without panicking.
    #[test]
    fn landmark_warp_is_bounded() {
        let src = test_face_cpu(64, 64);
        // Landmarks pushed to the corners (worst case).
        let lm = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.0],
        ];
        let extreme = [1.0f32; 64];
        let out = warp_from_landmarks(&src, &extreme, &lm, 1.0);
        assert_eq!(out.dimensions(), src.dimensions());
    }

    /// T3: the *positions* are actually used — the same coefficients with two
    /// different landmark sets produce different frames (this is the whole
    /// point of replacing the canonical points with detected ones).
    #[test]
    fn landmark_positions_drive_the_warp() {
        let src = test_face_cpu(128, 128);
        let mut coeffs = [0.0f32; 64];
        coeffs[0] = 1.0; // mouth open

        let lm_a = sample_landmarks9();
        // Shift the mouth control points to a different place.
        let mut lm_b = sample_landmarks9();
        lm_b[0] = [0.30, 0.60];
        lm_b[1] = [0.22, 0.58];
        lm_b[2] = [0.38, 0.58];

        let a = warp_from_landmarks(&src, &coeffs, &lm_a, 0.5);
        let b = warp_from_landmarks(&src, &coeffs, &lm_b, 0.5);
        assert!(
            pixel_diff(&a, &b) > 0.0,
            "different landmark positions should produce different warps"
        );
    }

    /// A nonzero brow coefficient (index 3) moves the brows (expression), via
    /// the landmark-driven warp.
    #[test]
    fn brow_coeff_moves_brows() {
        let src = test_face_cpu(128, 128);
        let lm = sample_landmarks9();

        let c0 = [0.0f32; 64];
        let mut c1 = [0.0f32; 64];
        c1[3] = 1.0; // brow raise
        let a = warp_from_landmarks(&src, &c0, &lm, 0.5);
        let b = warp_from_landmarks(&src, &c1, &lm, 0.5);
        assert!(pixel_diff(&a, &b) > 0.0);
    }

    /// T6: a nonzero pose changes the frame vs. no pose.
    #[test]
    fn pose_changes_frame() {
        let src = test_face_cpu(128, 128);

        // A roll (rotation) pose should visibly change the frame.
        let pose = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let posed = pose_warp(&src, &pose, 0.5);
        assert!(
            pixel_diff(&src, &posed) > 0.0,
            "nonzero pose should change the frame"
        );

        // A translation (tx) pose should also change the frame.
        let pose_tx = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let posed_tx = pose_warp(&src, &pose_tx, 0.5);
        assert!(
            pixel_diff(&src, &posed_tx) > 0.0,
            "nonzero tx pose should change the frame"
        );
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
    /// patch with a bright R plane and dark G/B planes must convert to a *red*
    /// RGBA image — not a garbled/tiled or gray one. Indexing the buffer as
    /// interleaved HWC (the old bug) interleaves the planes and breaks this.
    #[test]
    fn patch_channels_are_planar_not_interleaved() {
        // CHW: plane 0 (R) = 1.0, planes 1 (G) and 2 (B) = 0.0.
        const PLANE: usize = 96 * 96;
        let mut patch = vec![0.0f32; 3 * PLANE];
        patch[..PLANE].fill(1.0);

        let img = patch_to_image(&patch);
        assert_eq!(img.dimensions(), (96, 96));

        let p = img.get_pixel(48, 48);
        assert!(
            p[0] > 200 && p[1] < 60 && p[2] < 60,
            "R plane should map to the red channel (got {:?}); \
             interleaved indexing would mix the planes",
            p
        );
    }

    /// `patch_to_image` clamps out-of-range values and keeps alpha opaque.
    #[test]
    fn patch_to_image_clamps_and_is_opaque() {
        let patch = vec![2.0f32; 96 * 96 * 3]; // above 1.0
        let img = patch_to_image(&patch);
        let p = img.get_pixel(10, 10);
        assert_eq!([p[0], p[1], p[2], p[3]], [255, 255, 255, 255]);
    }
}
