//! Tight bbox crop + in-paint (Plan 7, task 4 — pipeline steps 1 & 3).
//!
//! The reordered pipeline runs Wav2Lip on a **tightly cropped clean face**
//! (step 1) and pastes its **whole** output back into the N×N square (step 3),
//! instead of blending a sub-region through a fixed ellipse. Pasting the whole
//! face back removes the mouth-patch seam (Plan 7 goal G1).
//!
//! - **Step 1 — [`crop_face_padded`]:** detect the face bbox on the N×N square,
//!   crop it with the Wav2Lip pads (`pads = [0, 10, 0, 0]`) to an M×M rect.
//!   The caller feeds this crop to `wav2lip::face_input` (which resizes it to
//!   96×96) and runs Wav2Lip.
//! - **Step 3 — [`inpaint_regions`] / [`paste_face`]:** resize the 96×96
//!   Wav2Lip output back to M×M and paste it into the N×N square at the same
//!   bbox rect, with a feathered boundary (default **2 px**).
//!
//! The paste is written as a **table of regions** ([`InpaintRegion`]) rather
//! than a mouth-only special case, so a future region (e.g. an eye patch for
//! blinking) can be appended without reworking the pipeline — see the Plan 7
//! annex "Future improvements".

use image::{ImageBuffer, Rgba};

/// Wav2Lip face pads: `pads = [0, 10, 0, 0]` -> `(pady1, pady2, padx1, padx2)`.
///
/// Mirrors the private `wav2lip::PADS` so the crop here matches exactly what
/// Wav2Lip was trained on (extra 10 px below the chin).
pub const PADS: (i64, i64, i64, i64) = (0, 10, 0, 0);

/// Default feather (ramp width) of the in-paint boundary, in pixels.
pub const DEFAULT_FEATHER_PX: u32 = 10;

/// A face crop rectangle in the pixel coordinates of the N×N square.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceRect {
    /// Left edge (px).
    pub x: u32,
    /// Top edge (px).
    pub y: u32,
    /// Width (px), at least 1.
    pub w: u32,
    /// Height (px), at least 1.
    pub h: u32,
}

impl FaceRect {
    /// Right edge (exclusive).
    pub fn x1(&self) -> u32 {
        self.x + self.w
    }
    /// Bottom edge (exclusive).
    pub fn y1(&self) -> u32 {
        self.y + self.h
    }
}

/// Step 1: crops the N×N `square` to the detected `bbox` expanded by the
/// Wav2Lip pads, clamped to the image bounds.
///
/// Returns the M×M crop (for `wav2lip::face_input`, which resizes it to 96×96)
/// and the pixel rect it came from (for the step-3 paste). The rect is the
/// exact inverse of the paste, so the Wav2Lip face lands 1:1 on its source
/// region — no shift, no sub-region mask.
pub fn crop_face_padded(
    square: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    bbox: &rustface::Rectangle,
) -> (ImageBuffer<Rgba<u8>, Vec<u8>>, FaceRect) {
    let (iw, ih) = square.dimensions();
    let (pady1, pady2, padx1, padx2) = PADS;

    let x = (bbox.x() as i64 - padx1).max(0) as u32;
    let y = (bbox.y() as i64 - pady1).max(0) as u32;
    let x2 = (bbox.x() as i64 + bbox.width() as i64 + padx2).min(iw as i64) as u32;
    let y2 = (bbox.y() as i64 + bbox.height() as i64 + pady2).min(ih as i64) as u32;
    let w = x2.saturating_sub(x).max(1).min(iw.saturating_sub(x).max(1));
    let h = y2.saturating_sub(y).max(1).min(ih.saturating_sub(y).max(1));

    let crop = image::imageops::crop_imm(square, x, y, w, h).to_image();
    (crop, FaceRect { x, y, w, h })
}

/// One in-paint region: the whole `patch` is resized to `dst` (a normalized
/// rect of the frame) and pasted with a feathered border.
///
/// A rectangular paste with a ramped border (the in-paint has no elliptical
/// mask). It is a "region table" so further regions (e.g. an eye patch for
/// blinking) can be added without reworking the pipeline.
#[derive(Debug, Clone, Copy)]
pub struct InpaintRegion {
    /// Destination rect in the frame `[x0, y0, x1, y1]`, normalized to 0..1.
    pub dst: [f32; 4],
    /// Feather ramp width in pixels (alpha ramps 0..1 over this many pixels
    /// inward from each edge). 0 = hard edge.
    pub feather_px: u32,
}

/// The in-paint table entry that pastes the Wav2Lip face back at `rect`.
///
/// `n` is the square's side (the frame is N×N); `rect` is the pixel rect from
/// [`crop_face_padded`]. Kept as a constructor (not inlined) so the pipeline
/// builds an explicit region table.
pub fn face_inpaint_region(rect: FaceRect, n: u32, feather_px: u32) -> InpaintRegion {
    let n = n.max(1) as f32;
    InpaintRegion {
        dst: [
            rect.x as f32 / n,
            rect.y as f32 / n,
            rect.x1() as f32 / n,
            rect.y1() as f32 / n,
        ],
        feather_px,
    }
}

/// Step 3: pastes `patch` into `frame` for each region in `regions`, with a
/// feathered border, and returns a new frame.
///
/// The patch is resized to each region's `dst` rect and alpha-blended: alpha is
/// 1 in the interior and ramps to 0 over `feather_px` pixels from each edge, so
/// there is no hard seam at the boundary.
pub fn inpaint_regions(
    frame: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    patch: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    regions: &[InpaintRegion],
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut out = frame.clone();
    let (fw, fh) = out.dimensions();

    for region in regions {
        let dx0 = (region.dst[0].clamp(0.0, 1.0) * fw as f32).round() as u32;
        let dy0 = (region.dst[1].clamp(0.0, 1.0) * fh as f32).round() as u32;
        let dx1 = ((region.dst[2].clamp(0.0, 1.0) * fw as f32).round() as u32).min(fw);
        let dy1 = ((region.dst[3].clamp(0.0, 1.0) * fh as f32).round() as u32).min(fh);
        let dx0 = dx0.min(dx1);
        let dy0 = dy0.min(dy1);
        let dw = dx1.saturating_sub(dx0).max(1);
        let dh = dy1.saturating_sub(dy0).max(1);

        // Resize the (96×96) patch to the destination rect (M×M).
        let resized = image::imageops::resize(
            patch,
            dw,
            dh,
            image::imageops::FilterType::Lanczos3,
        );

        let f = region.feather_px as f32;
        for y in 0..dh {
            for x in 0..dw {
                let fx = dx0 + x;
                let fy = dy0 + y;
                if fx >= fw || fy >= fh {
                    continue;
                }
                // Distance (px) to the nearest edge of the destination rect.
                let d = x.min(y).min(dw - 1 - x).min(dh - 1 - y) as f32;
                let alpha = if f <= 0.0 { 1.0 } else { (d / f).min(1.0) };
                if alpha <= 0.0 {
                    continue;
                }
                let p = resized.get_pixel(x, y);
                let dst = out.get_pixel_mut(fx, fy);
                for c in 0..3 {
                    let v = p[c] as f32 * alpha + dst[c] as f32 * (1.0 - alpha);
                    dst[c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// Convenience wrapper: paste a single Wav2Lip `face` back into `square` at
/// `rect` with `feather_px`. Equivalent to [`inpaint_regions`] with the single
/// [`face_inpaint_region`] entry.
pub fn paste_face(
    square: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    face: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    rect: FaceRect,
    feather_px: u32,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (n, _) = square.dimensions();
    let region = face_inpaint_region(rect, n, feather_px);
    inpaint_regions(square, face, &[region])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solid-colour RGBA image.
    fn solid(w: u32, h: u32, rgb: [u8; 3]) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        ImageBuffer::from_pixel(w, h, Rgba([rgb[0], rgb[1], rgb[2], 255]))
    }

    /// Step 1: the padded crop returns the bbox expanded by the Wav2Lip pads,
    /// and the rect matches the crop size.
    #[test]
    fn crop_face_padded_uses_wav2lip_pads() {
        let square = solid(100, 100, [10, 10, 10]);
        // bbox at (20, 20) size 40x40; pads add 10 px below (pady2).
        let bbox = rustface::Rectangle::new(20, 20, 40, 40);
        let (crop, rect) = crop_face_padded(&square, &bbox);

        assert_eq!(rect.x, 20);
        assert_eq!(rect.y, 20);
        assert_eq!(rect.w, 40);
        assert_eq!(rect.h, 50); // 40 + 10 px pad below the chin
        assert_eq!(crop.dimensions(), (rect.w, rect.h));
    }

    /// Step 1: the crop is clamped to the square's bounds (no out-of-range).
    #[test]
    fn crop_face_padded_clamps_to_bounds() {
        let square = solid(60, 60, [10, 10, 10]);
        // bbox touching the bottom edge; the +10 px pad must clamp.
        let bbox = rustface::Rectangle::new(0, 40, 60, 20);
        let (crop, rect) = crop_face_padded(&square, &bbox);
        assert_eq!(rect.y1(), 60);
        assert_eq!(crop.dimensions(), (rect.w, rect.h));
    }

    /// Step 3: the pasted patch lands at the bbox rect (the pixels inside the
    /// rect take the patch colour; the pixels outside keep the frame colour).
    #[test]
    fn paste_lands_at_bbox_rect() {
        let square = solid(100, 100, [0, 0, 0]);
        let face = solid(96, 96, [255, 0, 0]); // red face
        let rect = FaceRect { x: 20, y: 20, w: 40, h: 50 };

        let out = paste_face(&square, &face, rect, DEFAULT_FEATHER_PX);
        assert_eq!(out.dimensions(), (100, 100));

        // Interior of the rect is red.
        let inside = out.get_pixel(40, 45);
        assert_eq!([inside[0], inside[1], inside[2]], [255, 0, 0]);

        // Outside the rect stays black.
        let outside = out.get_pixel(5, 5);
        assert_eq!([outside[0], outside[1], outside[2]], [0, 0, 0]);
        let below = out.get_pixel(40, 90);
        assert_eq!([below[0], below[1], below[2]], [0, 0, 0]);
    }

    /// Step 3: the boundary is feathered — a pixel just inside the rect edge is
    /// a partial blend (neither the pure frame nor the pure patch colour), so
    /// there is no hard seam.
    #[test]
    fn paste_boundary_is_feathered() {
        let square = solid(100, 100, [0, 0, 0]);
        let face = solid(96, 96, [255, 255, 255]); // white face
        let rect = FaceRect { x: 20, y: 20, w: 40, h: 50 };

        let out = paste_face(&square, &face, rect, DEFAULT_FEATHER_PX);

        // The outermost row of the rect (d = 0) is fully the frame (black).
        let edge = out.get_pixel(40, 20);
        assert_eq!(edge[0], 0, "outermost edge should be the frame colour");

        // One pixel in (d = 1) -> alpha = 1 / feather -> a partial blend.
        let ramp = out.get_pixel(40, 21);
        assert!(
            ramp[0] > 0 && ramp[0] < 255,
            "feather pixel should be a partial blend, got {}",
            ramp[0]
        );

        // `feather` pixels in (d = feather) -> alpha 1 -> full patch colour.
        // Derived from the constant so the test does not pin the ramp width.
        let core = out.get_pixel(40, 20 + DEFAULT_FEATHER_PX);
        assert_eq!(core[0], 255, "interior should be the full patch colour");
    }

    /// Step 3: a zero feather gives a hard edge (the rect boundary is abrupt).
    #[test]
    fn zero_feather_is_hard_edge() {
        let square = solid(100, 100, [0, 0, 0]);
        let face = solid(96, 96, [255, 255, 255]);
        let rect = FaceRect { x: 20, y: 20, w: 40, h: 50 };

        let out = paste_face(&square, &face, rect, 0);
        // With no feather the outermost row is already the full patch colour.
        let edge = out.get_pixel(40, 20);
        assert_eq!(edge[0], 255);
    }

    /// Step 3: the region table supports more than one region (the table is
    /// generic, not mouth-only) — two disjoint rects both receive the patch.
    #[test]
    fn region_table_supports_multiple_regions() {
        let square = solid(100, 100, [0, 0, 0]);
        let face = solid(96, 96, [0, 255, 0]); // green
        let regions = [
            InpaintRegion { dst: [0.1, 0.1, 0.3, 0.3], feather_px: 0 },
            InpaintRegion { dst: [0.6, 0.6, 0.9, 0.9], feather_px: 0 },
        ];
        let out = inpaint_regions(&square, &face, &regions);

        assert_eq!(out.get_pixel(20, 20)[1], 255, "first region should be green");
        assert_eq!(out.get_pixel(75, 75)[1], 255, "second region should be green");
        assert_eq!(out.get_pixel(50, 50)[1], 0, "between regions stays black");
    }
}
