//! Image warping helpers used by the talking-head fusion stage (Plan 5/6).
//!
//! The classic expression-atlas path (7-band GPU atlas, wgpu compute shader)
//! is removed. What remains is shared image tooling used by
//! [`crate::fusion`] and [`crate::talking_head`]: the inverse-distance
//! displacement warp (`warp_cpu`/`ControlPoint`/`WarpParams`), image
//! load/save, square center-crop, and the synthetic test face.

use image::{ImageBuffer, Rgba};
use std::path::Path;

/// A control point for the inverse-distance warp: a normalized position and
/// a normalized displacement.
#[derive(Debug, Clone)]
pub struct ControlPoint {
    /// Position in normalized coordinates [0, 1].
    pub pos: [f32; 2],
    /// Displacement in normalized coordinates (added to the position).
    pub disp: [f32; 2],
}

/// Warp parameters for one warp (used by the expression deformation in
/// [`crate::fusion`]).
#[derive(Debug, Clone)]
pub struct WarpParams {
    /// Global warp strength (photos ~0.5, cartoons ~1.0).
    pub strength: f32,
    /// Control points for this warp.
    pub points: Vec<ControlPoint>,
}

/// CPU inverse-distance warp: `src` is RGBA8; returns the warped image of
/// the same size.
///
/// For each output pixel, the displacement is an inverse-distance-weighted
/// sum of the control-point displacements; the source position is then
/// displaced by `-strength * disp` and bilinearly sampled.
pub fn warp_cpu(
    src: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    params: &WarpParams,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut out = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let pn = [x as f32 / w as f32, y as f32 / h as f32];
            let mut disp = [0.0f32, 0.0];
            let mut wsum = 0.0f32;
            for cp in &params.points {
                let d = (pn[0] - cp.pos[0]).hypot(pn[1] - cp.pos[1]);
                let weight = 1.0 / (d * d + 0.004);
                disp[0] += weight * cp.disp[0];
                disp[1] += weight * cp.disp[1];
                wsum += weight;
            }
            disp[0] /= wsum;
            disp[1] /= wsum;
            let sp = [
                (pn[0] - disp[0] * params.strength).clamp(0.0, 1.0),
                (pn[1] - disp[1] * params.strength).clamp(0.0, 1.0),
            ];
            // Bilinear sample (uv * dims convention).
            let sx = sp[0] * w as f32;
            let sy = sp[1] * h as f32;
            let x0 = (sx.floor() as usize).min(w - 1);
            let y0 = (sy.floor() as usize).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let px = |xx: usize, yy: usize| -> [f32; 4] {
                let p = src.get_pixel(xx as u32, yy as u32);
                [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
            };
            let c00 = px(x0, y0);
            let c10 = px(x1, y0);
            let c01 = px(x0, y1);
            let c11 = px(x1, y1);
            let mut c = [0.0f32; 4];
            for k in 0..4 {
                let c0 = c00[k] * (1.0 - fx) + c10[k] * fx;
                let c1 = c01[k] * (1.0 - fx) + c11[k] * fx;
                c[k] = c0 * (1.0 - fy) + c1 * fy;
            }
            out.put_pixel(
                x as u32,
                y as u32,
                Rgba([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8]),
            );
        }
    }
    out
}

/// Builds a synthetic "face" image (gradient background) for tests and demos.
pub fn test_face_cpu(w: u32, h: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w.max(1)) as u8;
            let g = (y * 255 / h.max(1)) as u8;
            img.put_pixel(x, y, Rgba([r, g, 128, 255]));
        }
    }
    img
}

/// Loads an image from a file (PNG/JPEG).
pub fn load_image(path: &Path) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let img = image::open(path)?;
    Ok(img.to_rgba8())
}

/// Saves an image to a file (format inferred from extension).
pub fn save_image(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, path: &Path) -> anyhow::Result<()> {
    img.save(path)?;
    Ok(())
}

/// Center-crops `src` to a square (side = min(w, h)).
///
/// The crop is centered on the longer axis, so a face already centered by
/// face-crop stays centered. Non-square images (e.g. wide photos or a
/// non-square face crop) are cropped to a square working frame.
pub fn center_crop_square(src: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (w, h) = src.dimensions();
    let side = w.min(h);
    if side == 0 {
        return src.clone();
    }
    let x = (w - side) / 2;
    let y = (h - side) / 2;
    image::imageops::crop_imm(src, x, y, side, side).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// warp_cpu returns the same-sized image and moves a control point's
    /// region.
    #[test]
    fn warp_cpu_moves_points() {
        let src = test_face_cpu(64, 64);
        let params = WarpParams {
            strength: 0.5,
            points: vec![ControlPoint {
                pos: [0.5, 0.5],
                disp: [0.1, 0.0],
            }],
        };
        let out = warp_cpu(&src, &params);
        assert_eq!(out.dimensions(), src.dimensions());
        // The center column should differ from the source (a displacement was
        // applied).
        let src_center = src.get_pixel(32, 32);
        let out_center = out.get_pixel(32, 32);
        assert!(src_center != out_center);
    }

    /// Center-crop makes a non-square image square and preserves the center.
    #[test]
    fn center_crop_square_makes_square() {
        let wide = test_face_cpu(400, 200);
        let sq = center_crop_square(&wide);
        assert_eq!(sq.dimensions(), (200, 200));

        let tall = test_face_cpu(200, 400);
        let sq2 = center_crop_square(&tall);
        assert_eq!(sq2.dimensions(), (200, 200));

        let sq3 = test_face_cpu(64, 64);
        assert_eq!(center_crop_square(&sq3).dimensions(), (64, 64));

        let src = test_face_cpu(400, 200);
        let center_src = src.get_pixel(200, 100);
        let sq4 = center_crop_square(&src);
        assert_eq!(sq4.get_pixel(100, 100), center_src);
    }

    /// Image round-trip: save then load preserves dimensions.
    #[test]
    fn image_roundtrip() {
        let tmp = std::env::temp_dir().join("avatar_atlas_roundtrip.png");
        let img = test_face_cpu(32, 32);
        save_image(&img, &tmp).unwrap();
        let loaded = load_image(&tmp).unwrap();
        assert_eq!(loaded.dimensions(), (32, 32));
        let _ = std::fs::remove_file(&tmp);
    }
}
