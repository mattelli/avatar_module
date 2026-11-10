# Plan 4 — Face-detection + center-crop before atlas warping

**References:** `PLAN_AVATAR_MODULE.md` task 6 (expression atlas), task 9 (pipeline glue);
`PLAN_FIX_RENDER_ISSUES.md` task 4 (input downscale). Companion: `AVATAR_MODULE_REPORT.md`
§3.6 (atlas/render).

**Context:** The pipeline currently treats the entire input picture as the face. The expression
atlas (`atlas.rs`) warps the full image using 9 hardcoded normalized control points (mouth
corners, brows, eye corners, nose, chin) that assume a centered, frame-filling face, and the
renderer (`renderer.wgsl`) applies lip-sync at hardcoded normalized coordinates
(`MOUTH_Y = 0.72`, `MOUTH_HALF_W = 0.12`, `JAW = 0.16`). When the face is off-center or small
in the photo (e.g. `.res/avatar.jpg`), the output shows an arbitrary corner of the image and
the warp/mouth animation lands on the wrong pixels.

This plan adds an automatic **face-detection + center-crop** step that runs after loading the
picture and before atlas warping, so the face fills the working frame and the hardcoded control
points align with real facial features.

**Decisions applied (user-confirmed):**
- **Q1 = rustface** — pure-Rust SeetaFace-based face detector (BSD-2-Clause), no native deps,
  fits the "entirely in Rust" project philosophy.
- **Q2 = fallback to whole image** when no face is detected (never fail rendering).
- **Q3 = crop with a margin** (default ~35%) so forehead/hair/chin are not cut off.
- **Q4 = configurable** via `EngineConfig` (`face_crop: bool` default `true`,
  `face_crop_margin: f32` default `0.35`).
- **Q5 = bundle the model** in `.res/` and ship it in the wheel (like espeak-ng data), with a
  graceful fallback to no-crop if the model file is missing.

## Goals

- **G1** — Add a `face_crop` module to `avatar-core` that detects a face bounding box in an
  RGBA image using `rustface` and returns a margin-expanded, clamped crop rectangle.
- **G2** — Apply the crop in the pipeline: after `load_image`, before `downscale`/`atlas_cpu`,
  so the face is centered and fills the working frame.
- **G3** — Expose `face_crop: bool` (default `true`) and `face_crop_margin: f32` (default
  `0.35`) on `EngineConfig`, forwarded through the Python `Engine` constructor.
- **G4** — Bundle the SeetaFace model (`seeta_fd_frontal_v1.0.bin`) in `.res/` and ship it in
  the wheel; fall back to whole-image if the model is missing or detection fails.
- **G5** — Keep all existing tests green and add unit tests for the crop logic.

## Current behavior

- `EngineConfig` has no face-crop options; the picture is loaded at full resolution, downscaled
  to ≤128 px, and warped as-is.
- `.res/avatar.jpg` (740×740) has the face off-center, so the 64×64 output shows a corner of the
  photo and the warp control points don't match the face.

## Expected behavior

- With a picture containing a detectable face, the pipeline crops to the face bounding box
  expanded by `face_crop_margin` (default 0.35) on each side, clamps to the image bounds, then
  downscales and warps as before. The output avatar is centered on the face.
- If `face_crop` is `false`, or no face is detected, or the model file is missing, the picture is
  used whole (current behavior) — rendering never fails.
- `EngineConfig` and the Python `Engine(...)` accept `face_crop` and `face_crop_margin`.
- The wheel includes the face-detection model so the Python example works out of the box.
- All unit tests pass (`cargo test` and `cargo test --features aac`).

## Tasks (one commit per task; simple tasks)

> **Commit policy:** one commit per completed task, with a brief but detailed commit message
> (what changed, why, files touched, observable effect). Commit immediately once the task's work
> and tests are complete.

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | Add `rustface` dependency to `avatar-core` and obtain/bundle the SeetaFace model `seeta_fd_frontal_v1.0.bin` into `.res/`; add it to the wheel `include` in `pyproject.toml`; document the model source in README | `task1: add rustface dep + bundle model` | G1, G4 |
| 2 | Implement `face_crop` module: `detect_face_box(&ImageBuffer<Rgba<u8>>, model_path) -> Option<Rectangle>` (convert to grayscale, run `rustface::create_detector`, pick highest-score face) and `expand_margin(box, img_w, img_h, margin) -> (x, y, w, h)` clamped to bounds; unit tests with a synthetic "face-like" image and an empty image | `task2: face crop module` | G1 |
| 3 | Wire into pipeline: after `load_image`, if `cfg.face_crop`, detect + crop (margin-expanded, clamped), fall back to whole image on failure; then downscale + atlas as before; update `EngineConfig` with `face_crop`/`face_crop_margin` fields and builder methods; forward through Python `Engine` constructor (`face_crop=True`, `face_crop_margin=0.35`) | `task3: wire face crop into pipeline` | G2, G3 |
| 4 | Tests + docs: unit test that a centered synthetic face is cropped to a region containing the face and smaller than the input; test that margin expansion clamps to bounds; test that `face_crop=false` returns the whole image; update README (face-crop behavior, model bundling) and `examples/render_clip.py` if needed | `task4: face crop tests + docs` | G4, G5 |

## Unit-test verification

- **T1** — dependency/model: `cargo build` succeeds with `rustface`; `.res/seeta_fd_frontal_v1.0.bin`
  exists and is non-empty; wheel build (`maturin build`) includes the model.
- **T2** — detection: a synthetic grayscale image with a high-contrast "face" blob returns
  `Some(bbox)` with positive width/height within image bounds; an empty/blank image returns
  `None`; `expand_margin` clamps to image bounds and grows the box by the requested fraction.
- **T3** — pipeline: with `face_crop=true` and a detectable face, the atlas source is smaller
  than the input and centered on the face; with `face_crop=false` or a missing model, the whole
  image is used; rendering still produces a valid MP4.
- **T4** — end-to-end: `render("Hello <happy> world", out)` with a face picture produces an MP4
  whose frames show the face centered (spot-check via atlas source dimensions); all existing
  tests pass.

## Cross-references

- Implements improvement over: `PLAN_AVATAR_MODULE.md` task 6 (atlas warping assumes centered
  face), task 9 (pipeline glue).
- Builds on: `PLAN_FIX_RENDER_ISSUES.md` task 4 (input downscale) — the crop runs before
  downscaling.
- Report: `AVATAR_MODULE_REPORT.md` §3.6 (atlas/render).