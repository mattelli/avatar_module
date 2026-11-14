# Plan 7 — Reordered talking-head pipeline (Wav2Lip → in-paint → PFLD expression → pose)

**Status: proposed (not implemented).**

**References:** `steps-analysis.md` (the step-by-step analysis this plan
implements), `plans/PLAN_TALKING_HEAD_ONLY.md` (Plan 6 — talking-head only,
sizes, Audio2Pose), `plans/PLAN_WAV2LIP_EXPNET.md` (Plan 5 — Wav2Lip + ExpNet).

---

## Goals

- **G1 — Fix the mouth-patch seam.** The current pipeline runs Wav2Lip on the
  *unwarped* source and blends its mouth into the *warped* frame through a
  **fixed** ellipse → a visible patch seam. Reorder so Wav2Lip runs on a clean,
  tightly cropped face and its **whole** output is pasted back (no sub-region
  mask).
- **G2 — Detect the expression control points.** Replace the 9 **hardcoded**
  canonical control points with points **detected by PFLD** on the clean face.
- **G3 — Add PFLD.** Convert `pfld.onnx` → `.bpk`, load it like Wav2Lip/ExpNet,
  and use its 98 landmarks to drive the expression warp.
- **G4 — New step order.** Implement the agreed sequence:

  ```
  src ─▶ [0] square N×N ─▶ [1] bbox crop M×M → 96×96 ─▶ [2] Wav2Lip(clean)
      ─▶ [3] 96×96 → M×M, in-paint into N×N ─▶ [4] N×N → 112×112, expression warp (PFLD)
      ─▶ [5] pose warp ─▶ [6] resize → frame
  ```

---

## Decisions applied (user-confirmed)

- **Landmark model:** PFLD. `pfld.onnx` → `.bpk` in `.res/models/pfld/`, loaded
  like Wav2Lip/ExpNet.
- **PFLD license:** not a concern.
- **Detection face:** landmarks detected on the **clean** (unwarped) face;
  **expression warp before pose warp**.
- **ExpNet → displacement mapping:** stays **hand-tuned** (landmarks fix
  *positions*, not *displacements*).
- **Wav2Lip framing:** tighter **detected-bbox** crop (`crop_face`), not the
  whole square.
- **Per-frame Wav2Lip:** one clean face per **chunk**, reused across frames;
  batch chunks (default 8).
- **In-paint boundary ramp:** **2 px**.
- **N:** the input image's square size (e.g. **N = 740** for `avatar.jpg`).
- **Steps 4–5 resolution:** **112×112** (PFLD's resolution).
- **Upscale blur:** accepted.
- **One commit per task; each commit must have a detailed message** (summary of
  what changed and why — see the "Commit messages" section).

---

## Current behavior

`crates/avatar-core/src/talking_head.rs::render_talking_head`:

1. Resample TTS audio to 16 kHz; mel; `video_frames`.
2. Load Wav2Lip + ExpNet + Audio2Pose weights.
3. `expnet.predict` → 64 expression coeffs/frame.
4. `wav2lip::mel_chunks` → chunks.
5. `wav2lip.audio_embeddings` → Audio2Pose → 6 pose values/frame.
6. `load_source_face(picture, size)` → a **size×size** square (face-crop + margin
   + center-square + resize to `size`).
7. `wav2lip::face_input(&src)` (the **whole square**, 6 channels) → `predict` →
   `[3,96,96]` patches (one per chunk).
8. Per frame: `fuse_frame(&src, coeffs, pose, mouth, cfg)` =
   `warp_from_coeffs` → `pose_warp` → `blend_mouth` (fixed ellipse) → resize.

`fusion.rs`:
- `warp_from_coeffs` uses **hardcoded** `face_points_neutral()` positions.
- `blend_mouth` blends the patch into a **fixed** bottom-centre ellipse.

**Defects:** (a) Wav2Lip runs on the unwarped source while its mouth is blended
into the warped frame → seam; (b) control points are canonical, not detected.

---

## Expected behavior

- Wav2Lip runs on a **tightly cropped clean face** (bbox + pads → 96×96); its
  whole output is **pasted back** into the N×N square (2 px feathered boundary),
  so there is **no seam**.
- The expression warp uses **PFLD-detected** control points (98 → 9), on the
  clean face.
- Output MP4 is `size × size` (64/128/256/512), default 256, as today.
- `pose_seed` behaviour (stochastic vs reproducible) is unchanged.
- All existing tests pass; new tests cover the new steps.

---

## Tasks (one commit per task)

> **Commit messages:** every task commit must carry a **detailed message** (a
> short summary line + a body describing what changed, why, and how it was
> verified). This is required for all tasks in this plan.

### Task 1 — Convert PFLD to Burn (`.bpk`)
- Add PFLD to `scripts/convert_weights.py` (done): load
  `.res/pfld_checkpoint.pth.tar`, export `pfld.onnx` (input `input_1
  [1,3,112,112]`, outputs `out1 [1,64,28,28]` + `landmarks [1,196]`), then
  `onnx2burn` → `.res/models/pfld/pfld.bpk`.
- Add the model source + checksum to `scripts/README_WEIGHTS.md` and
  `CHECKSUMS.txt`.
- Commit: `task1: convert PFLD landmarks model to burn (.bpk)`

### Task 2 — Add the `pfld` crate (inference wrapper)
- New `crates/pfld` (mirrors `crates/pose`): load the `.bpk`, run
  `forward(input) -> landmarks [1,196]`, expose `PFLD::predict(&image_112) ->
  Vec<[f32;2]>` (98 normalized points).
- Add a `98 → 9` control-point mapping helper (WFLW layout; see
  `steps-analysis.md`).
- Unit tests: shape (98×2), finiteness, determinism, and that a synthetic face
  yields points in `0..1`.
- Commit: `task2: add pfld crate with 98-landmark inference and 98->9 mapping`

### Task 3 — Expression warp from detected landmarks
- In `fusion.rs`, add `warp_from_landmarks(src, exp_coeffs, landmarks9,
  strength)` (control-point **positions** from landmarks; **displacements** from
  the existing ExpNet mapping).
- Keep `warp_from_coeffs` for now (used by tests) or refactor it to take
  positions; add tests that a nonzero coefficient changes the frame and that the
  warp is bounded.
- Commit: `task3: expression warp driven by detected landmarks`

### Task 4 — Tight bbox crop + in-paint (steps 1 & 3)
- Add a `crop_face`-based path: detect the bbox on the N×N square, crop with
  Wav2Lip pads → M×M → resize to 96×96 for Wav2Lip.
- Add the **in-paint**: resize the 96×96 Wav2Lip output back to M×M and paste it
  into the N×N square at the bbox rect, with a **2 px** feathered boundary.
- Unit tests: paste lands at the bbox; boundary is feathered (no hard edge);
  output size is N×N.
- Commit: `task4: tight bbox crop and 2px-feathered in-paint of the Wav2Lip face`

### Task 5 — Reorder the pipeline (steps 0–6)
- Rewrite `render_talking_head` to the new order:
  `square N×N → bbox crop → Wav2Lip → in-paint → resize 112 → expression warp
  (PFLD) → pose warp → resize`.
- Wav2Lip still runs **once per chunk** with the same clean face reused.
- Keep `pose_seed` threading and the existing `EngineConfig` surface.
- Commit: `task5: reorder talking-head pipeline (wav2lip -> inpaint -> pfld expr -> pose)`

### Task 6 — Tests, docs, cleanup
- End-to-end test: render a short clip; assert frame count, size, no seam
  (boundary continuity), and that expression+pose change frames.
- Update `README.md`, `PERF_NOTES.md`, and `steps-analysis.md` status.
- Remove dead code (`blend_mouth`/`RegionBlend` if unused; `face_points_neutral`
  if replaced).
- Commit: `task6: tests, docs and cleanup for the reordered pipeline`

---

## Verification (unit tests + manual)

- `cargo test --workspace` passes; new tests for:
  - PFLD: 98×2 shape, finite, deterministic, in-range.
  - `98 → 9` mapping: known synthetic points map to the expected control points.
  - Expression warp: nonzero coeff changes the frame; bounded.
  - In-paint: paste at the bbox; 2 px feather; N×N output.
  - End-to-end: frame count, `size×size`, boundary continuity (no seam).
- `python examples/render_clip.py --size 64/128/256/512` renders; visual check
  shows a **single coherent face, no seam, moving lips**.
- `pose_seed=None` differs between runs; `pose_seed=Some(n)` is byte-identical.
- `maturin build` succeeds and `live_avatar` imports.

---

## Cross-references

- Implements: `steps-analysis.md`.
- Builds on: `PLAN_TALKING_HEAD_ONLY.md` (Plan 6), `PLAN_WAV2LIP_EXPNET.md` (Plan 5).

---

## Annex — Open risks

- **PFLD robustness:** verified to need a centered, cropped face (landmarks
  scatter otherwise) — step 1's crop is a prerequisite.
- **Performance:** step 3 pastes into an N×N frame (e.g. 740×740); step 4
  resizes N×N→112×112. Consider capping N (e.g. `min(N, 512)`) if slow.
- **In-paint quality:** a 2 px feather may still show on high-contrast
  boundaries; revisit if visible.
- **ExpNet→displacement mapping** remains hand-tuned (unchanged).

---

## Annex — Future improvements (reference only, NOT part of this plan)

The items below are recorded for possible future work and are **intentionally
out of scope** for this plan. They are **not tasks** and must not be
implemented as part of Plan 7.

### Eye blinking via a dedicated blink model (option B)

**Problem.** The current pipeline has no eyelid motion. Neither ExpNet nor
Wav2Lip produces a blink: ExpNet supplies expression coefficients (brows, eyes,
mouth, chin) but no eyelid closure, and Wav2Lip's non-mouth region is just a
blurry copy of the input face. A blink is **new imagery** (a closed eyelid),
which the warp stages cannot create — `warp_cpu` only *displaces* existing
pixels, and the source face has an open eye with no eyelid texture to reveal.

**Why it does not fit the warp.** Both the expression warp (step 4) and the
pose warp (step 5) are displacement warps. Squashing the eye region vertically
yields a *squished eye*, not a closed eyelid.

**Where it fits.** The **region-blend / in-paint mechanism** (step 3,
`RegionBlend`), which pastes imagery produced by a *model* rather than
displacing existing pixels. A blink would be a second region:

```
RegionBlend {
    src:    <eyelid imagery from the blink model>,
    dst:    <eye rect, derived from PFLD eye landmarks>,
    mask:   <eye ellipse>,
    feather: 2 px,
}
```

driven by a **blink schedule** (periodic, ~every 3–5 s, ~150 ms ≈ 2–4 frames at
25 fps; optionally audio-aware, since blinks cluster at speech pauses) rather
than by the audio.

**The dedicated blink model (option B).** A small model that takes the eye
region (from the current frame) and returns the same region with a **closed
eyelid**. This is the "source" of eyelid pixels that the region blend needs.
Sketch:

- **Input:** the eye crop (e.g. 64×64 or 96×96), optionally conditioned on a
  blink amount `t ∈ [0,1]` (0 = open, 1 = closed) for smooth animation.
- **Output:** the same-size eye crop with the eyelid rendered.
- **Training data:** eye-region crops labelled open/closed (e.g. from
  MRL Eye Dataset, CEW, or frames extracted from talking-head video with
  blink annotations). A simple paired open→closed formulation suffices.
- **Integration:** run per frame (or only on blink frames), then blend via the
  generic `RegionBlend` with the eye rect from PFLD landmarks (points 60–67
  left, 68–75 right). The blink amount `t` drives the schedule.

**Prerequisites already in place.**

- `RegionBlend` is already generic (src/dst/mask/feather), so adding an eye
  region is a one-line addition — provided Task 4 writes the in-paint as a
  **region-blend table** rather than mouth-only.
- PFLD already returns the eye contours and pupils, so the eye rect / mask can
  be derived from detected landmarks (no hardcoded eye position).

**Alternatives (recorded for completeness).**

- **A. Procedural** — warp nearby skin (brow/cheek) over the eye. Cheapest, but
  crude (looks like a smudge). No model needed.
- **C. Full face-reenactment** (LivePortrait / FOMM / face-vid2vid) — the model
  produces the whole face including blinks, gaze and pose. Best quality, but
  replaces ExpNet + Audio2Pose + Wav2Lip entirely (a much larger change).

**Scope note.** This annex is **reference only**. It is not implemented in
Plan 7; the only related action taken in Plan 7 is to keep `RegionBlend`
generic (so this can be added later without reworking the pipeline).
