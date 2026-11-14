# Plan 6 — Talking-head only backend, configurable output sizes, and Audio2Pose head motion

**Status: implemented (tasks 1–7).** All tasks below are committed; the
talking-head backend (Wav2Lip + ExpNet + Audio2Pose) is the only backend.

**References:** `PLAN_WAV2LIP_EXPNET.md` (Plan 5, the talking-head backend being kept),
`PLAN_AVATAR_MODULE.md` (Plan 1, the classic atlas/wgpu path being removed),
`AVATAR_MODULE_REPORT.md`. This plan supersedes the classic backend and the two
unimplemented companion plans.

---

## Goals

- **G1 — Remove the classic backend.** Delete the deterministic viseme + expression-atlas +
  wgpu-renderer path. The Wav2Lip + ExpNet talking-head path becomes the **only** backend.
- **G2 — Configurable output sizes.** Supported output: **64×64, 128×128, 256×256, 512×512** px.
  **Default = 256×256.** The output size is no longer a hardcoded constant.
- **G3 — Remove unimplemented plans.** Delete `plans/PLAN_LIPSYNC_TRAINING.md`,
  `plans/PLAN_SADTALKER_SERVICE.md`, and the `crates/lipsync-train` placeholder. Amend all
  code and docs that reference them.
- **G4 — Model-driven head pose (Audio2Pose).** Add the SadTalker **Audio2Pose** CVAE decoder
  so the head moves (yaw/pitch/roll + translation) from the audio, applied as a 2D affine warp
  in the fusion stage. The latent `z` is **seeded optionally**: `None` = stochastic per render,
  `Some(seed)` = reproducible.

---

## Decisions applied (user-confirmed)

- **Backend:** talking-head is the only backend; the `talking-head` cargo feature gate is
  removed (Wav2Lip + ExpNet are always compiled).
- **Sizes:** `64 | 128 | 256 | 512`, default `256`. `EngineConfig.size` drives the output; the
  `TALKING_HEAD_SIZE` constant is replaced by a validated size set.
- **Plans to remove:** `PLAN_LIPSYNC_TRAINING.md`, `PLAN_SADTALKER_SERVICE.md`,
  `crates/lipsync-train`.
- **Pose determinism:** optional via a seed. `None` = random latent each render; `Some(n)` =
  fixed/reproducible.
- **Pose `class` input:** fixed to a single default class (`0`).
- **Pose `ref` input:** zeros (neutral start).
- **Pose warp:** yaw/pitch/roll → 2D affine (rotate + slight scale/translate); translation
  `tx/ty/tz` → 2D offset + scale.
- **Written plan, one commit per task, changes are committed.**

---

## Current behavior

- Two backends coexist: the classic atlas/wgpu path (default) and the talking-head path behind
  the optional `talking-head` feature.
- Output size is a hardcoded `TALKING_HEAD_SIZE = 256` for talking-head; the classic path uses
  `size ∈ {32, 64, 128}`.
- `crates/lipsync-train` is an unimplemented placeholder; `PLAN_LIPSYNC_TRAINING.md` and
  `PLAN_SADTALKER_SERVICE.md` are unimplemented plans referenced across docs.
- The talking-head output has a **static head**: ExpNet drives mouth/brow/eye/chin expression
  only; there is no head pose, no blink, no head motion.

## Expected behavior

- `Engine(talking_head=...)` is gone; the engine always renders via Wav2Lip + ExpNet (+
  Audio2Pose). `Engine(size=...)` accepts `64 | 128 | 256 | 512`, default `256`.
- The output MP4 is `size × size` at the requested resolution.
- `plans/PLAN_LIPSYNC_TRAINING.md`, `plans/PLAN_SADTALKER_SERVICE.md`, and
  `crates/lipsync-train/` are deleted; no code or docs reference them.
- The talking-head clip has **audio-driven head motion** (yaw/pitch/roll + translation) applied
  as a 2D affine warp, in addition to the existing lip-sync and expression. The render is
  reproducible when a pose seed is set, stochastic otherwise.

---

## Tasks (one commit per task)

### Task 1 — Remove the classic backend from `avatar-core`
- Delete `renderer.rs`, `renderer.wgsl`, `visemes.rs`; remove the classic-only branches from
  `pipeline.rs` (`analysis::analyze` head-bob/breathing, `viseme_frames`, atlas/renderer usage).
- Remove `wgpu`, `bytemuck`, `tokio` deps no longer used; drop `atlas_cpu`, `AtlasBuilder`,
  `warp_params`, `ATLAS_MODES`, `downscale`, `MAX_ATLAS_SOURCE` from `atlas.rs` (keep
  `load_image`, `center_crop_square`, `test_face_cpu`, `save_image`, `warp_cpu`, `ControlPoint`,
  `WarpParams` used by fusion/talking-head).
- Remove the `talking-head` feature gate: `wav2lip` + `expnet` become non-optional deps; the
  `#[cfg(feature = "talking-head")]` guards are removed.
- Update `lib.rs` module list and the `render_with_stats` pipeline to only the talking-head path.
- Commit: `task1: remove classic backend, make talking-head the only path`

### Task 2 — Configurable output sizes (64/128/256/512, default 256)
- Replace `TALKING_HEAD_SIZE` with a validated size set. `EngineConfig.size` defaults to `256`
  and is validated to `{64, 128, 256, 512}` (in both `EngineConfig::with_size` and the Python
  wrapper).
- Wire `cfg.size` through the talking-head path (`FusionConfig::with_size(cfg.size)`).
- Update `avatar-py` `Engine.size` validation and docs; update `examples/render_clip.py`
  `--size` choices.
- Commit: `task2: configurable 64/128/256/512 output, default 256`

### Task 3 — Remove unimplemented plans + `lipsync-train`
- Delete `plans/PLAN_LIPSYNC_TRAINING.md`, `plans/PLAN_SADTALKER_SERVICE.md`,
  `crates/lipsync-train/`.
- Remove `crates/lipsync-train` from workspace `Cargo.toml` members.
- Update `README.md`, `plans/PLAN_WAV2LIP_EXPNET.md`, `plans/PLAN_AVATAR_MODULE.md` to drop
  references to the removed plans/crate.
- Commit: `task3: remove unimplemented lipsync-train and SadTalker-service plans`

### Task 4 — Convert the Audio2Pose model
- Extend `scripts/convert_weights.py` to convert `auido2pose_00140-model.pth` (the CVAE
  **decoder** inference path) to ONNX → Burn `.bpk`. Reuse the Wav2Lip audio encoder for the
  512-d audio embeddings (already present in `crates/wav2lip`).
- Fix the `class` input to a single default and `ref` to zeros; export the decoder with a
  `z` (latent) input.
- Add the checkpoint to `.res/models/CHECKSUMS.txt` and `scripts/README_WEIGHTS.md`.
- Commit: `task4: convert Audio2Pose decoder to burn`

### Task 5 — Add the `pose` crate/module
- Add a `crates/pose` (or `avatar-core::pose`) module that loads the converted decoder weights
  and runs `decoder(ref, audio_emb, class, z)` per sequence → 6 pose values per frame.
- `with_seed(Option<u64>)`: `None` = sample `z ~ N(0,1)` per render; `Some(seed)` = fixed
  reproducible latent.
- Add unit tests (shape, finiteness, determinism with seed, stochasticity without).
- Commit: `task5: pose module with optional-seeded latent`

### Task 6 — Apply pose as a 2D affine warp in fusion
- In `fusion.rs`, add a pose warp stage: map yaw/pitch/roll to a 2D affine (rotate + slight
  scale/translate) and `tx/ty/tz` to 2D offset + scale; apply to the warped source before the
  mouth blend.
- Wire the pose values from `talking_head.rs` into `fuse_frame`.
- Add tests that a nonzero pose changes the frame and that the warp is bounded.
- Commit: `task6: apply audio-driven head pose warp in fusion`

### Task 7 — Wire pose into the pipeline + docs
- Thread the pose seed through `EngineConfig` (and the Python `Engine`) into the talking-head
  path.
- Add the Wav2Lip audio-encoder embedding extractor (`Model::audio_embedding` /
  `Wav2Lip::audio_embeddings`) that Audio2Pose consumes (the encoder was previously only an
  internal stage of `forward`).
- Add the `pose` crate as a dependency of `avatar-core`; load Audio2Pose in
  `render_talking_head`, decode the audio embeddings into a per-frame pose sequence, and feed it
  to `fuse_frame`.
- Update `README.md`, `PERF_NOTES.md`, and the plan cross-references to describe the new
  talking-head-only pipeline, size options, and pose behavior.
- Commit: `task7: wire pose seed through pipeline and update docs`

---

## Verification

- `cargo test` passes across the workspace (classic-only tests removed; talking-head/pose/fusion
  tests added).
- `cargo build` succeeds without the `talking-head` feature flag (feature removed).
- `python examples/render_clip.py --size 64/128/256/512` produces MP4s at the requested
  resolution; default is 256×256.
- A rendered clip with `pose_seed=None` differs between runs; with `pose_seed=Some(n)` it is
  byte-identical across runs (deterministic).
- `git grep` for `PLAN_LIPSYNC_TRAINING`, `PLAN_SADTALKER_SERVICE`, `lipsync-train`, and the
  classic modules returns nothing in code/docs.
- The wheel still builds (`maturin build`) and `live_avatar` imports.

## Cross-references

- Supersedes: `PLAN_AVATAR_MODULE.md` (classic path), `PLAN_LIPSYNC_TRAINING.md`,
  `PLAN_SADTALKER_SERVICE.md`.
- Keeps: `PLAN_WAV2LIP_EXPNET.md` (talking-head backend), `PLAN_FIX_RENDER_ISSUES.md`
  (encoder/mux fixes), `PLAN_FACE_CROP.md` (face detection).
- Report: `AVATAR_MODULE_REPORT.md` §3.6 (atlas/render, superseded), §3.7 (encode/mux, reused),
  §3.8 (Python wrapper).

---

## Annex — Future improvements (reference only, not part of this plan)

The items below are **not tasks** in this plan. They are recorded as reference for possible
future work and are intentionally out of scope.

- **Configurable input `class`.** The Audio2Pose CVAE takes a `class` (speaker/emotion class)
  input. This plan fixes it to a single default (`0`). A future improvement would expose the
  class as a config option (`EngineConfig` / Python `Engine`), letting callers select different
  head-motion styles.
- **Configurable `ref` pose.** The decoder's initial pose (`ref`, 6 values) is fixed to zeros
  here. A future option could accept a user-supplied initial pose.
- **Full 70-dim 3DMM coefficients.** The current ExpNet uses only the first 64 of 70
  coefficients (8 of them in `fusion.rs`). A future improvement could use the full set for
  richer, more faithful expression.
- **Eye blinking.** Neither ExpNet nor Audio2Pose produces a natural eyelid blink. A future
  improvement could add a blink schedule (periodic or audio-aware) driven by the eye
  coefficients or a timer.
- **Full 3D pose warp.** This plan maps pose to a 2D affine warp (rotate/scale/translate). A
  future improvement could apply a true 3D projective warp (perspective) for more realism.
- **Determinism as an explicit toggle.** The pose latent is seeded optionally here. A future
  improvement could make the seed a first-class, documented rendering option with more control
  (e.g. per-clip or per-sequence seeds).
- **Performance tuning of the pose path.** The pose decoder adds a per-sequence forward pass. A
  future improvement could batch/parallelize it or fuse it with the existing ExpNet/Wav2Lip
  batching to reduce per-frame cost.