# Plan 3 — Fix `render_clip.py` output: silent audio, flickering video, and input picture downscaling

**References:** `PLAN_AVATAR_MODULE.md` task 8 (encoder/muxer), task 9 (pipeline glue), task 6
(expression atlas). Companion: `AVATAR_MODULE_REPORT.md` §3.7 (encode/mux), §3.6 (atlas/render).

**Context:** After adding `picture=".res/avatar.jpg"` to `examples/render_clip.py` (commit
`e8661e3`), the produced `clip.mp4` shows a flickering white image and has no audible voice.
Root-cause analysis (see below) shows three independent defects:

1. **Silent audio** — the `aac` feature of `avatar-core` is not enabled in the wheel build, so
   `AacEncoder::encode_pcm` takes the `#[cfg(not(feature = "aac"))]` placeholder path and writes
   silent AAC frames.
2. **Flickering video** — `encode_and_mux` splits the H.264 Annex-B stream into per-frame MP4
   samples by *guessing* the NAL count per frame, which does not match OpenH264's actual output
   (IDR frames carry SPS/PPS + slice; P-frames are one slice; periodic keyframes add more). The
   misaligned sample boundaries corrupt frame reconstruction → flicker.
3. **White image** — the source `avatar.jpg` is a near-white photo; this is *content*, not a bug.
   However, the atlas is warped at full source resolution (740×740 → 740×5180 atlas) before the
   renderer downsamples to 32/64 px, which is wasteful. Downscaling the input picture up-front is
   a performance fix and does not change the (white) look.

## Goals

- **G1** — Make the wheel produce an audible AAC audio track (obtain/bundle the FDK-AAC native
  library, enable the `aac` feature, and wire it through the Python build).
- **G2** — Eliminate video flicker by making H.264 → MP4 sample boundaries accurate (one sample
  per real frame, using exact byte ranges instead of NAL-count guessing).
- **G3** — Downscale the input picture to a small working resolution (128 px) before atlas
  warping, as a performance fix (no visual change).
- **G4** — Keep all existing unit tests green and add regression tests for the encoder fix.

## Current behavior

- `examples/render_clip.py` renders a clip whose audio track is silent (AAC placeholder path).
- The video track flickers because `encode_and_mux` splits H.264 NALs into MP4 samples using a
  heuristic (`nal_count / frames.len()`) that misaligns with real frame boundaries.
- The expression atlas is built at the input picture's full resolution (e.g. 740×740 → 740×5180)
  before being downsampled to the 32/64 px output, wasting CPU/GPU work.

## Expected behavior

- `examples/render_clip.py` produces a playable MP4 with an audible voice track.
- The video is stable (no flicker): each MP4 video sample corresponds to exactly one rendered
  frame, in order, with correct duration.
- The pipeline downscales any input picture to ≤128 px (longest side) before atlas warping,
  reducing warp cost ~25× for a 740×740 source with no change to the 32/64 px output.
- All unit tests pass (`cargo test`) and the wheel still builds (`maturin build`).

## Tasks (one commit per task; simple tasks)

> **Commit policy:** the coding agent shall perform **one commit per completed task**, with a
> brief but detailed commit message describing what changed and why (e.g. the exact bug fixed,
> the file(s) touched, and the observable effect). Each task in the table below is committed
> immediately once its work and tests are complete.

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | Obtain/bundle FDK-AAC native library: download/build the FDK-AAC native library for the target platform, vendor it in the repo (e.g. `third_party/fdk-aac/` or a build script), and ensure `fdk-aac`/`fdk-aac-sys` finds it at build time | `task1: bundle fdk-aac native lib` | G1 |
| 2 | Enable `aac` feature: add `avatar-core/aac` to `avatar-py`'s feature forwarding (and/or maturin `features` in `pyproject.toml`); document the bundled FDK-AAC requirement in README | `task2: enable aac audio` | G1 |
| 3 | Encoder fix: rewrite `encode_and_mux` to encode each frame separately and record exact per-frame byte ranges of the Annex-B bitstream; build one AVCC MP4 sample per frame from those ranges (drop the NAL-count heuristic) | `task3: accurate frame samples` | G2 |
| 4 | Picture downscale: in `pipeline.rs`, after `load_image`, downscale the source to ≤128 px (longest side) before `atlas_cpu`; add a small helper + unit test | `task4: downscale input picture` | G3 |
| 5 | Regression tests + docs: test that a multi-frame render yields N MP4 video samples each decodable in order (no flicker proxy: sample count == frame count, monotonic timestamps); update README/example notes | `task5: regression tests + docs` | G2, G4 |

## Unit-test verification

- **T1** — FDK-AAC bundling: the native library is vendored in the repo and `fdk-aac`/`fdk-aac-sys`
  links against it at build time on the target platform.
- **T2** — feature wiring: `cargo build --features aac` compiles the FDK path; wheel build
  (`maturin build`) succeeds with `aac` enabled.
- **T3** — encoder: encode N synthetic frames → the AVCC sample list has exactly N samples;
  each sample's length-prefixed NALs sum to the frame's exact byte range; SPS/PPS only on
  sample 0; sample timestamps are monotonic at 1/fps.
- **T4** — downscale: a 740×740 test image passed through the pipeline helper becomes ≤128 px
  (longest side) with aspect preserved; a 64×64 image is unchanged (already ≤128).
- **T5** — end-to-end: `render("Hello <happy> world", out)` → MP4 parses with 2 tracks; video
  sample count == rendered frame count; audio track present with non-silent AAC payload (when
  `aac` feature on); duration ≈ audio duration ± 100 ms.

## Cross-references

- Implements fix for: `PLAN_AVATAR_MODULE.md` task 8 (encoder/muxer), task 9 (pipeline glue),
  task 6 (atlas source handling).
- Report: `AVATAR_MODULE_REPORT.md` §3.7 (encode/mux), §3.6 (atlas/render).
- Example: `examples/render_clip.py` (the script whose output is being fixed).