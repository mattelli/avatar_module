# Plan 1 — Avatar Module: text → MP4 avatar video (no lip-sync training pipeline)

**References:** `AVATAR_MODULE_REPORT.md` (v3) — this plan implements report §2 (pipeline),
§3.1–§3.3, §3.6–§3.8, and §3.4 **option b** (deterministic phoneme→viseme mapping).
The learned lip-sync model is a separate plan: `PLAN_LIPSYNC_TRAINING.md` (report §3.4 Path A,
§3.5 datasets), which later replaces this plan's task 5 mapping.

**Decisions applied:** Q1=b (deterministic phoneme→viseme lip-sync), Q2=one Cargo workspace
(`avatar-core` + `avatar-py` + `lipsync-train` placeholder), Q4=Python wrapper in this plan,
Q5=pin `openh264` + `aac` (FDK) + `mp4` crate (`muxide` fallback), Q6=local Windows 11 tests
with AMD GPU.

## Goals

- **G1** — Cargo workspace `avatar` with crates: `avatar-core` (pipeline), `avatar-py`
  (Python ≥ 3.10 wrapper), `lipsync-train` (placeholder for Plan 2).
- **G2** — Text input with `<mode>` markers → Piper TTS (embedded `piper-rs`, configurable
  voice model) → WAV. Report §3.2.
- **G3** — Deterministic phoneme→viseme lip-sync from Piper phoneme timings. Report §3.4 (b).
- **G4** — 2D avatar from a single front-facing picture (photo or cartoon); expression atlas
  via GPU warping; `<happy>`/`<mad>`/`<sad>`/… switch at marker timestamps (~150 ms cross-fade);
  unknown/missing markers ignored. Report §3.1, §3.6.
- **G5** — WebGPU (`wgpu`) compute rendering at **32×32 / 64×64 px** (configurable), ≤ 25 fps.
  Report §3.6.
- **G6** — MP4 output: **H.264** (`openh264`) + **AAC** (`aac` FDK bindings), muxed with the
  `mp4` crate (`muxide` fallback); audio muxed into the file. Report §3.7.
- **G7** — Python ≥ 3.10 wrapper (PyO3 + maturin, `abi3`, GIL released during render);
  espeak-ng data shipped with the wheel. Report §3.8.
- **G8** — RMS-driven head bob / breathing from the audio envelope. Report §3.3.

## Current behavior

- No project exists: the working directory is not a git repository; only
  `AVATAR_MODULE_REPORT.md` exists. No code, no pipeline, no tests.

## Expected behavior

```python
import live_avatar
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",  # configurable Piper model
    picture="portrait.png",
    size=64, fps=25,                             # 32 | 64
)
engine.render("Hello! <happy> Great news. <sad> It is over.", out="clip.mp4")
```

- Produces a playable MP4 (H.264 + AAC) whose duration ≈ synthesized audio duration.
- Mouth animates per viseme (deterministic mapping from Piper phoneme timings).
- Expressions switch at marker timestamps with cross-fade; unknown/missing markers ignored.
- Subtle head bob / breathing driven by RMS envelope.
- Output size 32×32 or 64×64 px, fps ≤ 25.
- All unit tests pass locally on Windows 11 (AMD GPU available for wgpu tests).

## Tasks (one commit per task; simple tasks)

> Note: the user will set up the project directory later. Task 1 creates the repo wherever the
> user points us; until then, all work is prepared in the current directory.

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | `git init` + Cargo workspace skeleton: `avatar-core`, `avatar-py`, `lipsync-train` (placeholder), `cargo test` green | `task1: workspace skeleton` | G1 |
| 2 | Marker parser: scan `<tag>` tokens, split into `(segment, mode)` spans; unknown/missing → neutral; unit tests | `task2: marker parser` | G4, §3.1 |
| 3 | Piper TTS integration (embedded `piper-rs`): configurable voice model, per-segment synthesis + phoneme timings; subprocess fallback behind feature flag; espeak-ng data bundled for the wheel | `task3: piper tts` | G2, §3.2 |
| 4 | Audio analysis: WAV read (`hound`), per-frame RMS envelope (40 ms @ 25 fps) → head bob / breathing params; audio feature frames for lip-sync | `task4: audio analysis` | G8, §3.3 |
| 5 | Deterministic phoneme→viseme mapping: Piper phoneme timings → viseme weights per frame (hardcoded viseme table, no ML) | `task5: phoneme viseme map` | G3, §3.4(b) |
| 6 | Expression atlas: GPU warping of base picture into neutral/happy/mad/sad/… variants; parameterized for photos and cartoons | `task6: expression atlas` | G4, §3.6 |
| 7 | wgpu renderer: compute shader per frame (base + variant textures + viseme weights + uniforms), 32/64 px, ≤ 25 fps, cross-fade ~150 ms, frame readback | `task7: wgpu renderer` | G5, §3.6 |
| 8 | H.264 encoder (`openh264`) + AAC encoder (`aac` FDK) + MP4 muxer (`mp4` crate; `muxide` fallback); audio muxed in | `task8: encode mux mp4` | G6, §3.7 |
| 9 | Pipeline glue: text → markers → TTS → analysis → visemes → render → encode → `.mp4`; end-to-end integration test | `task9: pipeline glue` | G1–G8 |
| 10 | Python wrapper (`avatar-py`): PyO3 + maturin, `abi3`, `Engine` class, GIL released during render; espeak-ng data in wheel; wheel build test | `task10: python wrapper` | G7, §3.8 |
| 11 | Documentation: README, API docs, example script; tag `v0.1.0` | `task11: docs v0.1.0` | G7 |

## Unit-test verification

- **T2** — parser: `"<happy> hi"` → 1 span mode=happy; `"<xyz> hi"` → unknown ignored, neutral;
  `"<sad> a <mad> b"` → 2 spans in order; no markers → single neutral span.
- **T3** — TTS: given a bundled test voice, synthesis returns non-empty WAV; duration > 0;
  phoneme timings count > 0 and monotonic.
- **T4** — envelope: synthetic WAV (sine burst) → RMS peaks at expected frame indices; silence →
  near-zero RMS.
- **T5** — viseme map: phoneme sequence `M A P` → closed-open-closed weight pattern; unknown
  phoneme → neutral viseme.
- **T6** — atlas: warping a synthetic 64×64 test face yields variants differing from neutral
  (pixel diff > threshold) in mouth/eyebrow regions.
- **T7** — renderer: wgpu device available (AMD GPU); renders 32 and 64 px frames; output
  dimensions correct; cross-fade between two variants produces intermediate pixel values.
- **T8** — encode/mux: encode N synthetic frames + synthetic AAC → MP4 parses (moov box present,
  2 tracks, durations match); `muxide` fallback path produces equivalent file.
- **T9** — end-to-end: `render("Hello <happy> world", out)` → MP4 exists, duration ≈ audio
  duration ± 100 ms, size 32 or 64 px.
- **T10** — wrapper: wheel builds (`maturin build`); import in Python ≥ 3.10; `Engine.render`
  produces MP4 from Python; GIL released (render runs in `py.allow_threads`).

## Cross-references

- Implements report: `AVATAR_MODULE_REPORT.md` §2, §3.1, §3.2, §3.3, §3.4(b), §3.6, §3.7, §3.8.
- Companion plan: `PLAN_LIPSYNC_TRAINING.md` — trains the Burn lip-sync model (report §3.4
  Path A, §3.5 datasets); its trained model replaces task 5's deterministic mapping at runtime
  (inference on `burn-wgpu`, same renderer inputs).
