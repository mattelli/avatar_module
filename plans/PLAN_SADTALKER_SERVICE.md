# Plan 4 — SadTalker as a Python service (complementing the Rust engine)

**References:** `AVATAR_MODULE_REPORT.md` (v3) §2 pipeline, §3.1 markers, §3.6 atlas/render,
§3.7 encode/mux, §3.8 Python wrapper. Companion plans: `PLAN_AVATAR_MODULE.md` (the Rust
engine this plan complements), `PLAN_LIPSYNC_TRAINING.md` (the learned lip-sync the Rust
engine already uses), `PLAN_WAV2LIP_EXPNET.md` (the alternative talking-head backend).

**Decisions applied (user-confirmed):**
- **Integration = complement (1b):** the Rust `avatar-core` engine stays the orchestrator and
  calls out to a Python service that runs SadTalker. No full replacement.
- **Inference = CPU:** no NVIDIA GPU available (AMD machine, Windows 11). PyTorch runs in CPU
  mode; performance expectations are set accordingly.
- **Output = photorealistic high-res (3):** SadTalker's native 512×512 talking head replaces
  the current 32/64/128 px stylized warp as the visual output.
- **Markers dropped (4):** `<happy>`/`<sad>`/... expression markers are removed from the
  pipeline. Expression is fully audio-driven by SadTalker's ExpNet.

## Goals

- **G1** — A Python service (`sadtalker_service`) that wraps SadTalker (PyTorch, CPU) and
  exposes a render endpoint: `(audio, source_image) → talking-head video frames`.
- **G2** — The Rust engine (`avatar-core`) integrates the service **without replacing itself**:
  it keeps TTS (Piper) + audio analysis, and delegates the talking-head generation to the
  service, then encodes the returned frames to MP4 (H.264 + AAC) with the existing encoder.
- **G3** — Marker support removed: `markers::parse` no longer splits on `<happy>`/`<sad>`;
  the whole text is a single neutral span. Expression comes from audio via SadTalker.
- **G4** — Photorealistic high-res output (SadTalker 512×512) replaces the 32/64/128 px
  expression-atlas renderer in the delivered video.
- **G5** — CPU-only operation: model weights, preprocessing, and inference all run on the CPU
  with acceptable throughput for short clips (target: real-time or better on a single clip;
  see Performance).
- **G6** — Reproducible setup: pinned Python env, model-weight download script, config file,
  and a service health-check.

## Current behavior

- The Rust engine renders text → markers → Piper TTS → deterministic/learned viseme lip-sync →
  expression-atlas warp (32/64/128 px) → MP4, entirely in-process.
- `<happy>`/`<sad>`/... markers switch expression bands at timestamps with a ~150 ms cross-fade
  (see `avatar-core::markers`, `avatar-core::pipeline::expression_timeline`).
- No Python talking-head model exists in the codebase; the only Python is the `live_avatar`
  PyO3 wrapper and helper scripts.

## Expected behavior

```
text ──▶ Piper TTS (Rust, unchanged) ──▶ audio (WAV)
                                              │
picture ──────────────────────────────────────┤
                                              ▼
        Python service: SadTalker (PyTorch, CPU)
              audio + image ─▶ 3DMM/ExpNet ─▶ expression + pose
                              ─▶ face render ─▶ 512×512 talking-head frames
                                              │
                                              ▼
        Rust engine: existing encoder/muxer ─▶ MP4 (H.264 + AAC)
```

- `engine.render(text, out)` still works from Python, but the produced video is a
  photorealistic 512×512 talking head driven by audio (no expression markers).
- The Rust engine spawns/connects to the SadTalker service for the talking-head frames, then
  reuses its own H.264/AAC MP4 encoder.
- Text markers are ignored: `<happy> ... <sad> ...` renders as a single neutral, audio-driven
  clip.

## Tasks (one commit per task; simple tasks)

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | Python service skeleton: `sadtalker_service/` package with a config file, pinned `requirements.txt` (torch CPU, torchvision, numpy, opencv, einops, etc.), and a `main.py` entrypoint exposing a health endpoint | `task1: sadtalker service skeleton` | G1, G6 |
| 2 | Model-weight download script (`fetch_weights.py`) that pulls the SadTalker checkpoints (face3d, wav2lip-style audio encoder, mapping, head render) into a local `weights/` dir; verify checksums | `task2: sadtalker weights download` | G1, G6 |
| 3 | SadTalker inference wrapper module: load models on CPU, preprocess source image (face detect/crop/align), run the audio→expression+pose stage (ExpNet) and the head-render stage; unit test on a tiny synthetic audio+image | `task3: sadtalker cpu inference` | G1, G5 |
| 4 | Service endpoint `POST /render` accepting `(audio bytes, image bytes)` and returning video frames (e.g. PNG/raw RGBA per frame or a temp MP4); run behind a simple HTTP server (FastAPI/Flask) or stdlib | `task4: sadtalker render endpoint` | G1 |
| 5 | Rust integration: add a `sadtalker` client module in `avatar-core` that sends the synthesized audio + picture to the service and receives the talking-head frames; config for service URL | `task5: rust sadtalker client` | G2 |
| 6 | Remove expression markers: strip marker parsing from `avatar-core::markers` (single neutral span), remove `expression_timeline` / atlas band switching, drop `<happy>`/`<sad>` handling in `pipeline` and `avatar-py` | `task6: drop expression markers` | G3 |
| 7 | Wire the pipeline: `render()` uses Piper TTS → SadTalker service for frames → existing encoder/muxer → MP4; keep the old warp path behind a config flag for comparison/fallback | `task7: wire sadtalker into pipeline` | G2 |
| 8 | Output sizing: accept/scale SadTalker 512×512 frames and configure output resolution; update the Python `Engine` wrapper (size semantics) and docs | `task8: high-res output config` | G4 |
| 9 | Performance pass: benchmark CPU inference per clip, add frame batching / lower-res intermediate + upscale option, document expected times and any `--cpu-threads` tuning | `task9: cpu perf tuning` | G5 |
| 10 | Tests + docs: unit tests for marker removal and client; end-to-end smoke test producing an MP4; update README + `render_clip.py` | `task10: tests and docs` | G3, G4 |

## Tests

- **T3** — inference module: on a tiny synthetic audio clip + test image, produces the expected
  frame count and 512×512 (or configured) output; runs on CPU without a GPU.
- **T5** — client: with the service running, the Rust client returns frames for a sample audio;
  service-down path returns a clear error.
- **T6** — marker removal: `markers::parse("Hello <happy> world <sad>")` yields a single
  neutral span; the pipeline no longer references expression bands.
- **T7** — end-to-end: `render("Hello world", out)` → MP4 with 2 tracks, video sample count ==
  rendered frame count, duration ≈ audio duration ± 100 ms, frames are 512×512 photorealistic.
- **T9** — performance: a short clip (e.g. 3 s) renders on CPU within a documented budget;
  benchmark numbers recorded in the plan/report.

## Cross-references

- Complements: `PLAN_AVATAR_MODULE.md` (Rust engine: TTS, encoder, wrapper stay in place).
- Alternative backend: `PLAN_WAV2LIP_EXPNET.md` (Wav2Lip + ExpNet combination).
- Report: `AVATAR_MODULE_REPORT.md` §2, §3.1 (markers, now removed), §3.6 (atlas/render,
  superseded by SadTalker for output), §3.7 (encode/mux, reused), §3.8 (Python wrapper).