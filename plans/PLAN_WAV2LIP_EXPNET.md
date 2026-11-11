# Plan 5 — Wav2Lip + ExpNet combined talking-head backend (Rust / burn-wgpu)

**References:** `AVATAR_MODULE_REPORT.md` (v3) §2 pipeline, §3.1 markers, §3.6 atlas/render,
§3.7 encode/mux, §3.8 Python wrapper. Companion plans: `PLAN_AVATAR_MODULE.md` (Rust engine),
`PLAN_SADTALKER_SERVICE.md` (the alternative talking-head backend, which stays a **Python
service**), `PLAN_LIPSYNC_TRAINING.md` (the Rust engine's existing learned lip-sync, already on
burn-wgpu).

**Decisions applied (user-confirmed):**
- **Integration = complement (1b):** the Rust `avatar-core` engine stays the orchestrator. The
  Wav2Lip + ExpNet models are added as **Rust crates in the same workspace** (`burn-wgpu`), not
  as an external Python service. No full replacement.
- **Runtime = Rust / burn-wgpu:** both models run as inference-only Rust code on Burn's `wgpu`
  backend. This reuses the same framework the `lipsync-train` plan already commits to.
- **GPU with CPU fallback:** primary target is the GPU via `burn-wgpu` (DX12/Vulkan on the AMD
  machine); when no GPU/adapter is available, fall back to Burn's `ndarray` CPU backend. The
  runtime picks the backend automatically.
- **Output = photorealistic high-res (3):** the combined model produces talking-head video at
  the source image resolution (typically 96×96 mouth patch upscaled into the full frame).
- **Markers dropped (4):** `<happy>`/`<sad>`/... expression markers are removed; expression is
  audio-driven by ExpNet.

## Why combine Wav2Lip and ExpNet

- **Wav2Lip** is a lip-sync GAN: it takes an audio window + a (possibly muted) face frame and
  outputs a frame with the mouth re-rendered to match the speech. It is excellent at *lip
  articulation* but does **not** model expression — it leaves the rest of the face as-is.
- **ExpNet** (from SadTalker) maps the audio to 3DMM expression coefficients, i.e. the
  *non-lip* facial motion (eyebrows, cheeks, subtle expression changes) driven by the audio.
- Combining them: **ExpNet supplies the audio-driven expression/pose deformation**, and
  **Wav2Lip supplies the precise mouth shape**, so the two are fused per frame instead of
  relying on either alone.

## Feasibility note (Rust / wgpu)

Both models are **inference-only** here (no training, no discriminator), which keeps them in
reach of Burn:

- **ExpNet** — a small MLP over audio features. **Low risk**; runs trivially on burn-wgpu.
- **Wav2Lip generator** — a convolutional net (audio conv encoder, visual conv encoder, fusion,
  decoder with residual blocks + transposed convolutions + batch norm). All standard ops that
  Burn supports on the wgpu backend. **Feasible, moderate-to-high effort** — the work is weight
  conversion and layer plumbing, not the math.

The recommended stack is **burn + burn-wgpu** (consistent with `PLAN_LIPSYNC_TRAINING.md`),
with weights imported via Burn's ONNX importer (`burn-import`). Hand-writing raw WGSL for every
layer is possible but not recommended (large, fragile, no benefit over Burn).

## Goals

- **G1** — Two Rust crates in the workspace: `wav2lip` and `expnet`, each exposing a
  `burn-wgpu` inference module that loads converted weights and runs the model on GPU (with CPU
  fallback).
- **G2** — Weight conversion pipeline: PyTorch checkpoints → ONNX → Burn-importable records,
  with a reproducible script and checksums.
- **G3** — A fusion stage that combines ExpNet's expression/pose deformation with Wav2Lip's
  mouth patch per frame.
- **G4** — The Rust engine integrates the crates as a complement: keeps Piper TTS + audio
  analysis, delegates talking-head generation to the fused Wav2Lip+ExpNet, then encodes returned
  frames to MP4 (H.264 + AAC) with the existing encoder.
- **G5** — Marker support removed: `<happy>`/`<sad>` parsing dropped; the whole text is a single
  neutral span; expression is audio-driven by ExpNet.
- **G6** — GPU-first with CPU fallback: backend auto-selection (burn-wgpu when an adapter is
  available, burn-ndarray otherwise), with documented performance for both.
- **G7** — Photorealistic high-res output at the source image resolution.

## Current behavior

- The Rust engine renders text → markers → Piper TTS → viseme lip-sync → expression-atlas warp
  (32/64/128 px) → MP4, entirely in-process.
- `<happy>`/`<sad>` markers switch expression bands with ~150 ms cross-fade.
- No Wav2Lip/ExpNet code exists; `lipsync-train` already uses burn-wgpu, so the framework is
  proven in the workspace.

## Expected behavior

```
text ──▶ Piper TTS (Rust, unchanged) ──▶ audio (WAV)
                                              │
picture ──────────────────────────────────────┤
                                              ▼
        Rust crates (burn-wgpu, CPU fallback)
              audio ─▶ ExpNet ─▶ expression/pose coefficients
              audio + face frames ─▶ Wav2Lip ─▶ precise mouth
              fuse expression + lip per frame ─▶ talking-head frames
                                              │
                                              ▼
        Rust engine: existing encoder/muxer ─▶ MP4 (H.264 + AAC)
```

- `engine.render(text, out)` still works from Python, producing a photorealistic talking head
  whose lips follow the audio (Wav2Lip) and whose expression is audio-driven (ExpNet).
- Text markers are ignored: `<happy> ... <sad> ...` renders as a single neutral, audio-driven
  clip.
- Backend auto-selection: GPU (burn-wgpu) when available, CPU (burn-ndarray) otherwise.

## Tasks (one commit per task; simple tasks)

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | Workspace scaffolding: add `wav2lip` and `expnet` crates to the Cargo workspace; add `burn`, `burn-ndarray`, `burn-wgpu`, `burn-import` deps; wire into `avatar-core` | `task1: wav2lip+expnet crates` | G1 |
| 2 | Weight conversion tooling: Python script that converts the PyTorch Wav2Lip + ExpNet checkpoints to ONNX, then Burn-imports them into `.burn` records; pin torch CPU; verify checksums | `task2: weight conversion pipeline` | G2 |
| 3 | ExpNet module: load converted weights, preprocess image + audio features, run audio→expression coefficients on burn-wgpu; unit test on synthetic audio | `task3: expnet burn inference` | G1, G6 |
| 4 | Wav2Lip module: face detection/crop, audio-window mel feature extraction, lip-sync GAN inference on burn-wgpu; unit test on synthetic audio+image | `task4: wav2lip burn inference` | G1, G6 |
| 5 | Backend auto-selection: a small runtime helper that picks burn-wgpu (GPU) when an adapter is available and falls back to burn-ndarray (CPU); config flag to force either | `task5: gpu/cpu backend selection` | G6 |
| 6 | Fusion stage: per frame combine ExpNet expression/pose deformation with Wav2Lip's mouth patch (warp/mask blend); unit test that both signals appear in the output | `task6: wav2lip+expnet fusion` | G3 |
| 7 | Remove expression markers: strip marker parsing from `avatar-core::markers`, remove `expression_timeline`/atlas band switching, drop marker handling in `pipeline` and `avatar-py` | `task7: drop expression markers` | G5 |
| 8 | Wire the pipeline: `render()` uses Piper TTS → fused Wav2Lip+ExpNet for frames → existing encoder/muxer → MP4; keep old warp path behind a config flag for comparison/fallback | `task8: wire wav2lip+expnet into pipeline` | G4 |
| 9 | Output sizing: configure output resolution (source-image-based); update Python `Engine` wrapper (size semantics) and docs | `task9: high-res output config` | G7 |
| 10 | Performance pass: benchmark GPU and CPU inference per clip, add frame batching / lower-res intermediate + upscale, document times and backend tuning | `task10: perf tuning (gpu/cpu)` | G6 |
| 11 | Tests + docs: unit tests for marker removal, fusion, and backend selection; end-to-end smoke test producing an MP4 on both backends; update README + `render_clip.py` | `task11: tests and docs` | G5, G6, G7 |

## Tests

- **T2** — weight conversion: PyTorch → ONNX → `.burn` records round-trip; shapes and value
  ranges match the original checkpoint on a sample input.
- **T3** — ExpNet: on synthetic audio + test image, produces a valid expression-coefficient
  sequence on burn-wgpu (and on the CPU fallback).
- **T4** — Wav2Lip: on synthetic audio+image, produces lip-synced frames at the expected
  resolution; mouth region changes with audio.
- **T5** — backend selection: with a GPU adapter present, inference runs on burn-wgpu; with the
  adapter disabled/absent, it runs on burn-ndarray and produces the same frame count/shape.
- **T6** — fusion: output frames contain both the ExpNet-driven expression deformation and the
  Wav2Lip mouth patch; no visual conflict (mask boundaries clean).
- **T7** — marker removal: `markers::parse("Hello <happy> world <sad>")` yields a single
  neutral span; pipeline no longer references expression bands.
- **T8** — end-to-end: `render("Hello world", out)` → MP4 with 2 tracks, video sample count ==
  rendered frame count, duration ≈ audio duration ± 100 ms, frames photorealistic at the
  configured resolution.
- **T10** — performance: a short clip (e.g. 3 s) renders within a documented budget on both the
  GPU (burn-wgpu) and CPU (burn-ndarray) backends; benchmark numbers recorded.

## Cross-references

- Complements: `PLAN_AVATAR_MODULE.md` (Rust engine: TTS, encoder, wrapper stay in place).
- Framework reuse: `PLAN_LIPSYNC_TRAINING.md` (already uses burn-wgpu in the workspace).
- Alternative backend: `PLAN_SADTALKER_SERVICE.md` (SadTalker as a **Python service** — the
  contrast to this plan's all-Rust approach).
- Report: `AVATAR_MODULE_REPORT.md` §2, §3.1 (markers, now removed), §3.6 (atlas/render,
  superseded for output), §3.7 (encode/mux, reused), §3.8 (Python wrapper).