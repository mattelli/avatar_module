# Investigation Report - Speech-to-2D-Avatar Video Module (Rust + WebGPU + Python)

Version 3 - updated 2026-11-08. Changes from v2: output sizes restricted to 32x32 / 64x64 px
(configurable); Path A (pure-Rust Burn on burn-wgpu) selected for lip-sync training; training
datasets specified; AAC audio confirmed; espeak-ng data bundled with the wheel; project directory
will be set up later by the user (current dir only hosts this report).

## 1. Scope Recap (final decisions)

| # | Requirement | Decision |
|---|---|---|
| 1 | Input | **Text** -> speech synthesized with a Piper voice model -> avatar video |
| 2 | Avatar | 2D avatar generated from a single front-facing picture (photo or stylized/cartoon), **with lip-sync** |
| 3 | Mode markers | `<happy>`, `<mad>`, `<sad>`, ... switch expression **at that point in the timeline**; unknown/missing markers **ignored** |
| 4 | Output | **MP4 (H.264)**; **AAC** audio muxed into the file; <= 25 fps |
| 5 | TTS | Voice model configurable; embedded vs. subprocess chosen by performance |
| 6 | Picture source | Real photo or stylized/cartoon, single front-facing image only |
| 7 | Stack | Rust core, WebGPU (wgpu) rendering **and** GPU inference, Python >= 3.10 wrapper |
| 8 | Sizes | **32x32 and 64x64 px** (configurable) |

## 2. Pipeline

```
text --> Marker Parser --> Piper TTS --> WAV (22.05/44.1 kHz mono)
                                |              |
                                |              +--> RMS envelope (head bob / breathing)
                                |              |
                                |              v
                                |     Audio Features (per video frame @ <=25 fps)
                                |              |
                                |              v
                                |     Lip-Sync Model (Burn, trained on burn-wgpu)
                                |              |  inference on burn-wgpu
                                |              v  mouth parameters / viseme weights
picture --> Expression Atlas --> WebGPU Renderer (wgpu compute) --> RGBA frames
             (variants)            (32/64 px, <=25 fps)                 |
                                                                        v
                                            H.264 (OpenH264) + AAC --> MP4 muxer --> .mp4
```

## 3. Component Findings

### 3.1 Marker parser
Scan for `<tag>` tokens (regex or hand-rolled state machine), split text into `(segment, mode)`
spans. Unknown tags -> dropped silently; missing tags -> neutral expression. Span start times are
known only **after** TTS, so the parser tags spans and the TTS stage reports per-segment durations
to build the expression timeline.

### 3.2 Piper TTS - embedded vs. subprocess

| Criterion | Embedded (`piper-rs` crate) | Subprocess (`piper` binary) |
|---|---|---|
| Latency | No spawn/IPC overhead; best for many short segments | Spawn + pipe overhead per call |
| Throughput | ONNX Runtime in-process, batchable | Same engine, plus serialization cost |
| Robustness | A model crash affects the host process | Crash isolation, easy timeouts |
| Deployment | Cargo build pulls `onnxruntime` + espeak-ng data | Requires external binary install |

**Recommendation:** embedded via **`piper-rs`** (thewh1teagle) as default - best performance for
segment-level synthesis and keeps one Rust binary; keep a **subprocess mode** behind a feature flag
as fallback. Voice model configurable (path to `.onnx` + JSON config). espeak-ng phonemization data
**ships with the wheel** (decision 5).

### 3.3 Audio analysis
Read WAV with **`hound`**. Per video-frame period (40 ms at 25 fps) compute:
- RMS energy -> head bob, breathing scale, brightness;
- audio features for the lip-sync model (see 3.4): mel-spectrogram frames or a compact vector
  (RMS + zero-crossing rate + band energies), windowed to the model's input shape.

### 3.4 Lip-sync model - trained with Burn on the burn-wgpu backend (Path A selected)

**Approach:** a compact audio-to-mouth model maps audio features to **mouth parameters** per
video frame (viseme/blendshape weights or 2D mouth-warp parameters: jaw opening, width, lip
protrusion).

**Architecture (small by design):**
- Input: N frames of audio features (e.g., 8-frame sliding window of mel or band-energy vectors).
- Network: 2-3 dense layers (or 1D conv) -> sigmoid/softmax output of K viseme weights
  (K = 8-12: silence, A, E, I, O, U, M/B/P closed, F/V, ...).
- Output drives a **WGSL compute shader** that warps the mouth region of the avatar picture
  (photo or cartoon) accordingly - same deformation pipeline as the expression atlas.

**Training - Path A (pure Rust, selected):**
- Train with **Burn** on the **`burn-wgpu`** backend (Vulkan/DX12 on Windows).
- Dataset loader, training loop and inference all in Rust; weights stored in Burn format
  (`.mpk` record). Fully aligned with the "Rust + WebGPU" requirement.
- Runtime inference executes on **burn-wgpu** - the same WebGPU stack as rendering.
- Path B (Python/PyTorch -> ONNX -> `burn-onnx` import) remains a documented fallback, not used.

### 3.5 Training datasets (specified)

| Dataset | Content | Use in this project | License / access |
|---|---|---|---|
| **GRID** (Sheffield) | 34 speakers x 1000 sentences, audio+front video, controlled phonetic balance | **Primary viseme training corpus** - clean aligned audio+mouth video | Free for research; download from Univ. of Sheffield / Zenodo |
| **TCD-TIMIT** (Trinity College Dublin) | 62 speakers, straight-on + 30 deg video, phoneme-rich sentences | Speaker/mouth variability augmentation | Free for research; request form at sigmedia.tv |
| **CREMA-D** | 91 actors, 7,442 clips, 6 emotions (happy/sad/angry/...) | **Emotion-conditioned expression training** (ties into `<happy>`/`<sad>` markers) | Open Database License (ODbL); on GitHub/HuggingFace |
| **MEAD** | 60 actors x 8 emotions x 3 intensities, multi-view | Emotion + intensity variation for expression atlas calibration | Research license; request via project page (uniBruce/Mead) |
| **LRS2** (Oxford-BBC) | Large in-the-wild TV speech corpus | Optional scale-up once pipeline works; heavy download, restricted license | Academic license via request |

**Preprocessing (Rust or one-off scripts):** extract per-frame mouth-region landmarks from the
video, crop/normalize mouth crops, compute audio features at 25 fps, derive viseme labels via
forced alignment (e.g., Montreal Forced Aligner) -> (audio features -> viseme weights) pairs.
Because output is 32/64 px, viseme-level accuracy is sufficient - no photoreal teeth/tongue
modeling needed.

### 3.6 Avatar generation & WebGPU rendering - `wgpu`
- Native wgpu on Windows 11 (DX12 backend) - fully supported.
- **Compute shader per frame:** inputs = base texture, expression-variant textures, lip-sync
  mouth parameters, uniforms (expression weights, time, RMS level, blink phase) -> RGBA8 output.
- Expression atlas: neutral + happy/mad/sad/... variants generated by GPU warping (eyebrows,
  mouth curvature, eye squint) - parameterized, works for photos and stylized art.
- Transitions: cross-fade between expression variants over ~150 ms at marker timestamps.
- Output sizes **32x32 and 64x64 px, configurable** (16x16 dropped per decision 1).
- Frame readback via buffer mapping; CPU-side assembly into the encoder.

### 3.7 Encoding & muxing - MP4 with H.264 + AAC (all-Rust, no FFmpeg dependency)

| Stage | Choice | Crates |
|---|---|---|
| Video codec | **H.264** - Cisco OpenH264, bundled source, compiles via `cc` out of the box | `openh264` / `openh264-sys2` |
| Audio codec | **AAC** (decision 4) | `aac` (FDK-AAC) bindings |
| Muxer | Pure-Rust MP4 muxer accepting timestamped H.264 + AAC elementary streams | `mp4`, `muxide`, or `mp4e` |

Notes: `mp4e` supports H.264/HEVC video with AAC and Opus audio; `muxide` is a minimal pure-Rust
muxer for pre-encoded frames. OpenH264 licensing: check MPEG LA terms if distributing prebuilt
binaries commercially (binaries may be sourced under Cisco's license).

### 3.8 Python >= 3.10 wrapper - PyO3 + maturin

```python
import live_avatar

engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",  # configurable Piper model
    picture="portrait.png",
    size=64, fps=25,                             # 32 | 64
    lip_sync_model="visemes.burn",               # Burn-trained lip-sync weights
)
engine.render("Hello! <happy> Great news. <sad> It is over.",
              out="clip.mp4")
```

- One `maturin`-built native extension (`abi3` -> one wheel for Python 3.10+).
- GIL released during render (`py.allow_threads`).
- **espeak-ng phonemization data ships inside the wheel** (decision 5) - no system install needed.
- Optional Python-side helpers for dataset preparation and training-data inspection.

## 4. Risks / Open Points (updated)
1. **32 px floor:** 16x16 output removed; at 32 px the mouth is ~4-8 px - visemes are clearly
   perceptible. Size is configurable (32 or 64).
2. **Burn training maturity:** Path A keeps the whole ML loop in Rust; dataset preprocessing
   (landmark extraction, forced alignment) may still need external tools (e.g., Montreal Forced
   Aligner) run once, offline.
3. **Dataset licensing:** GRID / TCD-TIMIT / CREMA-D / MEAD are research licenses - verify terms
   before any commercial redistribution of trained weights.
4. **AAC in pure Rust:** FDK-AAC bindings are the pragmatic choice; no pure-Rust AAC encoder
   of comparable quality exists.
5. **H.264 licensing** for commercial prebuilt-binary distribution (MPEG LA / Cisco terms).
6. **espeak-ng data** ships with the wheel (decision 5) - increases wheel size (~20 MB), accepted.
7. **Project directory:** the current directory only hosts this report; the user will set up the
   actual project directory later. `git init` will be the first task of the implementation plan
   (one commit per task requires it).

## 5. Recommended Crate Set (updated)
`piper-rs`, `hound`, `burn` + `burn-wgpu` (training **and** inference), `wgpu`, `image`,
`openh264`, `aac` (FDK), `mp4` / `muxide` / `mp4e`, `anyhow`/`thiserror`, `pyo3` + `maturin`.
