# avatar_module

Text → MP4 avatar video, entirely in Rust (with a Python ≥ 3.10 wrapper).

Give it a sentence and a front-facing photo, and it produces an MP4 of a
talking avatar. Two backends:

- **Classic (default):** deterministic viseme lip-sync + expression-atlas
  warp at 32/64/128 px.
- **Talking-head (Plan 5):** Wav2Lip + ExpNet (Burn, GPU with CPU fallback)
  producing a photorealistic **256×256** clip — precise lips from Wav2Lip,
  audio-driven expression from ExpNet.

Expression markers (`<happy>`/`<sad>`/...) are stripped — expression is
audio-driven.

```python
import live_avatar

# Classic atlas path (32/64/128 px):
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",  # any Piper voice
    picture="portrait.png",                      # optional; synthetic face otherwise
    size=64,                                     # 32, 64, or 128 px
    face_crop=True,                              # center-crop to the detected face
    face_crop_margin=0.35,                       # margin around the face box
)
engine.render("Hello world. Great news.", "clip.mp4")

# Photorealistic talking-head path (fixed 256x256, requires model weights):
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",
    picture="portrait.png",
    talking_head=True,
)
engine.render("Hello world. Great news.", "clip_256.mp4")
```

## How it works

Classic path:

```
text ──▶ Piper TTS ──▶ audio analysis ──▶ viseme map ──▶ wgpu render ──▶ H.264+AAC ──▶ MP4
        (phonemes,     (RMS → head bob,  (phoneme →     (expression    (openh264,
         timings)       breathing)       mouth shape)    atlas)         FDK-AAC, mp4 crate)
```

Talking-head path (Plan 5):

```
text ──▶ Piper TTS ──▶ audio (16 kHz) ──▶ ExpNet ──▶ expression coefficients ──┐
                                              │                                 ├─▶ fusion (warp + mouth blend) ──▶ 256×256 frames ──▶ MP4
picture ────────────────────────────────▶ Wav2Lip ──▶ mouth patches ────────────┘
```

| Stage | Crate / module | What it does |
|-------|----------------|--------------|
| TTS | `avatar-core::tts` | Embedded Piper (`piper-rs`): PCM + phoneme timings |
| Analysis | `avatar-core::analysis` | 40 ms RMS frames → head bob, breathing |
| Visemes | `avatar-core::visemes` | Phoneme → 12-viseme weights (classic path) |
| Atlas | `avatar-core::atlas` | 7 expression bands warped from one picture (classic path) |
| Render | `avatar-core::renderer` | wgpu compute shader; 32, 64, or 128 px frames @ 25 fps (classic path) |
| ExpNet | `crates/expnet` | Audio → 3DMM-style expression coefficients (Burn, task 3) |
| Wav2Lip | `crates/wav2lip` | Audio + face → precise 96×96 mouth patch (Burn, task 4) |
| Fusion | `avatar-core::fusion` | Per-frame expression warp + mouth-patch blend (task 6) |
| Talking head | `avatar-core::talking_head` | Orchestrates ExpNet + Wav2Lip + fusion → 256×256 frames (task 8) |
| Encode | `avatar-core::encoder` | H.264 (`openh264`) + AAC (`fdk-aac` feature; silent placeholder otherwise) → MP4 (`mp4` crate) |
| Python | `avatar-py` | PyO3 `abi3` wheel, GIL released during render |

## Building

```bash
cargo test                 # 60 tests: parser, TTS, analysis, visemes, atlas, renderer, encoder, pipeline
```

The Python wheel (one wheel for all Python ≥ 3.10 thanks to abi3):

```bash
pip install maturin
maturin build --release -o dist -m crates/avatar-py/Cargo.toml
pip install dist/live_avatar-*.whl
```

> **AAC audio:** the wheel enables the `aac` feature (via `avatar-py/aac`),
> which compiles the FDK-AAC encoder. The FDK-AAC source is bundled inside
> the `fdk-aac-sys` crate and built with a C++ toolchain, so building the
> wheel requires a C++ compiler (MSVC on Windows, `g++`/`clang++` elsewhere).
> Without this feature the MP4 audio track is a silent placeholder.

### Rebuilding the wheel with the talking-head backend

The **talking-head** backend (Wav2Lip + ExpNet, Plan 5) is behind the
`talking-head` cargo feature. A wheel built **without** it only supports the
classic atlas path — `Engine(talking_head=True)` / `render_clip.py
--talking-head` will fail with a clear error. To use the photorealistic
backend, rebuild the wheel with the feature (and, to hear the AAC audio
track, keep the default `aac` feature):

```bash
# From the repo root. Requires maturin + a C++ toolchain (MSVC on Windows).
maturin build --release -o dist -m crates/avatar-py/Cargo.toml --features talking-head
.venv\Scripts\pip.exe install --force-reinstall dist/live_avatar-*.whl   # Windows venv
# or: pip install --force-reinstall dist/live_avatar-*.whl
```

> The `talking-head` feature also requires the model weights
> (`wav2lip_gan.bpk` + `expnet.bpk` under `.res/models/{wav2lip,expnet}/`)
> and a GPU adapter for reasonable speed — see `PERF_NOTES.md`. The classic
> path (default, no feature) stays fully self-contained.

## Python environment setup

The core pipeline only needs the `live_avatar` wheel (install it with
`pip install dist/live_avatar-*.whl` after building, see above). The
example script `examples/render_clip.py` additionally uses a few optional
packages to **play back** the generated video (video + audio) and to dump
individual frames as PNGs.

Install everything into your Python environment with:

```bash
pip install opencv-python sounddevice soundfile imageio_ffmpeg
```

| Package | Used for | Required for |
|---------|----------|--------------|
| `opencv-python` (`cv2`) | video playback window, frame dumping | `examples/render_clip.py` playback / `--frame` |
| `sounddevice` | audio playback (PortAudio) | audio while playing the clip |
| `soundfile` | reads the extracted WAV audio | audio while playing the clip |
| `imageio_ffmpeg` | bundled ffmpeg binary that extracts the MP4 audio track | audio while playing the clip |

All four are optional and the example degrades gracefully: without `cv2` it
renders only; without the audio trio it plays video only and prints a hint.

**Using a virtual environment** (e.g. the project's `.venv`):

```bash
.venv/Scripts/python.exe -m pip install opencv-python sounddevice soundfile imageio_ffmpeg
.venv/Scripts/python.exe examples/render_clip.py
```

**Using a system Python** that already has some of these:

```bash
python -m pip install sounddevice soundfile imageio_ffmpeg   # cv2 usually already present
pip install dist/live_avatar-*.whl                            # the wheel itself
python examples/render_clip.py
```

> **Note:** `live_avatar` is a separate wheel and is never pulled in by the
> packages above — make sure it is installed in the same environment you run
> the example from.

## Resources

The pipeline needs a Piper voice model (`model.onnx` + `model.onnx.json`) and
espeak-ng phonemization data. Test assets live in `.res/`
(`en_US-hfc_female-medium.onnx`); the espeak-ng data ships inside the wheel.

Face detection (optional, plan 4) uses the SeetaFace frontal detector model
`.res/seeta_fd_frontal_v1.0.bin` (BSD-2-Clause), sourced from the
[`rustface`](https://github.com/atomashpolskiy/rustface) repository
(`model/seeta_fd_frontal_v1.0.bin`, 1.2 MB). It ships inside the wheel so
face-crop works out of the box; if the model is missing or detection fails,
the pipeline falls back to using the whole picture (rendering never fails).

## Rust API

```rust
use avatar_core::pipeline::{render, render_with_stats, EngineConfig};

// Classic atlas path (32/64/128 px):
let cfg = EngineConfig::voice(".res/en_US-hfc_female-medium.onnx")
    .with_picture("portrait.png")
    .with_size(64)
    .with_strength(0.5);          // photo ~0.5, cartoon ~1.0
let (frames, elapsed) = render_with_stats("Hello world", "out.mp4", &cfg)?;

// Talking-head path (fixed 256x256, requires the `talking-head` feature +
// model weights in .res/models/{expnet,wav2lip}/*.bpk):
let cfg = EngineConfig::voice(".res/en_US-hfc_female-medium.onnx")
    .with_picture("portrait.png")
    .with_talking_head(true);
let (frames, elapsed) = render_with_stats("Hello world", "out_256.mp4", &cfg)?;
```

## Python API

```python
import live_avatar

live_avatar.version()             # crate version
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",
    picture=None,                 # optional PNG/JPEG
    size=64,                      # 32 | 64 | 128 (classic path)
    strength=0.5,                 # warp strength
    breath_period_secs=4.0,       # breathing cycle
    face_crop=True,               # center-crop to the detected face
    face_crop_margin=0.35,        # margin around the face box
    talking_head=False,           # True = Wav2Lip + ExpNet 256x256 backend
)
frames = engine.render("Hello world", "out.mp4")  # GIL released
```

Expression markers are stripped — expression is audio-driven (Plan 5).

## Example

See `examples/render_clip.py` for a runnable script that renders a short
marked-up clip with the bundled test voice and plays it back (video + audio)
in an OpenCV window. It needs the packages listed under
[Python environment setup](#python-environment-setup).

```bash
python examples/render_clip.py              # render + play with audio
python examples/render_clip.py --no-play    # render only
python examples/render_clip.py --frame 10   # dump frame 10 as PNG
```

> **Input pictures:** the pipeline detects the face (SeetaFace via `rustface`,
> plan 4) and center-crops the photo with a 35% margin before building the
> expression atlas, so an off-center or small face is centered in the output.
> Face-crop can be disabled with `face_crop=False` (or `with_face_crop(false)`
> in Rust); the margin is configurable via `face_crop_margin`. If the model is
> missing or no face is found, the whole picture is used — rendering never
> fails. The cropped region is then center-cropped to a square (the renderer
> requires a square atlas band, so non-square photos — e.g. wide or tall
> images — are handled by cropping the longer axis centered) and downscaled
> to at most 128 px (longest side) before warping. The output is 32, 64, or
> 128 px regardless of the input size. A picture that is mostly one color (e.g.
> a very light photo) will produce a correspondingly flat-looking avatar.

## Plans

- `plans/PLAN_AVATAR_MODULE.md` — this module (tasks 1–11)
- `plans/PLAN_WAV2LIP_EXPNET.md` — Wav2Lip + ExpNet talking-head backend
  (tasks 1–11, all implemented)
- `plans/PLAN_LIPSYNC_TRAINING.md` — learned lip-sync model (`lipsync-train`
  placeholder crate) that will replace the deterministic viseme mapping
- `plans/PLAN_SADTALKER_SERVICE.md` — alternative SadTalker Python service
  (not implemented; the all-Rust Wav2Lip+ExpNet path was chosen instead)
- `plans/PLAN_FIX_RENDER_ISSUES.md` — fixes silent audio, flicker, and input
  picture downscaling

## License

MIT — see [LICENSE](LICENSE).