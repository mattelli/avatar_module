# avatar_module

Text → MP4 avatar video, entirely in Rust (with a Python ≥ 3.10 wrapper).

Give it a sentence and a front-facing photo, and it produces an MP4 of a
photorealistic talking avatar. The **talking-head** backend is the only
backend (Plan 6): Wav2Lip + ExpNet + Audio2Pose (Burn, GPU with CPU fallback)
produce a clip at **64×64, 128×128, 256×256, or 512×512** px (default
**256×256**) — precise lips from Wav2Lip, audio-driven expression from ExpNet,
and audio-driven head motion (yaw/pitch/roll + translation) from Audio2Pose.

Expression markers (`<happy>`/`<sad>`/...) are stripped — expression is
audio-driven.

```python
import live_avatar

engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",  # any Piper voice
    picture="portrait.png",                      # optional; synthetic face otherwise
    size=256,                                    # 64, 128, 256, or 512 px
    face_crop=True,                              # center-crop to the detected face
    face_crop_margin=0.35,                       # margin around the face box
    pose_seed=None,                              # None = stochastic head motion; int = reproducible
)
engine.render("Hello world. Great news.", "clip.mp4")
```

## How it works

```
text ──▶ Piper TTS ──▶ audio (16 kHz) ──▶ ExpNet ──▶ expression coefficients ──┐
                                          │                                     │
                                          └▶ Wav2Lip audio encoder ──▶ 512-d ──▶ Audio2Pose ──▶ pose (6/frame) ─┤
picture ──▶ square N×N ──▶ bbox crop ──▶ Wav2Lip ──▶ face ──▶ PFLD expr warp ──┴─▶ in-paint ──▶ pose warp ──▶ N×N frames ──▶ MP4
```

| Stage | Crate / module | What it does |
|-------|----------------|--------------|
| TTS | `avatar-core::tts` | Embedded Piper (`piper-rs`): PCM + phoneme timings |
| ExpNet | `crates/expnet` | Audio → 3DMM-style expression coefficients (Burn, task 3) |
| Wav2Lip | `crates/wav2lip` | Audio + face → precise 96×96 face; also exposes the 512-d audio encoder embedding (Burn, task 4) |
| PFLD | `crates/pfld` | 112×112 face → 98 landmarks → 9 expression-warp control points (Burn, Plan 7) |
| Audio2Pose | `crates/pose` | Audio embeddings → head-pose motion (yaw/pitch/roll + tx/ty/tz), optional-seeded latent (Burn, Plan 6) |
| Fusion | `avatar-core::fusion` | Per-frame landmark-driven expression warp + 2D affine pose warp (Plan 7) |
| In-paint | `avatar-core::inpaint` | Tight bbox crop + 2 px-feathered paste of the Wav2Lip face back into the N×N square (Plan 7) |
| Talking head | `avatar-core::talking_head` | Orchestrates the reordered pipeline → N×N frames (Plan 7) |
| Encode | `avatar-core::encoder` | H.264 (`openh264`) + AAC (`fdk-aac` feature; silent placeholder otherwise) → MP4 (`mp4` crate) |
| Python | `avatar-py` | PyO3 `abi3` wheel, GIL released during render |

## Building

```bash
cargo test                 # 93 tests: parser, TTS, encoder, pipeline, expnet, wav2lip, pfld, pose, fusion, inpaint, talking-head
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

### Rebuilding the wheel

The talking-head backend (Wav2Lip + ExpNet + Audio2Pose) is the **only**
backend, so it is always compiled — there is no feature flag to enable. A
wheel built with the default `aac` feature has an audible audio track:

```bash
# From the repo root. Requires maturin + a C++ toolchain (MSVC on Windows).
maturin build --release -o dist -m crates/avatar-py/Cargo.toml
.venv\Scripts\pip.exe install --force-reinstall dist/live_avatar-*.whl   # Windows venv
# or: pip install --force-reinstall dist/live_avatar-*.whl
```

> The backend requires the model weights (`wav2lip_gan.bpk`, `expnet.bpk`,
> `pose.bpk`, and `pfld.bpk` under `.res/models/{wav2lip,expnet,pose,pfld}/`)
> and a GPU adapter for reasonable speed — see `PERF_NOTES.md`.

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

// Talking-head backend (the only backend; requires model weights in
// .res/models/{wav2lip,expnet,pose,pfld}/*.bpk):
let cfg = EngineConfig::voice(".res/en_US-hfc_female-medium.onnx")
    .with_picture("portrait.png")
    .with_size(256)               // 64 | 128 | 256 | 512
    .with_strength(0.5)           // photo ~0.5, cartoon ~1.0
    .with_pose_seed(Some(42));    // None = stochastic head motion
let (frames, elapsed) = render_with_stats("Hello world", "out.mp4", &cfg)?;
```

## Python API

```python
import live_avatar

live_avatar.version()             # crate version
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",
    picture=None,                 # optional PNG/JPEG
    size=256,                     # 64 | 128 | 256 | 512
    strength=0.5,                 # warp strength
    face_crop=True,               # center-crop to the detected face
    face_crop_margin=0.35,        # margin around the face box
    backend="auto",               # "auto" | "gpu" | "cpu"
    pose_seed=None,               # None = stochastic head motion; int = reproducible
)
frames = engine.render("Hello world", "out.mp4")  # GIL released
```

Expression markers are stripped — expression is audio-driven (Plan 5). Head
motion (yaw/pitch/roll + translation) is audio-driven (Audio2Pose, Plan 6);
pass `pose_seed=<int>` for a reproducible render, or leave `None` for a fresh
latent each render.

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

> **Input pictures:** the pipeline makes the picture square at its native
> resolution, detects the face (SeetaFace via `rustface`, plan 4), and crops
> it with the Wav2Lip pads for the lip-sync model; the re-rendered face is then
> pasted back into the square (Plan 7). If the model is missing or no face is
> found, the whole square is used — rendering never fails. The final frame is
> resized to the configured output size (64, 128, 256, or 512 px). A picture
> that is mostly one color (e.g. a very light photo) will produce a
> correspondingly flat-looking avatar.
>
> **Note:** the `face_crop` / `face_crop_margin` options are retained for API
> compatibility but are **not wired** into the reordered talking-head pipeline
> (the detected-bbox crop is intrinsic to steps 1–4).

## Plans

- `plans/PLAN_AVATAR_MODULE.md` — original module plan (classic path, superseded)
- `plans/PLAN_WAV2LIP_EXPNET.md` — Wav2Lip + ExpNet talking-head backend
  (tasks 1–11, all implemented)
- `plans/PLAN_TALKING_HEAD_ONLY.md` — talking-head only: classic backend removed,
  configurable sizes (64/128/256/512), Audio2Pose head motion
- `plans/PLAN_FIX_RENDER_ISSUES.md` — fixes silent audio, flicker, and input
  picture downscaling
- `plans/PLAN_REORDER_PIPELINE.md` — reordered pipeline: Wav2Lip on a tight
  crop → PFLD expression warp → 2 px in-paint → pose warp (Plan 7, implemented)

## License

MIT — see [LICENSE](LICENSE).