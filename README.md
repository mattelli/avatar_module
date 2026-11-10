# avatar_module

Text → MP4 avatar video, entirely in Rust (with a Python ≥ 3.10 wrapper).

Give it a sentence with optional `<happy>` / `<mad>` / `<sad>` … markers and a
front-facing photo, and it produces a small MP4 of a talking avatar: mouth
shapes follow the phonemes (deterministic viseme mapping), the expression
switches at each marker with a ~150 ms cross-fade, and the head gently bobs
and breathes with the voice envelope.

```python
import live_avatar

engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",  # any Piper voice
    picture="portrait.png",                      # optional; synthetic face otherwise
    size=64,                                     # 32 or 64 px
    face_crop=True,                              # center-crop to the detected face
    face_crop_margin=0.35,                       # margin around the face box
)
engine.render("Hello! <happy> Great news. <sad> It is over.", "clip.mp4")
```

## How it works

```
text ──▶ marker parser ──▶ Piper TTS ──▶ audio analysis ──▶ viseme map ──▶ wgpu render ──▶ H.264+AAC ──▶ MP4
        (<happy> …)       (phonemes,     (RMS → head bob,  (phoneme →     (expression    (openh264,
                           timings)       breathing)       mouth shape)    atlas + fade)  FDK-AAC, mp4 crate)
```

| Stage | Crate / module | What it does |
|-------|----------------|--------------|
| Markers | `avatar-core::markers` | Splits text into `(segment, mode)` spans; unknown tags dropped |
| TTS | `avatar-core::tts` | Embedded Piper (`piper-rs`): PCM + phoneme timings per segment |
| Analysis | `avatar-core::analysis` | 40 ms RMS frames → head bob, breathing, lip-sync features |
| Visemes | `avatar-core::visemes` | Phoneme → 12-viseme weights (AI, E, O, MBP, FV, L, …) |
| Atlas | `avatar-core::atlas` | 7 expression bands (neutral, happy, mad, sad, surprised, disgusted, fearful) warped from one picture |
| Render | `avatar-core::renderer` | wgpu compute shader; 32 or 64 px frames @ 25 fps |
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

let cfg = EngineConfig::voice(".res/en_US-hfc_female-medium.onnx")
    .with_picture("portrait.png")
    .with_size(64)
    .with_strength(0.5);          // photo ~0.5, cartoon ~1.0

let (frames, elapsed) = render_with_stats("Hello <happy>
… [truncated, 4058 chars total] …
hon
import live_avatar

live_avatar.version()             # crate version
engine = live_avatar.Engine(
    voice_model="voices/en_US-amy-medium.onnx",
    picture=None,                 # optional PNG/JPEG
    size=64,                      # 32 | 64
    strength=0.5,                 # warp strength
    breath_period_secs=4.0,       # breathing cycle
    face_crop=True,               # center-crop to the detected face
    face_crop_margin=0.35,        # margin around the face box
)
frames = engine.render("Hello <happy> world", "out.mp4")  # GIL released
```

Markers: `<neutral> <happy> <mad> <angry> <sad> <surprised> <disgusted>
<fearful>` — unknown tags are ignored, text before any marker is neutral.

## Example

See `examples/render_clip.py` for a runnable script that renders a short
marked-up clip with the bundled test voice.

> **Input pictures:** the pipeline detects the face (SeetaFace via `rustface`,
> plan 4) and center-crops the photo with a 35% margin before building the
> expression atlas, so an off-center or small face is centered in the output.
> Face-crop can be disabled with `face_crop=False` (or `with_face_crop(false)`
> in Rust); the margin is configurable via `face_crop_margin`. If the model is
> missing or no face is found, the whole picture is used — rendering never
> fails. The cropped region is then center-cropped to a square (the renderer
> requires a square atlas band, so non-square photos — e.g. wide or tall
> images — are handled by cropping the longer axis centered) and downscaled
> to at most 128 px (longest side) before warping. The output is always 32 or
> 64 px regardless of the input size. A picture that is mostly one color (e.g.
> a very light photo) will produce a correspondingly flat-looking avatar.

## Plans

- `plans/PLAN_AVATAR_MODULE.md` — this module (tasks 1–11)
- `plans/PLAN_LIPSYNC_TRAINING.md` — learned lip-sync model (`lipsync-train`
  placeholder crate) that will replace the deterministic viseme mapping
- `plans/PLAN_FIX_RENDER_ISSUES.md` — fixes silent audio, flicker, and input
  picture downscaling

## License

MIT — see [LICENSE](LICENSE).