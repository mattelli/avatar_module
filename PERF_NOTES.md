# Performance notes (Plan 5, task 10; updated Plan 6, task 7)

Measurements on the dev machine (Windows 11, AMD, burn-wgpu via DX12/Vulkan,
burn-flex CPU fallback), debug build.

## Benchmark

`cargo run -p avatar-core --example talking_head_bench`

Renders a 440 Hz sine clip through the full talking-head path (resample →
ExpNet → Wav2Lip → Audio2Pose → fusion → N×N frames).

| Backend | Audio | Frames | Total | s/frame |
|---------|-------|--------|-------|---------|
| GPU (burn-wgpu) | 1.0 s | 25 | 15.4–17.2 s | ~0.62–0.69 |
| CPU (burn-flex) | 0.2 s | 5 | 36.8–37.8 s | ~7.4–7.6 |

**GPU is ~12× faster than CPU.**

## Batching

`expnet::ExpNet::predict`, `wav2lip::Wav2Lip::predict`, and the Wav2Lip audio
encoder (`Wav2Lip::audio_embeddings`, used by Audio2Pose) process frames in
batches:

- ExpNet default batch **32** — an MLP/conv audio encoder; negligible memory.
- Wav2Lip default batch **8** — a deep conv GAN; bounded to avoid GPU memory
  pressure (batch 16 occasionally triggered a transient "device lost" on
  first GPU init).
- Wav2Lip audio-encoder batch **8** (Plan 6) — same conv trunk, same bound.

Batching amortizes the per-call launch + tensor-copy overhead; on CPU it
roughly halved Wav2Lip's per-frame time (16 s → 7.5 s single→batched).

## Where the time goes (GPU path)

The per-frame GPU time (~0.62 s) is *not* the conv forward — it is dominated
by:

1. **Fusion on the CPU** — `fusion::fuse_frame` runs an N×N inverse
   displacement warp (`atlas::warp_cpu`) + pose affine warp + elliptical mask
   blend + resize per frame, all single-threaded. This is the biggest lever.
2. **GPU→CPU sync per batch** — `Tensor::into_data()` reads results back;
   larger batches reduce the number of syncs.

## Tuning options (future)

- **Parallelize / vectorize the fusion warp** (rayon, or move the warp to the
  GPU via the existing wgpu compute path).
- **Lower-res fusion intermediate + upscale**: fuse at 128×128 then resize to
  the output size (the mask is smooth; upscaling is cheap). ~4× fewer warp
  pixels.
- **Release build**: all numbers above are `debug`; a `--release` build is
  typically 2–4× faster.
- **Frame batching** is already in place; tune `predict_batched` batch sizes
  via the model crates if needed.

## Reproduction

```bash
# Full talking-head render (GPU auto-selected), 256×256 MP4:
python examples/render_clip.py

# Benchmark (GPU + CPU):
cargo run -p avatar-core --example talking_head_bench
```
