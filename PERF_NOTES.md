# Performance notes (Plan 5, task 10; updated Plan 6, task 7; Plan 7, task 6;
# pose warp removed)

Measurements on the dev machine (Windows 11, AMD **integrated** GPU,
burn-wgpu via DX12/Vulkan, burn-flex CPU fallback).

## Backend choice: CPU by default (Plan 7 follow-up)

`BackendPreference::Auto` selects the **CPU**. These models are small and
burn-wgpu's per-op kernel-launch + host↔device sync overhead dominates, so on
an integrated GPU the GPU path is **slower** than CPU. Measured (release build,
3 s clip → 75 frames at 256×256, `avatar.jpg`):

| Backend | Full render | Wav2Lip / chunk | PFLD / call |
|---------|-------------|-----------------|-------------|
| **CPU (burn-flex, Auto)** | **14.3 s** | **153 ms** | **14 ms** |
| GPU (burn-wgpu, integrated) | 54.2 s | 593 ms | 97 ms |

**CPU is ~3.8× faster end-to-end** (and ~3.9× / ~6.8× per model). On a
**discrete** GPU the GPU path may win, so it stays available via
`backend="gpu"` (Python) / `BackendPref::Gpu` (Rust).

## Benchmark

`cargo run -p avatar-core --example talking_head_bench`

Renders a 440 Hz sine clip through the full talking-head path (resample →
ExpNet → Wav2Lip → PFLD expression warp → in-paint → resize → optional
whole-frame Gaussian blur → N×N frames).

| Backend | Audio | Frames | Total | s/frame |
|---------|-------|--------|-------|---------|
| GPU (burn-wgpu) | 1.0 s | 25 | 15.4–17.2 s | ~0.62–0.69 |
| CPU (burn-flex) | 0.2 s | 5 | 36.8–37.8 s | ~7.4–7.6 |

> These early numbers were **debug**-build and predate the Plan 7 reorder; the
> release-build comparison above supersedes them (CPU now wins).

## Batching

`expnet::ExpNet::predict` and `wav2lip::Wav2Lip::predict` process frames in
batches:

- ExpNet default batch **32** — an MLP/conv audio encoder; negligible memory.
- Wav2Lip default batch **8** — a deep conv GAN; bounded to avoid GPU memory
  pressure (batch 16 occasionally triggered a transient "device lost" on
  first GPU init).

Batching amortizes the per-call launch + tensor-copy overhead; on CPU it
roughly halved Wav2Lip's per-frame time (16 s → 7.5 s single→batched).

## Where the time goes (GPU path)

The per-frame GPU time (~0.62 s) is *not* the conv forward — it is dominated
by:

1. **CPU warps / in-paint** — per frame the pipeline runs the PFLD expression
   warp (`atlas::warp_cpu`, 112×112) and the in-paint paste into the N×N square,
   all single-threaded. This is the biggest lever.
2. **GPU→CPU sync per batch** — `Tensor::into_data()` reads results back;
   larger batches reduce the number of syncs.

## Tuning options (future)

- **Parallelize / vectorize the warps** (rayon, or move them to the GPU via the
  existing wgpu compute path).
- **Cap N** (e.g. `min(N, 512)`): the in-paint runs at the source
  square size (740 for `avatar.jpg`); a smaller working square cuts the paste
  pixels ~2×.
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
