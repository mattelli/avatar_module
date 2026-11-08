# Plan 2 — Lip-Sync Model training pipeline (Burn, trained on burn-wgpu)

**References:** `AVATAR_MODULE_REPORT.md` (v3) §3.4 Path A (pure Rust, Burn on `burn-wgpu`),
§3.5 (datasets). Companion plan: `PLAN_AVATAR_MODULE.md` — its task 5 (deterministic
phoneme→viseme mapping) is the baseline this plan replaces; its task 7 (wgpu renderer) consumes
the trained model's output.

**Decisions applied:** Q3=local training on this machine (AMD GPU, burn-wgpu via DX12/Vulkan);
datasets downloaded manually by the user from:
- GRID: https://spandh.dcs.shef.ac.uk/gridcorpus (25 kHz endpointed audio, video normal)
- TIMIT: https://huggingface.co/datasets/kimetsu/Timit

## Goals

- **G1** — Dataset preparation pipeline in Rust: parse GRID + TIMIT (audio + video + transcripts),
  extract mouth-region crops per frame, compute audio features at 25 fps, derive viseme labels
  via forced alignment → `(audio features → viseme weights)` training pairs.
- **G2** — Burn model definition: compact audio→viseme network (8-frame sliding window input,
  2–3 dense layers or 1D conv, K = 8–12 viseme weights, sigmoid/softmax output).
- **G3** — Training loop with Burn on the **`burn-wgpu`** backend (DX12/Vulkan on AMD GPU);
  train/val split, loss (cross-entropy / MSE on viseme weights), checkpointing.
- **G4** — Evaluation: viseme accuracy on held-out speakers; qualitative check by rendering
  sample clips through the Plan 1 renderer.
- **G5** — Export trained weights (Burn `.mpk` record) loadable by `avatar-core` runtime
  inference on `burn-wgpu`; drop-in replacement for Plan 1 task 5.
- **G6** — Reproducibility: config file (dataset paths, hyperparameters), seeded RNG,
  training log.

## Current behavior

- No training pipeline exists. `lipsync-train` is a placeholder crate in the workspace created
  by Plan 1 task 1. The runtime lip-sync is deterministic (Plan 1 task 5).

## Expected behavior

- `lipsync-train` crate with binaries: `prepare-dataset` (G1), `train` (G2–G3), `evaluate` (G4).
- Training runs locally on the AMD GPU via burn-wgpu; checkpoints written per epoch.
- Final artifact: `visemes.burn` (weights) + config, consumed by `avatar-core` (Plan 1 §3.8
  `lip_sync_model="visemes.burn"`).
- Held-out speaker viseme accuracy ≥ baseline (deterministic mapping) — target: clearly better
  mouth articulation on rendered samples.

## Tasks (one commit per task; simple tasks)

| # | Task | Commit message | Ref |
|---|---|---|---|
| 1 | Dataset acquisition guide: README with download instructions (GRID from spandh.dcs.shef.ac.uk, TIMIT from huggingface.co/datasets/kimetsu/Timit), expected directory layout, checksums | `task1: dataset guide` | G1, §3.5 |
| 2 | GRID parser: read 25 kHz endpointed WAV + video + transcript files; frame extraction; unit tests on a small sample | `task2: grid parser` | G1, §3.5 |
| 3 | TIMIT parser: read HuggingFace TIMIT (audio + phonetic transcriptions); unit tests on a small sample | `task3: timit parser` | G1, §3.5 |
| 4 | Mouth-region extraction: face/mouth landmark detection per frame, mouth crop + normalization (32/64 px); unit tests on synthetic face | `task4: mouth crops` | G1, §3.5 |
| 5 | Viseme labeling: forced alignment (phoneme timings → viseme sequence), per-frame viseme weights; unit tests with known phoneme sequences | `task5: viseme labels` | G1, §3.5 |
| 6 | Audio feature extraction: mel-spectrogram / band-energy features at 25 fps, 8-frame sliding windows; unit tests | `task6: audio features` | G1, §3.4 |
| 7 | Dataset serialization: `(features → viseme weights)` pairs to disk (Burn dataset format), train/val split by speaker; unit tests | `task7: dataset serialize` | G1, G6 |
| 8 | Burn model definition: compact network (8-frame window → 2–3 dense layers / 1D conv → K viseme weights); unit tests on random tensors | `task8: burn model` | G2, §3.4 |
| 9 | Training loop on burn-wgpu: loss, optimizer, epochs, checkpointing, seeded RNG, config file; smoke test on small subset | `task9: training loop` | G3, G6 |
| 10 | Full training run on GRID+TIMIT; log curves; select best checkpoint | `task10: train run` | G3, G4 |
| 11 | Evaluation: held-out speaker viseme accuracy; render sample clips through Plan 1 renderer for qualitative check | `task11: evaluate` | G4 |
| 12 | Export + integration: save `visemes.burn`; wire into `avatar-core` as drop-in replacement for Plan 1 task 5; end-to-end test | `task12: export integrate` | G5 |
| 13 | Documentation: training README, hyperparameters, reproduction steps; tag `v0.1.0` | `task13: docs v0.1.0` | G6 |

## Unit-test verification

- **T2** — GRID parser: sample file → correct frame count, sample rate 25 kHz, transcript matches.
- **T3** — TIMIT parser: sample record → audio + phoneme list parsed; phoneme count > 0.
- **T4** — mouth crops: synthetic face image → mouth crop extracted at expected location; crop
  size normalized to 32/64 px.
- **T5** — viseme labels: phoneme sequence `M A P` → closed-open-closed weight pattern; unknown
  phoneme → neutral.
- **T6** — features: synthetic WAV → feature frames at 25 fps; sliding window shape correct.
- **T7** — serialization: written dataset round-trips; speaker-disjoint split (no speaker in
  both train and val).
- **T8** — model: forward pass on random input → output shape (K viseme weights), values in
  [0, 1]; backward pass produces finite gradients.
- **T9** — training loop: smoke test on small subset → loss decreases over a few epochs;
  checkpoint file written and reloadable.
- **T11** — evaluation: accuracy metric computed on held-out speakers; ≥ deterministic baseline.
- **T12** — integration: `avatar-core` loads `visemes.burn` and runs inference on burn-wgpu;
  end-to-end render with learned model produces MP4 (Plan 1 T9 criteria).

## Cross-references

- Implements report: `AVATAR_MODULE_REPORT.md` §3.4 (Path A), §3.5 (datasets).
- Companion plan: `PLAN_AVATAR_MODULE.md` — provides the workspace (its task 1), the renderer
  that consumes the model (its task 7), and the deterministic baseline (its task 5) that this
  plan's trained model replaces (its task 12 wires the replacement).
