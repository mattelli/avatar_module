# Avatar Asset Generation — Install & Usage

This folder contains `generate_avatar_assets.py`, which generates the **28 avatar
assets** (expressions, lip shapes, head poses, eyes) from a single reference image
using Stable Diffusion 1.5 + IP-Adapter (face) for identity consistency.

The prompts and filenames match `plans/instruction.md`.

---

## 1. Install the required Python packages

**Recommended: use a dedicated virtual environment** (Python ≥ 3.10):

```bash
# Create + activate a venv (Windows)
python -m venv .venv-avatar
.venv-avatar\Scripts\activate

# Create + activate a venv (macOS / Linux)
# python3 -m venv .venv-avatar
# source .venv-avatar/bin/activate
```

Then install the packages:

```bash
python.exe -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only (no CUDA)
# or, if you have an NVIDIA GPU with CUDA:
# pip install torch
pip install diffusers transformers accelerate pillow
```

> **Note on PyTorch:** the default `pip install torch` pulls the CUDA build
> (large, ~2.5 GB). If you only have a CPU (like the AMD-integrated-GPU laptop
> this project targets), use the CPU wheel above. If you have an NVIDIA GPU,
> install the CUDA build instead.

**Optional but recommended** for faster CPU inference:

```bash
pip install sentencepiece
```

---

## 2. What gets downloaded on first run

The script downloads these models from Hugging Face on first use (several GB total):

| Model | Size | Purpose |
|---|---|---|
| `runwayml/stable-diffusion-v1-5` | ~4 GB | Base SD1.5 model (UNet + VAE + text encoder) |
| `h94/IP-Adapter` (`ip-adapter-plus-face_sd15.bin`) | ~100 MB | Face-identity adapter |
| `h94/IP-Adapter` (`models/image_encoder`) | ~350 MB | CLIP vision encoder for IP-Adapter |

> If you have a slow connection, this first run will take a while. Subsequent
> runs reuse the cached models.

---

## 3. Usage

```bash
# Basic (CPU, 128x128, all 28 assets)
python scripts/generate_avatar_assets.py path/to/avatar.jpg

# Specify output dir + device + tune
python scripts/generate_avatar_assets.py avatar.jpg --out avatar_assets --device cpu --steps 30 --strength 0.4 --scale 0.8 --seed 42

# On an NVIDIA GPU
python scripts/generate_avatar_assets.py avatar.jpg --device cuda
```

The script will:
1. Ask you to **describe the subject** (e.g. "a young woman with brown hair") unless
   you pass `--subject "..."`.
2. Center-crop the reference to square and resize to **128×128**.
3. Generate **28 PNGs** into `avatar_assets/` with the exact names from
   `plans/instruction.md` (`expr_*.png`, `lip_*.png`, `head_*.png`, `eye_*.png`).

---

## 4. Key parameters

| Flag | Default | Meaning |
|---|---|---|
| `--device` | `cuda` if available, else `cpu` | `cpu` or `cuda` |
| `--steps` | 30 | Diffusion steps (more = better, slower) |
| `--strength` | 0.4 | Denoising strength 0–1. Lower = closer to reference (identity), higher = more expression change |
| `--scale` | 0.8 | IP-Adapter scale. Higher = stronger identity adherence |
| `--guidance` | 7.5 | Prompt guidance scale |
| `--seed` | 42 | Fixed seed for reproducibility (same face across assets) |
| `--subject` | (prompted) | Subject description prefix |

**Tuning tips for identity consistency:**
- If faces drift between assets → **lower `--strength`** (e.g. 0.3) or **raise `--scale`** (e.g. 0.9).
- If expressions are too weak → **raise `--strength`** (e.g. 0.5) or lower `--scale`.
- Keep `--seed` fixed so all assets share the same noise base.

---

## 5. Expected output

```
avatar_assets/
  expr_neutral.png ... expr_fearful.png      (7)
  lip_sil.png ... lip_rest.png               (12)
  head_center.png ... head_down.png          (5)
  eye_open.png ... eye_wide.png              (4)
```

All are 128×128 PNGs. **Check them visually** — every image must show the same
person. If any drift, regenerate with lower `--strength` / higher `--scale`.

---

## 6. Troubleshooting

- **`No module named 'torch'`** → install PyTorch first (see §1).
- **Out of memory (CPU)** → reduce `--steps` or use `--strength` lower; close other apps.
- **CUDA out of memory** → use `--device cpu`, or reduce `--steps`.
- **Slow first run** → expected; models download once.
- **Faces look like different people** → lower `--strength`, raise `--scale`, keep `--seed` fixed.
- **`AttributeError: 'tuple' object has no attribute 'shape'`** → this is a known
  diffusers + IP-Adapter incompatibility with memory-efficient attention. The script
  intentionally does **not** call `enable_attention_slicing()` / `enable_xformers...()`
  because IP-Adapter installs its own attention processors that those override and
  break. If you added such a call, remove it. (See diffusers issue #9448.)