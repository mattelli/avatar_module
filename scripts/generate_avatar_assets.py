#!/usr/bin/env python3
"""Generate the 28 avatar I2I assets (expressions, lip shapes, head poses, eyes).

Reads a single avatar reference image and uses Stable Diffusion 1.5 + IP-Adapter
(face) to generate a consistent set of squared PNGs in `avatar_assets/`.

Usage:
    python scripts/generate_avatar_assets.py <reference_image> [--out DIR] [--device cpu|cuda] [--steps N] [--strength F] [--scale F] [--seed N]

Requirements:
    pip install diffusers transformers accelerate torch pillow

See scripts/README_AVATAR_ASSETS.md for full install + usage instructions.
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- prompts ----

REFERENCE_PROMPT_TEMPLATE = (
    "portrait photo of {subject}, front-facing, neutral expression, "
    "mouth closed, looking at camera, even lighting, sharp focus, photorealistic"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, asymmetric eyes, extra limbs, "
    "deformed hands, text, watermark, multiple faces, identity change, different person"
)

# (filename, prompt-suffix) — suffix is appended to the reference prompt.
EXPRESSIONS = [
    ("expr_neutral.png", "neutral expression, relaxed face, mouth closed, looking at camera"),
    ("expr_happy.png", "big genuine smile, eyes slightly squinting, joyful expression"),
    ("expr_mad.png", "angry expression, furrowed brows, tense jaw, narrowed eyes"),
    ("expr_sad.png", "sad expression, downturned mouth, drooping eyelids, somber look"),
    ("expr_surprised.png", "surprised expression, raised eyebrows, wide eyes, mouth slightly open"),
    ("expr_disgusted.png", "disgusted expression, wrinkled nose, raised upper lip, narrowed eyes"),
    ("expr_fearful.png", "fearful expression, raised inner brows, wide eyes, tense mouth"),
]

LIP_SHAPES = [
    ("lip_sil.png", "mouth closed, lips gently together, neutral expression"),
    ("lip_AI.png", 'mouth wide open, jaw dropped, saying "ah" as in "father"'),
    ("lip_E.png", 'mouth moderately open, lips spread, saying "eh" as in "bed"'),
    ("lip_O.png", 'lips rounded and pushed forward, saying "oh" as in "go"'),
    ("lip_MBP.png", 'lips pressed firmly together, closed, as in "m", "b", "p"'),
    ("lip_FV.png", 'upper teeth lightly touching lower lip, as in "f" or "v"'),
    ("lip_L.png", 'tongue tip near upper front teeth, mouth slightly open, as in "l"'),
    ("lip_TH.png", 'tongue tip between teeth, mouth slightly open, as in "th"'),
    ("lip_U.png", 'lips in a small tight circle, saying "oo" as in "boot"'),
    ("lip_WQ.png", 'lips in a small rounded pucker, as in "w" or "wh"'),
    ("lip_etc.png", "mouth slightly open, neutral mid-position, generic consonant"),
    ("lip_rest.png", "relaxed mouth, neutral closed position, resting face"),
]

HEAD_POSES = [
    ("head_center.png", "head perfectly centered, facing camera directly"),
    ("head_left.png", "head turned slightly to the left, still looking at camera"),
    ("head_right.png", "head turned slightly to the right, still looking at camera"),
    ("head_up.png", "head tilted slightly upward, chin raised, looking slightly up"),
    ("head_down.png", "head tilted slightly downward, chin lowered, looking slightly down"),
]

EYES = [
    ("eye_open.png", "eyes fully open, looking at camera, neutral expression"),
    ("eye_blink.png", "eyes closed, eyelids shut, neutral expression"),
    ("eye_squint.png", "eyes slightly squinting, relaxed lids, neutral expression"),
    ("eye_wide.png", "eyes wide open, raised lids, alert expression"),
]

ALL_ASSETS = EXPRESSIONS + LIP_SHAPES + HEAD_POSES + EYES  # 28 total

# ------------------------------------------------------------- generation ----


def make_pipeline(device: str, torch_dtype):
    """Builds an SD1.5 img2img pipeline with IP-Adapter (face) loaded."""
    from diffusers import AutoPipelineForImage2Image
    from transformers import CLIPVisionModelWithProjection

    # Load a CLIP vision encoder for IP-Adapter (recommended for face consistency).
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch_dtype,
    )

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )

    # IP-Adapter Face: keeps the same person across all generations.
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus-face_sd15.bin",
    )
    pipe.set_ip_adapter_scale(0.8)

    pipe = pipe.to(device)
    # NOTE: do NOT call enable_attention_slicing() / enable_xformers... here.
    # IP-Adapter installs its own attention processors; memory-efficient
    # attention (slicing / xformers / SDPA) overrides them and breaks with
    # "AttributeError: 'tuple' object has no attribute 'shape'".
    return pipe


def generate_one(pipe, prompt: str, init_image, ip_image, args, generator) -> None:
    """Runs one img2img generation and returns the PIL image."""
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=init_image,          # the reference (identity anchor)
        ip_adapter_image=ip_image, # extra identity conditioning
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    return result.images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="avatar reference image (PNG/JPEG)")
    parser.add_argument("--out", type=Path, default=Path("avatar_assets"), help="output dir (default avatar_assets)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", help="device (default: cuda if available, else cpu)")
    parser.add_argument("--steps", type=int, default=30, help="diffusion steps (default 30)")
    parser.add_argument("--strength", type=float, default=0.4, help="denoising strength 0-1 (default 0.4)")
    parser.add_argument("--scale", type=float, default=0.8, help="IP-Adapter scale (default 0.8)")
    parser.add_argument("--guidance", type=float, default=7.5, help="guidance scale (default 7.5)")
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for reproducibility (default 42)")
    parser.add_argument("--subject", default=None, help="subject description; if omitted, prompted")
    args = parser.parse_args()

    print("=" * 60)
    print("Avatar asset generation — parameters")
    print("=" * 60)
    print(f"  reference image : {args.reference}")
    print(f"  output dir      : {args.out}")
    print(f"  device          : {args.device}")
    print(f"  steps           : {args.steps}")
    print(f"  strength        : {args.strength}")
    print(f"  ip_adapter_scale: {args.scale}")
    print(f"  guidance_scale  : {args.guidance}")
    print(f"  seed            : {args.seed}")
    print(f"  subject         : {args.subject if args.subject is not None else '(prompted below)'}")
    print("=" * 60)

    if not args.reference.exists():
        sys.exit(f"reference image not found: {args.reference}")

    import torch
    from PIL import Image

    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32
    device = args.device

    # Subject description (identity prefix).
    subject = args.subject
    if subject is None:
        subject = input("Describe the subject (e.g. 'a young woman with brown hair', 'a cartoon robot'): ").strip()
    if not subject:
        sys.exit("no subject description given")
    ref_prompt = REFERENCE_PROMPT_TEMPLATE.format(subject=subject)
    print(f"  reference prompt: {ref_prompt}")
    print("=" * 60)

    # Load reference image, center-crop to square, resize to 256x256.
    init_image = Image.open(args.reference).convert("RGB")
    w, h = init_image.size
    side = min(w, h)
    init_image = init_image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    ip_image = init_image  # same image used as IP-Adapter reference

    print(f"building pipeline on {device} (this downloads ~4GB on first run)...")
    pipe = make_pipeline(device, torch_dtype)
    pipe.set_ip_adapter_scale(args.scale)

    args.out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"generating {len(ALL_ASSETS)} assets into {args.out} ...")
    for i, (fname, suffix) in enumerate(ALL_ASSETS, 1):
        prompt = f"{ref_prompt}, {suffix}"
        img = generate_one(pipe, prompt, init_image, ip_image, args, generator)
        out_path = args.out / fname
        img.save(out_path)
        print(f"  [{i}/{len(ALL_ASSETS)}] {fname}")

    print(f"done. {len(ALL_ASSETS)} assets written to {args.out}")


if __name__ == "__main__":
    main()