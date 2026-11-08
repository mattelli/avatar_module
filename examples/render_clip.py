"""Renders a short marked-up clip with the bundled test voice.

Usage (from the repo root, wheel installed):
    .venv/Scripts/python.exe examples/render_clip.py [output.mp4]
"""

import sys
from pathlib import Path

import live_avatar

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".res" / "en_US-hfc_female-medium.onnx"
PICTURE = ROOT / ".res" / "avatar.jpg"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "clip.mp4"

    engine = live_avatar.Engine(
        voice_model=str(MODEL),
        picture=str(PICTURE),
        size=64,
        strength=0.5,
    )
    text = "Hello! <happy> Great news. <sad> It is over."
    frames = engine.render(text, str(out))
    print(f"wrote {frames} frames to {out}")


if __name__ == "__main__":
    main()
