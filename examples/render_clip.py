"""Renders a short marked-up clip with the bundled test voice and plays it
back with OpenCV so you can visualize the generated video.

Usage (from the repo root, wheel installed):
    .venv/Scripts/python.exe examples/render_clip.py [output.mp4] [--no-play] [--frame N] [--save-frame out.png]

Options:
    --no-play             render only, do not open the OpenCV playback window
    --frame N             dump frame N (0-based) as PNG and exit
    --save-frame out.png  dump frame N (default 0) to the given path and exit
    --scale S             upscale displayed/dumped frames by factor S (default 1,
                          since the output is only 32 or 64 px)

Requires OpenCV (cv2) for playback / frame dumping. If cv2 is not importable,
the script falls back to rendering only and prints a hint.
"""

import argparse
import sys
from pathlib import Path

import live_avatar

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".res" / "en_US-hfc_female-medium.onnx"
PICTURE = ROOT / ".res" / "generated_image_20261026-211004.png"


def _import_cv2():
    """Returns the cv2 module, or None if OpenCV is not installed."""
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def _upscale(cv2, frame, scale: int):
    """Nearest-neighbor upscale so the tiny 32/64 px frame is viewable."""
    if scale <= 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def dump_frame(cv2, video: Path, frame_index: int, out_png: Path, scale: int = 1) -> None:
    """Writes a single frame of `video` to `out_png` as a PNG."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video} with OpenCV")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total:
        cap.release()
        raise ValueError(f"frame {frame_index} out of range (video has {total} frames)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {video}")
    frame = _upscale(cv2, frame, scale)
    cv2.imwrite(str(out_png), frame)
    print(f"wrote frame {frame_index} to {out_png} ({frame.shape[1]}x{frame.shape[0]})")


def play(cv2, video: Path, scale: int = 1) -> None:
    """Plays `video` in an OpenCV window; press q or ESC to quit."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video} with OpenCV")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = int(1000.0 / fps)
    print(f"playing {video} ({fps:.1f} fps, x{scale}) - press q or ESC to quit")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cv2.imshow("avatar", _upscale(cv2, frame, scale))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default=str(ROOT / "clip.mp4"), help="output MP4 path")
    parser.add_argument("--no-play", action="store_true", help="render only, do not play")
    parser.add_argument("--frame", type=int, default=None, help="dump this frame (0-based) as PNG")
    parser.add_argument("--save-frame", default=None, help="path for the dumped frame (default: frame_N.png)")
    parser.add_argument("--scale", type=int, default=1, help="upscale factor for display/dump (default 1)")
    args = parser.parse_args()

    out = Path(args.out)

    engine = live_avatar.Engine(
        voice_model=str(MODEL),
        picture=str(PICTURE),
        size=64,
        strength=0.5,
        face_crop=True,        # center-crop to the detected face
        face_crop_margin=0.35,  # margin around the face box
    )
    text = "Hello! <happy> Great news. <sad> It is over."
    frames = engine.render(text, str(out))
    print(f"wrote {frames} frames to {out}")

    cv2 = _import_cv2()
    if cv2 is None:
        print(
            "OpenCV (cv2) is not installed; skipping playback. "
            "Install it with: pip install opencv-python"
        )
        return

    if args.frame is not None:
        out_png = Path(args.save_frame) if args.save_frame else ROOT / f"frame_{args.frame}.png"
        dump_frame(cv2, out, args.frame, out_png, args.scale)
        return

    if not args.no_play:
        play(cv2, out, args.scale)


if __name__ == "__main__":
    main()
