"""Renders a short marked-up clip with the bundled test voice and plays it
back with OpenCV (video) + sounddevice (audio), so you can see AND hear the
generated video.

Usage (from the repo root, wheel installed):
    python examples/render_clip.py [output.mp4] [--no-play] [--frame N] [--save-frame out.png]

Options:
    --no-play             render only, do not open the playback window
    --frame N             dump frame N (0-based) as PNG and exit
    --save-frame out.png  dump frame N (default 0) to the given path and exit
    --scale S             upscale displayed/dumped frames by factor S (default 8,
                          since the output is only 32, 64, or 128 px)
    --no-audio            disable audio playback (video only)
    --size N              output size in px: 32, 64, or 128 (default 64)

Dependencies (all optional, graceful fallback):
    - OpenCV (cv2): video playback + frame dumping
    - sounddevice + soundfile: audio playback
    - imageio_ffmpeg: extracts the audio track from the MP4 (bundled ffmpeg)

If cv2 is missing the script renders only. If the audio libs are missing it
plays video only and prints a hint.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import live_avatar

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".res" / "en_US-hfc_female-medium.onnx"
PICTURE = ROOT / ".res" / "avatar.jpg"


def _import(mod: str):
    """Returns the module `mod`, or None if it is not installed."""
    try:
        return __import__(mod)
    except ImportError:
        return None


def _upscale(cv2, frame, scale: int):
    """Nearest-neighbor upscale so the tiny 32/64 px frame is viewable."""
    if scale <= 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def extract_audio(imageio_ffmpeg, video: Path, wav: Path) -> bool:
    """Extracts the audio track of `video` to `wav` (PCM 16-bit).

    Returns False if the video has no audio track (ffmpeg exits non-zero).
    """
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ff, "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le", str(wav)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and wav.exists() and wav.stat().st_size > 0


def play_audio(sounddevice, soundfile, wav: Path, blocking: bool = True) -> None:
    """Plays a WAV file. `blocking=False` starts playback in the background."""
    data, sr = soundfile.read(str(wav))
    sounddevice.play(data, sr)
    if blocking:
        sounddevice.wait()


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


def play(cv2, video: Path, scale: int = 1, audio: bool = True) -> None:
    """Plays `video` in an OpenCV window (and audio via sounddevice if enabled);
    press q or ESC to quit."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video} with OpenCV")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = int(1000.0 / fps)

    # Optional audio: extract the track and start playback in the background.
    sd = _import("sounddevice")
    sf = _import("soundfile")
    ioff = _import("imageio_ffmpeg")
    wav_path = None
    if audio and sd and sf and ioff:
        wav_path = Path(tempfile.gettempdir()) / f"avatar_{video.stem}.wav"
        if extract_audio(ioff, video, wav_path):
            print(f"audio: playing {video.name} with sound ({fps:.1f} fps, x{scale}) - q/ESC quits")
            play_audio(sd, sf, wav_path, blocking=False)
        else:
            print("audio: no audio track found in the MP4 (video only)")
            wav_path = None
    else:
        missing = [m for m in ("sounddevice", "soundfile", "imageio_ffmpeg") if _import(m) is None]
        print(f"playing {video.name} (video only, {fps:.1f} fps, x{scale}) - q/ESC quits")
        if missing:
            print(f"  (audio skipped: missing {', '.join(missing)}; pip install {' '.join(missing)})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cv2.imshow("avatar", _upscale(cv2, frame, scale))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sd is not None and wav_path is not None:
            sd.stop()  # stop any still-playing audio
        if wav_path is not None:
            try:
                wav_path.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default=str(ROOT / "clip.mp4"), help="output MP4 path")
    parser.add_argument("--no-play", action="store_true", help="render only, do not play")
    parser.add_argument("--frame", type=int, default=None, help="dump this frame (0-based) as PNG")
    parser.add_argument("--save-frame", default=None, help="path for the dumped frame (default: frame_N.png)")
    parser.add_argument("--scale", type=int, default=1, help="upscale factor for display/dump (default 1)")
    parser.add_argument("--no-audio", action="store_true", help="disable audio playback")
    parser.add_argument("--size", type=int, default=128, choices=[32, 64, 128], help="output size in px (default 128)")
    args = parser.parse_args()

    out = Path(args.out)

    engine = live_avatar.Engine(
        voice_model=str(MODEL),
        picture=str(PICTURE),
        size=args.size,
        strength=0.5,
        face_crop=True,        # center-crop to the detected face
        face_crop_margin=0.35,  # margin around the face box
    )
    text = "Hello! <happy> Great news. <sad> It is over."
    frames = engine.render(text, str(out))
    print(f"wrote {frames} frames to {out}")

    cv2 = _import("cv2")
    if cv2 is None:
        print("OpenCV (cv2) is not installed; skipping playback. Install it with: pip install opencv-python")
        return

    if args.frame is not None:
        out_png = Path(args.save_frame) if args.save_frame else ROOT / f"frame_{args.frame}.png"
        dump_frame(cv2, out, args.frame, out_png, args.scale)
        return

    if not args.no_play:
        play(cv2, out, args.scale, audio=not args.no_audio)


if __name__ == "__main__":
    main()
