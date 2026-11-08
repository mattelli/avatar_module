"""T10 check: imports live_avatar, renders MP4s, verifies GIL release.

Run from the repo root after installing the wheel:
    .venv/Scripts/python.exe crates/avatar-py/tests/t10_check.py
"""
import tempfile
import threading
import time
from pathlib import Path

import live_avatar

MODEL = Path(__file__).resolve().parents[3] / ".res" / "en_US-hfc_female-medium.onnx"


def main() -> None:
    print("version:", live_avatar.version())
    engine = live_avatar.Engine(voice_model=str(MODEL), size=32)
    print("engine created, size =", engine.size)

    # 1. Render produces an MP4.
    out = Path(tempfile.gettempdir()) / "avatar_t10_py.mp4"
    if out.exists():
        out.unlink()
    t0 = time.time()
    frames = engine.render("Hello <happy> world", out=str(out))
    dt = time.time() - t0
    print(f"rendered {frames} frames in {dt:.1f}s")
    assert out.exists(), "mp4 not created"
    size = out.stat().st_size
    print("mp4 size:", size)
    assert size > 1000, "mp4 too small"

    # 2. GIL release: a pure-Python thread keeps running during render.
    done = threading.Event()
    ticks = [0]

    def spinner():
        while not done.is_set():
            ticks[0] += 1
            time.sleep(0.005)

    th = threading.Thread(target=spinner)
    t1 = time.time()
    th.start()
    engine.render("GIL <happy> test", out=str(out))
    done.set()
    th.join()
    print(f"main-thread ticks during render: {ticks[0]} ({time.time() - t1:.1f}s)")
    assert ticks[0] > 10, "GIL was NOT released during render"

    # 3. Two parallel renders (real concurrency).
    results = {}

    def worker(i):
        e = live_avatar.Engine(voice_model=str(MODEL), size=32)
        o = Path(tempfile.gettempdir()) / f"avatar_t10_{i}.mp4"
        e.render("Thread <sad> test", out=str(o))
        results[i] = o.stat().st_size

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    t2 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"2 parallel renders took {time.time() - t2:.1f}s, sizes: {results}")
    assert all(s > 1000 for s in results.values()), "parallel render failed"

    for i in range(2):
        p = Path(tempfile.gettempdir()) / f"avatar_t10_{i}.mp4"
        if p.exists():
            p.unlink()
    if out.exists():
        out.unlink()
    print("T10 OK")


if __name__ == "__main__":
    main()
