#!/usr/bin/env python3
"""Verifies the exported ONNX models match the PyTorch originals (plan T2).

Runs the PyTorch model and the ONNX model (via onnxruntime) on the same
random inputs and checks the outputs are close.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Make convert_weights importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_weights import (
    Wav2Lip,
    SimpleWrapperV2,
    load_wav2lip_state,
    load_expnet_state,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / ".res"


def verify_wav2lip():
    ckpt = torch.load(RES / "wav2lip_gan.pth", map_location="cpu", weights_only=False)
    model = Wav2Lip()
    load_wav2lip_state(model, ckpt)
    model.eval()

    audio = torch.randn(1, 1, 80, 16)
    face = torch.randn(1, 6, 96, 96)
    with torch.no_grad():
        ref = model(audio, face).numpy()

    sess = ort.InferenceSession(str(RES / "wav2lip_gan.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"audio_sequences": audio.numpy(), "face_sequences": face.numpy()})[0]

    print("wav2lip ref:", ref.shape, "onnx:", out.shape)
    print("max abs diff:", np.abs(ref - out).max())
    assert ref.shape == out.shape, f"shape mismatch {ref.shape} vs {out.shape}"
    # BatchNorm is folded into the conv weights during ONNX export, so a
    # small float32 tolerance is expected.
    assert np.abs(ref - out).max() < 1e-3, "wav2lip output mismatch"
    print("wav2lip OK")


def verify_expnet():
    ckpt = torch.load(RES / "auido2exp_00300-model.pth", map_location="cpu", weights_only=False)
    model = SimpleWrapperV2()
    load_expnet_state(model, ckpt)
    model.eval()

    mel = torch.randn(1, 1, 80, 16)
    ref = torch.randn(1, 64)
    ratio = torch.randn(1, 1)
    with torch.no_grad():
        exp_ref = model(mel, ref, ratio).numpy()

    sess = ort.InferenceSession(str(RES / "expnet.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"mel": mel.numpy(), "ref_coeff": ref.numpy(), "ratio": ratio.numpy()})[0]

    print("expnet ref:", exp_ref.shape, "onnx:", out.shape)
    print("max abs diff:", np.abs(exp_ref - out).max())
    assert exp_ref.shape == out.shape, f"shape mismatch {exp_ref.shape} vs {out.shape}"
    # BatchNorm folded into conv weights during export; float32 tolerance.
    assert np.abs(exp_ref - out).max() < 1e-3, "expnet output mismatch"
    print("expnet OK")


def main():
    verify_wav2lip()
    verify_expnet()
    print("All ONNX round-trip checks passed.")


if __name__ == "__main__":
    main()
