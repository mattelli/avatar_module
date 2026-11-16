#!/usr/bin/env python3
"""Converts Wav2Lip + ExpNet + PFLD PyTorch checkpoints to ONNX for Burn import.

Plan 5, task 2 — weight conversion pipeline.
Plan 7, task 1 — PFLD landmark model conversion (expression control points).

Downloads/uses:
  .res/wav2lip_gan.pth          (Wav2Lip GAN generator, 435 MB)
  .res/auido2exp_00300-model.pth (SadTalker ExpNet, 34 MB)
  .res/pfld_checkpoint.pth.tar  (PFLD 98-landmark detector, 7 MB)

Produces (into .res/):
  wav2lip_gan.onnx     — the Wav2Lip generator (audio + face -> mouth patch)
  expnet.onnx          — the ExpNet audio->expression MLP
  pfld.onnx            — the PFLD 98-landmark detector (112x112 -> 98 (x,y))

These .onnx files are then imported by burn-onnx (ModelGen in the wav2lip /
expnet crate build scripts) into .burnpack records, which the Rust runtime
loads. This script also prints SHA-256 checksums for reproducibility.

Only the generator / ExpNet / PFLD-backbone are exported — no discriminator,
no training heads. All run in inference (eval) mode.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / ".res"


# ---------------------------------------------------------------------------
# Wav2Lip generator (Rudrabha/Wav2Lip, models/wav2lip.py + models/conv.py)
# ---------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Wav2Lip(nn.Module):
    """Generator only (no discriminator). Input: (B, 1, 80, 16) audio mel +
    (B, 6, 96, 96) face (lower half). Output: (B, 3, 96, 96) mouth patch."""

    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96,96
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1,1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences: (B, 1, 80, 16), face_sequences: (B, 6, 96, 96)
        audio_embedding = self.audio_encoder(audio_sequences)  # B,512,1,1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            x = torch.cat((x, feats[-1]), dim=1)
            feats.pop()

        x = self.output_block(x)
        return x


# ---------------------------------------------------------------------------
# ExpNet (SadTalker src/audio2exp_models/networks.py, SimpleWrapperV2)
# ---------------------------------------------------------------------------
class ExpConv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        if self.use_act:
            return self.act(out)
        return out


class SimpleWrapperV2(nn.Module):
    """ExpNet: (B,1,80,16) mel + (B,64) ref + (B,1) ratio -> (B,64) exp coeffs."""

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            ExpConv2d(1, 32, kernel_size=3, stride=1, padding=1),
            ExpConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            ExpConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(64, 128, kernel_size=3, stride=3, padding=1),
            ExpConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            ExpConv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ExpConv2d(256, 512, kernel_size=3, stride=1, padding=0),
            ExpConv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.mapping1 = nn.Linear(512 + 64 + 1, 64)

    def forward(self, x, ref, ratio):
        # Use static reshape ([-1, 512]) instead of .view(x.size(0), -1) so
        # the exported ONNX graph avoids a Shape-slice node that burn-onnx's
        # formatter cannot handle. Inference always runs with batch=1.
        x = self.audio_encoder(x).reshape(-1, 512)  # B,512
        ref_reshape = ref.reshape(-1, 64)
        ratio = ratio.reshape(-1, 1)
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1))
        return y  # B,64


# ---------------------------------------------------------------------------
# PFLD (Practical Facial Landmark Detector, polarisZhao/PFLD-pytorch, WFLW 98)
# ---------------------------------------------------------------------------
# Plan 7, task 1 — 98-landmark detector used to place the expression-warp
# control points on the actual face (replacing the hardcoded canonical points).
# Input: (B, 3, 112, 112) RGB in 0..1. Output: (B, 196) = 98 * (x, y), each
# normalized to 0..1. The backbone also returns a feature map (out1) used only
# by the training-time auxiliary pose head; inference needs only `landmarks`.
def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class PFLDInference(nn.Module):
    """PFLD backbone: (B,3,112,112) -> (out1, landmarks (B,196))."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)
        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)

        self.conv7 = conv_bn(16, 32, 3, 2)
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)
        # `bn8` is declared upstream but unused in `forward` (a dead parameter
        # in the released checkpoint); kept so `load_state_dict` accepts it.
        self.bn8 = nn.BatchNorm2d(128)
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)

        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: (B, 3, 112, 112)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)

        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return out1, landmarks


def load_pfld_state(model: nn.Module, ckpt: dict) -> None:
    """Loads the PFLD backbone state_dict from the checkpoint.

    The checkpoint stores the backbone under `pfld_backbone` (the auxiliary
    pose head under `auxiliarynet`, which we do not export).
    """
    model.load_state_dict(ckpt["pfld_backbone"])


def export_pfld(model: nn.Module, out: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input_1"],
        output_names=["out1", "landmarks"],
        opset_version=17,
    )


# ---------------------------------------------------------------------------
# Checksum helper
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wav2lip_state(model: nn.Module, ckpt: dict) -> None:
    """Loads the generator state_dict from the Wav2Lip GAN checkpoint."""
    state = ckpt["state_dict"]
    # The checkpoint keys are prefixed with 'module.' (DataParallel). Strip.
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)


def load_expnet_state(model: nn.Module, ckpt: dict) -> None:
    """Loads the ExpNet state_dict from the SadTalker checkpoint."""
    state = ckpt["model"]
    model.load_state_dict(state, strict=False)


def export_wav2lip(model: nn.Module, out: Path) -> None:
    model.eval()
    audio = torch.randn(1, 1, 80, 16)
    face = torch.randn(1, 6, 96, 96)
    torch.onnx.export(
        model,
        (audio, face),
        str(out),
        input_names=["audio_sequences", "face_sequences"],
        output_names=["mouth"],
        dynamic_axes={
            "audio_sequences": {0: "batch"},
            "face_sequences": {0: "batch"},
            "mouth": {0: "batch"},
        },
        opset_version=17,
    )


def export_expnet(model: nn.Module, out: Path) -> None:
    model.eval()
    x = torch.randn(1, 1, 80, 16)
    ref = torch.randn(1, 64)
    ratio = torch.randn(1, 1)
    torch.onnx.export(
        model,
        (x, ref, ratio),
        str(out),
        # NOTE: the input is named `ref_coeff`, not `ref` — `ref` is a Rust
        # reserved keyword and burn-onnx would generate invalid Rust code
        # (`let ... [.., ref, ..]`).
        input_names=["mel", "ref_coeff", "ratio"],
        output_names=["exp_coeff"],
        dynamic_axes={
            "mel": {0: "batch"},
            "ref_coeff": {0: "batch"},
            "ratio": {0: "batch"},
            "exp_coeff": {0: "batch"},
        },
        opset_version=17,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=Path, default=RES, help="resource dir (default .res)")
    ap.add_argument("--wav2lip", type=Path, default=None, help="wav2lip_gan.pth path")
    ap.add_argument("--expnet", type=Path, default=None, help="auido2exp_00300-model.pth path")
    ap.add_argument("--pfld", type=Path, default=None, help="pfld_checkpoint.pth.tar path")
    ap.add_argument("--skip-wav2lip", action="store_true", help="skip wav2lip export")
    ap.add_argument("--skip-expnet", action="store_true", help="skip expnet export")
    ap.add_argument("--skip-pfld", action="store_true", help="skip pfld export")
    args = ap.parse_args()

    res = args.res
    res.mkdir(parents=True, exist_ok=True)
    wav2lip_ckpt = args.wav2lip or (res / "wav2lip_gan.pth")
    expnet_ckpt = args.expnet or (res / "auido2exp_00300-model.pth")
    pfld_ckpt = args.pfld or (res / "pfld_checkpoint.pth.tar")

    results = {}

    if not args.skip_wav2lip:
        if not wav2lip_ckpt.exists():
            print(f"missing {wav2lip_ckpt}", file=sys.stderr)
            return 1
        print(f"loading Wav2Lip from {wav2lip_ckpt}")
        ckpt = torch.load(wav2lip_ckpt, map_location="cpu", weights_only=False)
        model = Wav2Lip()
        load_wav2lip_state(model, ckpt)
        out = res / "wav2lip_gan.onnx"
        export_wav2lip(model, out)
        print(f"wrote {out} ({out.stat().st_size} bytes)")
        results["wav2lip_gan.onnx"] = sha256(out)

    if not args.skip_expnet:
        if not expnet_ckpt.exists():
            print(f"missing {expnet_ckpt}", file=sys.stderr)
            return 1
        print(f"loading ExpNet from {expnet_ckpt}")
        ckpt = torch.load(expnet_ckpt, map_location="cpu", weights_only=False)
        model = SimpleWrapperV2()
        load_expnet_state(model, ckpt)
        out = res / "expnet.onnx"
        export_expnet(model, out)
        print(f"wrote {out} ({out.stat().st_size} bytes)")
        results["expnet.onnx"] = sha256(out)

    if not args.skip_pfld:
        if not pfld_ckpt.exists():
            print(f"missing {pfld_ckpt}", file=sys.stderr)
            return 1
        print(f"loading PFLD from {pfld_ckpt}")
        ckpt = torch.load(pfld_ckpt, map_location="cpu", weights_only=False)
        model = PFLDInference()
        load_pfld_state(model, ckpt)
        out = res / "pfld.onnx"
        export_pfld(model, out)
        print(f"wrote {out} ({out.stat().st_size} bytes)")
        results["pfld.onnx"] = sha256(out)

    print("\nSHA-256 checksums:")
    for name, h in results.items():
        print(f"  {h}  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
