#!/usr/bin/env python3
"""Reference SadTalker mel spectrogram (librosa-based), for cross-checking the
Rust `mel::melspectrogram` implementation (plan task 3, T3/T4).

Writes a sine wave's mel to `mel_ref.npy` and prints a compact fingerprint so
the Rust test can compare.
"""

import numpy as np
import librosa

# SadTalker hparams
SAMPLE_RATE = 16000
PREEMPHASIS = 0.97
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
N_MELS = 80
F_MIN = 55
F_MAX = 7600
REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100
MAX_ABS_VALUE = 4.0


def preemphasis(wav, k=PREEMPHASIS):
    return librosa.effects.preemphasis(wav, coef=k)


def melspectrogram(wav):
    D = librosa.stft(
        preemphasis(wav),
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
        center=True,
        pad_mode="reflect",
    )
    S = librosa.feature.melspectrogram(
        S=np.abs(D),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        htk=False,
        norm="slaney",
    )
    # _amp_to_db
    min_level = np.exp(MIN_LEVEL_DB / 20 * np.log(10))
    S_db = 20 * np.log10(np.maximum(min_level, S)) - REF_LEVEL_DB
    # _normalize (symmetric, clipped)
    S_norm = 2 * MAX_ABS_VALUE * ((S_db - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE
    S_norm = np.clip(S_norm, -MAX_ABS_VALUE, MAX_ABS_VALUE)
    return S_norm  # (n_mels, n_frames)


def main():
    n = int(SAMPLE_RATE * 1.0)  # 1 s
    t = np.arange(n) / SAMPLE_RATE
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    mel = melspectrogram(wav)  # (80, frames)
    print("mel shape:", mel.shape)
    print("min:", float(mel.min()), "max:", float(mel.max()))

    # Fingerprint: frame 10 values at bins 9..11 (the 440 Hz band).
    fi = min(10, mel.shape[1] - 1)
    fp = ",".join(f"{v:.4f}" for v in mel[9:12, fi])
    print("FINGERPRINT_FRAME", fi, "bins9..11:", fp)


if __name__ == "__main__":
    main()
