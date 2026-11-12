//! Mel-spectrogram preprocessing matching SadTalker's `src/utils/audio.py`.
//!
//! SadTalker (and the ExpNet model) consumes 16 kHz audio, computes a mel
//! spectrogram with the same conventions as `librosa.stft` +
//! `librosa.filters.mel`:
//!
//! - preemphasis filter `y[n] = x[n] - 0.97 * x[n-1]`
//! - STFT: `n_fft = 800`, `hop = 200`, `win = 800`, periodic Hann window,
//!   reflect-padded by `n_fft/2` on both sides (center=True)
//! - magnitude spectrum (`librosa.feature.melspectrogram(S=np.abs(D), ...)`
//!   uses the pre-computed values as-is — the `power=2.0` default only applies
//!   when the raw waveform `y` is passed)
//! - mel filterbank: 80 bins, `fmin = 55`, `fmax = 7600`, Slaney scale,
//!   Slaney normalization (`norm="slaney"`)
//! - `_amp_to_db`: `20*log10(max(min_level, |S|))`, `min_level = exp(-100/20*ln10)`
//! - subtract `ref_level_db = 20`
//! - `_normalize`: symmetric, `max_abs_value = 4`, clipped to `[-4, 4]`
//!
//! The output is a `(time_frames, 80)` mel matrix — the same `orig_mel.T`
//! that `generate_batch.py` turns into 16-column windows.

use std::f32::consts::PI;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Pre-emphasis coefficient (SadTalker hparam).
const PREEMPHASIS: f32 = 0.97;
/// STFT FFT size.
pub const N_FFT: usize = 800;
/// STFT hop size (200 = 12.5 ms at 16 kHz).
pub const HOP_SIZE: usize = 200;
/// Window length (800).
pub const WIN_SIZE: usize = 800;
/// Sample rate ExpNet expects.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of mel bins.
pub const N_MELS: usize = 80;
/// Minimum mel frequency.
const F_MIN: f32 = 55.0;
/// Maximum mel frequency.
const F_MAX: f32 = 7600.0;
/// Reference level db (subtracted after db conversion).
const REF_LEVEL_DB: f32 = 20.0;
/// Minimum level db (clamps the linear magnitude before db).
const MIN_LEVEL_DB: f32 = -100.0;
/// Symmetric normalization max absolute value.
const MAX_ABS_VALUE: f32 = 4.0;

/// A mel-spectrogram: `frames × n_mels` in row-major order.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// `[n_frames, n_mels]`.
    pub data: Vec<f32>,
    /// Number of time frames.
    pub n_frames: usize,
    /// Number of mel bins (`N_MELS`).
    pub n_mels: usize,
}

impl MelSpectrogram {
    /// Row `i` (time frame) as an `[n_mels]` slice.
    pub fn frame(&self, i: usize) -> &[f32] {
        &self.data[i * self.n_mels..(i + 1) * self.n_mels]
    }
}

/// Slaney mel scale: linear below 1 kHz, log above.
fn hz_to_mel(hz: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_min = 0.0f32;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Builds the Slaney-normalized mel filterbank (librosa `mel`).
///
/// Returns `[n_mels, n_fft/2 + 1]` weights in row-major order.
pub fn mel_filterbank() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let mut weights = vec![0.0f32; N_MELS * n_freqs];

    // Mel points: N_MELS+2 spaced between f_min and f_max.
    let mel_min = hz_to_mel(F_MIN);
    let mel_max = hz_to_mel(F_MAX);
    let mel_step = (mel_max - mel_min) / (N_MELS as f32 + 1.0);
    let hz_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mel_min + i as f32 * mel_step))
        .collect();
    let fft_bins: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * N_FFT as f32 / SAMPLE_RATE as f32)
        .collect();

    for i in 0..N_MELS {
        let left = fft_bins[i];
        let center = fft_bins[i + 1];
        let right = fft_bins[i + 2];

        for j in 0..n_freqs {
            let jf = j as f32;
            let w = if jf >= left && jf < center {
                (jf - left) / (center - left)
            } else if jf >= center && jf <= right {
                (right - jf) / (right - center)
            } else {
                0.0
            };
            weights[i * n_freqs + j] = w;
        }
        // Slaney normalization (librosa `norm="slaney"`): each filter is scaled
        // by `2 / (mel_f[i+2] - mel_f[i])`, where `mel_f` is the mel frequency
        // in **Hz** (librosa uses `mel_frequencies`, not fft bin indices).
        let width_hz = (hz_points[i + 2] - hz_points[i]).max(1e-6);
        let scale = 2.0 / width_hz;
        for j in 0..n_freqs {
            weights[i * n_freqs + j] *= scale;
        }
    }
    weights
}

/// Periodic Hann window of length `WIN_SIZE` (librosa `get_window('hann')`).
fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / len as f32).cos()))
        .collect()
}

/// Computes the STFT magnitude spectrogram `[n_frames, n_freqs]` of `samples`.
///
/// Mirrors `librosa.stft(y, n_fft=800, hop_length=200, win_length=800,
/// center=True)` — reflect-pad by `n_fft/2` each side, frame, window, FFT,
/// magnitude.
pub fn stft_magnitude(samples: &[f32]) -> (Vec<f32>, usize) {
    // Reflect pad by n_fft/2 on both sides (numpy `pad(..., mode='reflect')`,
    // which is what librosa uses for `center=True`). For a pad of P on a
    // length-N signal, the left pad is:
    //   [s[P-1], s[P-2], ..., s[1], s[0]]  (P elements, NOT including s[0]
    //   twice — reflect drops the edge sample)
    // and the right pad is:
    //   [s[N-2], s[N-3], ..., s[N-P-1]].
    let pad = N_FFT / 2;
    let n = samples.len();
    let mut padded = Vec::with_capacity(n + 2 * pad);
    if pad > 0 && n > 1 {
        // Left reflect: s[pad-1], s[pad-2], ..., s[0]
        for i in (0..pad).rev() {
            padded.push(samples[i.min(n - 1)]);
        }
        padded.extend_from_slice(samples);
        // Right reflect: s[n-2], s[n-3], ..., s[n-pad-1]
        for i in 1..=pad {
            padded.push(samples[(n - 1 - i).max(0).min(n - 1)]);
        }
    } else {
        padded.extend_from_slice(samples);
    }

    let n_freqs = N_FFT / 2 + 1;
    let n_frames = 1 + (padded.len() - N_FFT) / HOP_SIZE;
    let window = hann_window(WIN_SIZE);
    let mut planner = FftPlanner::<f32>::new();
    let fft: std::sync::Arc<dyn Fft<f32>> = planner.plan_fft_forward(N_FFT);

    let mut mags = vec![0.0f32; n_frames * n_freqs];
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];

    for (fi, frame_start) in (0..padded.len() - N_FFT + 1).step_by(HOP_SIZE).enumerate() {
        for k in 0..N_FFT {
            buf[k] = Complex::new(padded[frame_start + k] * window[k], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..n_freqs {
            mags[fi * n_freqs + k] = buf[k].norm();
        }
    }

    (mags, n_frames)
}

/// Applies the mel filterbank to a magnitude spectrum row (one time frame)
/// and returns the 80 mel energies.
fn apply_mel_filters(mag_row: &[f32], fb: &[f32], n_freqs: usize) -> Vec<f32> {
    let mut mel = vec![0.0f32; N_MELS];
    for i in 0..N_MELS {
        let row = &fb[i * n_freqs..(i + 1) * n_freqs];
        let mut acc = 0.0f32;
        for (j, w) in row.iter().enumerate() {
            acc += w * mag_row[j];
        }
        mel[i] = acc;
    }
    mel
}

/// `_amp_to_db`: `20*log10(max(min_level, x)) - ref`.
fn amp_to_db(x: f32) -> f32 {
    let min_level = (MIN_LEVEL_DB / 20.0 * std::f32::consts::LN_10).exp();
    20.0 * x.max(min_level).log10() - REF_LEVEL_DB
}

/// `_normalize` (symmetric): clip `2*max_abs * (S - min_db)/(-min_db) - max_abs`
fn normalize_db(s: f32) -> f32 {
    let v = 2.0 * MAX_ABS_VALUE * ((s - MIN_LEVEL_DB) / (-MIN_LEVEL_DB)) - MAX_ABS_VALUE;
    v.clamp(-MAX_ABS_VALUE, MAX_ABS_VALUE)
}

/// Computes the SadTalker mel spectrogram of mono 16 kHz samples.
///
/// Returns a `MelSpectrogram` with `n_frames` rows of `N_MELS` normalized
/// mel values (the `orig_mel.T` matrix).
pub fn melspectrogram(samples: &[f32]) -> MelSpectrogram {
    // Preemphasis.
    let mut y = Vec::with_capacity(samples.len());
    for (i, &s) in samples.iter().enumerate() {
        let prev = if i == 0 { 0.0 } else { samples[i - 1] };
        y.push(s - PREEMPHASIS * prev);
    }

    let (mags, n_frames) = stft_magnitude(&y);
    let n_freqs = N_FFT / 2 + 1;
    let fb = mel_filterbank();

    let mut data = Vec::with_capacity(n_frames * N_MELS);
    for fi in 0..n_frames {
        let row = &mags[fi * n_freqs..(fi + 1) * n_freqs];
        // SadTalker passes S=|D| (magnitude) to melspectrogram, which uses the
        // values as-is (no power squaring).
        let mel = apply_mel_filters(row, &fb, n_freqs);
        for v in mel {
            data.push(normalize_db(amp_to_db(v)));
        }
    }

    MelSpectrogram {
        data,
        n_frames,
        n_mels: N_MELS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T3-adjacent: a synthetic sine produces a finite, bounded mel spectrum
    /// with the expected shape.
    #[test]
    fn mel_of_sine_is_finite() {
        let sr = SAMPLE_RATE as usize;
        let n = sr; // 1 second
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let mel = melspectrogram(&samples);
        assert!(mel.n_frames > 1);
        assert_eq!(mel.n_mels, 80);
        for v in &mel.data {
            assert!(v.is_finite(), "mel value not finite: {v}");
            assert!((-4.0..=4.0).contains(v), "mel out of range: {v}");
        }
    }

    /// Cross-check against the SadTalker/librosa reference (`mel_reference.py`):
    /// a 1 s 440 Hz sine has normalized mel values `[1.6546, 1.9898, 0.4906]` at
    /// bins 9..11 of frame 10. This pins the mel pipeline (STFT scale, Slaney
    /// filterbank normalization, amp->db, and symmetric normalize) exactly.
    #[test]
    fn mel_matches_python_reference() {
        let sr = SAMPLE_RATE as usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let m = melspectrogram(&samples);
        assert_eq!(m.n_frames, 81, "librosa gives 81 frames for 1 s");
        let frame10 = m.frame(10);
        let expected = [1.6546, 1.9898, 0.4906];
        for (bin, exp) in expected.iter().enumerate() {
            let got = frame10[9 + bin];
            assert!(
                (got - exp).abs() < 0.01,
                "bin {} frame10: got {got}, expected {exp}",
                9 + bin
            );
        }
    }

    /// Pure silence maps to the minimum value (all zeros after normalize).
    #[test]
    fn mel_of_silence_is_min() {
        let samples = vec![0.0f32; SAMPLE_RATE as usize];
        let mel = melspectrogram(&samples);
        // Silence -> magnitude 0 -> log10 clamped to min_level -> -100 dB ->
        // normalize -> -4.0 (the symmetric minimum).
        for v in &mel.data {
            assert!((v + 4.0).abs() < 1e-3, "silence mel should be -4, got {v}");
        }
    }

    /// The filterbank rows use Slaney normalization (each row sums to ~0.05,
    /// matching librosa `norm="slaney"` for n_mels=80, n_fft=800).
    #[test]
    fn filterbank_rows_normalized() {
        let fb = mel_filterbank();
        let n_freqs = N_FFT / 2 + 1;
        for i in 0..N_MELS {
            let row = &fb[i * n_freqs..(i + 1) * n_freqs];
            let sum: f32 = row.iter().sum();
            assert!(
                (0.04..=0.06).contains(&sum),
                "row {i} sum {sum} (expected ~0.05)"
            );
        }
    }
}
