//! Audio analysis (report §3.3, plan task 4).
//!
//! Reads a WAV file (via `hound`) or takes in-memory f32 samples, and computes:
//! - a per-frame RMS envelope at 40 ms frames (25 fps),
//! - head-bob / breathing parameters derived from the envelope,
//! - audio feature frames for the lip-sync stage (task 5).
//!
//! Head bob: the avatar's head sways slightly with the voice energy. We model
//! it as a vertical offset proportional to a smoothed envelope, plus a slow
//! sinusoidal breathing cycle (default ~4 s period) modulated by loudness.

use std::path::Path;

/// Frame size in seconds: 40 ms @ 25 fps.
pub const FRAME_SECS: f32 = 0.04;
/// Default frames per second.
pub const FPS: f32 = 25.0;

/// Per-frame analysis features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// RMS energy of the frame in [0, 1].
    pub rms: f32,
    /// Head bob vertical offset in [-1, 1] (normalized).
    pub head_bob: f32,
    /// Breathing amplitude in [0, 1] (normalized).
    pub breathing: f32,
    /// Audio feature vector for lip-sync (task 5): [rms, rms_delta, rms_avg].
    pub features: [f32; 3],
}

/// Analysis result for a whole audio clip.
#[derive(Debug, Clone)]
pub struct Analysis {
    /// Frames at `FPS` (40 ms each).
    pub frames: Vec<Frame>,
    /// Sample rate of the source audio.
    pub sample_rate: u32,
    /// Peak RMS across all frames (for normalization).
    pub peak_rms: f32,
}

impl Analysis {
    /// Audio duration in seconds.
    pub fn duration(&self) -> f32 {
        self.frames.len() as f32 / FPS
    }

    /// Number of frames (= number of video frames at 25 fps).
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

/// Reads a WAV file (16-bit PCM or 32-bit float) and returns mono f32 samples
/// in [-1, 1] plus the sample rate.
pub fn read_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;
    let mut samples: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for s in reader.samples::<i32>() {
                samples.push(s? as f32 / max);
            }
        }
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                samples.push(s?);
            }
        }
    }
    // Downmix to mono by averaging channels.
    if channels > 1 {
        let mono: Vec<f32> = samples
            .chunks_exact(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect();
        samples = mono;
    }
    Ok((samples, sample_rate))
}

/// Computes the per-frame RMS envelope of `samples` at `sample_rate`.
/// Each frame covers `FRAME_SECS` (40 ms) of audio.
pub fn rms_envelope(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let frame_len = (sample_rate as f32 * FRAME_SECS).round() as usize;
    if frame_len == 0 || samples.is_empty() {
        return Vec::new();
    }
    samples
        .chunks(frame_len)
        .map(|chunk| {
            let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
            (sum_sq / chunk.len() as f32).sqrt()
        })
        .collect()
}

/// Full analysis: RMS envelope + head bob + breathing + lip-sync features.
///
/// - `peak_rms` is the maximum frame RMS; frames are normalized by it.
/// - Head bob: smoothed, normalized envelope mapped to [-1, 1] with a small
///   amplitude (voice energy → slight head sway).
/// - Breathing: slow sinusoid (period `breath_period_secs`, default 4 s)
///   scaled by (1 - loudness) so breathing is most visible during silence.
pub fn analyze(samples: &[f32], sample_rate: u32, breath_period_secs: f32) -> Analysis {
    let env = rms_envelope(samples, sample_rate);
    let peak = env.iter().cloned().fold(0.0f32, f32::max).max(1e-6);

    // Exponential smoothing for head bob (attack fast, release slower).
    let mut smoothed = 0.0f32;
    let period = breath_period_secs.max(0.5);
    let frames: Vec<Frame> = env
        .iter()
        .enumerate()
        .map(|(i, &rms)| {
            let norm = (rms / peak).clamp(0.0, 1.0);
            // Smooth: rise quickly, fall slowly.
            let alpha = if norm > smoothed { 0.5 } else { 0.15 };
            smoothed += alpha * (norm - smoothed);
            let t = i as f32 * FRAME_SECS;
            // Breathing: slow sinusoid, stronger during silence.
            let breath = (1.0 - smoothed) * 0.5 * (1.0 - (2.0 * std::f32::consts::PI * t / period).cos());
            Frame {
                rms: norm,
                head_bob: smoothed * 2.0 - 1.0, // [-1, 1]
                breathing: breath.clamp(0.0, 1.0),
                features: [norm, delta(&env, i, peak), smoothed],
            }
        })
        .collect();

    Analysis {
        frames,
        sample_rate,
        peak_rms: peak,
    }
}

/// Frame-to-frame RMS delta (normalized), used as a lip-sync feature.
fn delta(env: &[f32], i: usize, peak: f32) -> f32 {
    if i == 0 {
        return 0.0;
    }
    let d = (env[i] - env[i - 1]) / peak;
    d.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a synthetic WAV file (16-bit PCM mono) with the given samples.
    fn write_test_wav(path: &Path, samples: &[i16], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    /// Sine wave with amplitude `a` and frequency `f`.
    fn sine(freq: f32, amp: f32, secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    /// T4: envelope frame count = ceil(duration / 40 ms) at 25 fps.
    #[test]
    fn envelope_frame_count() {
        let sr = 22050;
        let samples = sine(220.0, 0.5, 1.0, sr); // exactly 1 s
        let env = rms_envelope(&samples, sr);
        assert_eq!(env.len(), 25); // 1 s @ 25 fps
        // Partial frame: 0.5 s → 13 frames (12 full + 1 partial).
        let samples = sine(220.0, 0.5, 0.5, sr);
        let env = rms_envelope(&samples, sr);
        assert_eq!(env.len(), 13);
        // Empty input.
        assert!(rms_envelope(&[], sr).is_empty());
    }

    /// T4: RMS of a sine wave ≈ amp / sqrt(2).
    #[test]
    fn rms_of_sine() {
        let sr = 22050;
        let amp = 0.8f32;
        let samples = sine(440.0, amp, 1.0, sr);
        let env = rms_envelope(&samples, sr);
        let expected = amp / (2.0f32).sqrt();
        for &r in &env {
            assert!((r - expected).abs() < 0.05, "rms {} vs {}", r, expected);
        }
    }

    /// T4: RMS bounds in [0, 1] and normalization by peak.
    #[test]
    fn rms_bounds_and_normalization() {
        let sr = 22050u32;
        // 0.5 s silence then 0.5 s loud sine.
        let mut samples = vec![0.0f32; (sr / 2) as usize];
        samples.extend(sine(440.0, 0.9, 0.5, sr));
        let a = analyze(&samples, sr, 4.0);
        assert_eq!(a.frame_count(), 25);
        assert!(a.peak_rms > 0.0);
        for f in &a.frames {
            assert!((0.0..=1.0).contains(&f.rms));
            assert!((-1.0..=1.0).contains(&f.head_bob));
            assert!((0.0..=1.0).contains(&f.breathing));
        }
        // First frames (silence) ≈ 0, last frames (loud) ≈ 1.
        assert!(a.frames[0].rms < 0.1);
        assert!(a.frames[24].rms > 0.5);
    }

    /// T4: head bob follows the envelope (smoothed, delayed).
    #[test]
    fn head_bob_follows_envelope() {
        let sr = 22050;
        let samples = sine(440.0, 0.9, 1.0, sr);
        let a = analyze(&samples, sr, 4.0);
        // Head bob should rise over the first frames (smoothing).
        assert!(a.frames[0].head_bob < a.frames[10].head_bob);
        assert!(a.frames[10].head_bob < a.frames[24].head_bob);
    }

    /// T4: breathing is a slow sinusoid, stronger during silence.
    #[test]
    fn breathing_silence_vs_speech() {
        let sr = 22050u32;
        let mut samples = vec![0.0f32; (sr * 2) as usize]; // 2 s silence
        samples.extend(sine(440.0, 0.9, 2.0, sr)); // 2 s speech
        let a = analyze(&samples, sr, 4.0);
        let silence_breath: f32 = a.frames[..25].iter().map(|f| f.breathing).sum::<f32>() / 25.0;
        let speech_breath: f32 = a.frames[50..].iter().map(|f| f.breathing).sum::<f32>() / 50.0;
        assert!(silence_breath > speech_breath, "silence {} speech {}", silence_breath, speech_breath);
    }

    /// T4: WAV read round-trip via hound.
    #[test]
    fn wav_read_roundtrip() {
        let tmp = std::env::temp_dir().join("avatar_test_wav.wav");
        let samples: Vec<i16> = (0..2205).map(|i| ((i % 100) * 300 - 15000) as i16).collect();
        write_test_wav(&tmp, &samples, 22050);
        let (out, sr) = read_wav(&tmp).unwrap();
        assert_eq!(sr, 22050);
        assert_eq!(out.len(), 2205);
        // First sample matches.
        assert!((out[0] - samples[0] as f32 / 32768.0).abs() < 1e-3);
        let _ = std::fs::remove_file(&tmp);
    }

    /// T4: stereo WAV downmixed to mono.
    #[test]
    fn wav_stereo_downmix() {
        let tmp = std::env::temp_dir().join("avatar_test_stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&tmp, spec).unwrap();
        // L=1000, R=-1000 → mono 0.
        for _ in 0..10 {
            w.write_sample(1000i16).unwrap();
            w.write_sample(-1000i16).unwrap();
        }
        w.finalize().unwrap();
        let (out, _) = read_wav(&tmp).unwrap();
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|s| s.abs() < 1e-3));
        let _ = std::fs::remove_file(&tmp);
    }

    /// T4: features vector has 3 components in [-1, 1].
    #[test]
    fn features_shape() {
        let sr = 22050;
        let samples = sine(440.0, 0.5, 0.2, sr);
        let a = analyze(&samples, sr, 4.0);
        for f in &a.frames {
            assert_eq!(f.features.len(), 3);
            for v in &f.features {
                assert!((-1.0..=1.0).contains(v));
            }
        }
    }
}
