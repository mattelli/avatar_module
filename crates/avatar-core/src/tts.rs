//! Piper TTS integration (report §3.2, plan task 3).
//!
//! Two backends:
//! - **embedded** (default feature `embedded`): `piper-rs` runs the ONNX voice
//!   model in-process via ONNX Runtime; espeak-ng is compiled in statically.
//! - **subprocess** (feature `subprocess`): shells out to the `piper` CLI
//!   binary, useful when the embedded build is unavailable.
//!
//! Per-segment synthesis returns f32 samples + the phoneme string with
//! estimated timings. Piper models do not expose per-phoneme durations through
//! `piper-rs`, so timings are estimated by distributing the total audio
//! duration across phonemes with class-based weights (vowels longer than
//! consonants, punctuation → pause). This is deterministic and sufficient for
//! the viseme mapping (task 5).

use std::path::{Path, PathBuf};

/// A phoneme with its estimated time span within the segment audio.
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeTiming {
    /// The phoneme as produced by espeak-ng (e.g. `h`, `ə`, `l`, `oʊ`).
    pub phoneme: String,
    /// Start time in seconds (from segment start).
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

/// Result of synthesizing one segment.
#[derive(Debug, Clone)]
pub struct SynthResult {
    /// f32 PCM samples in [-1, 1].
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Phoneme timings covering `[0, duration]`.
    pub phonemes: Vec<PhonemeTiming>,
}

impl SynthResult {
    /// Audio duration in seconds.
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Common synthesis interface for both backends.
pub trait Synthesizer {
    /// Synthesizes `text` into samples + phoneme timings.
    fn synthesize(&mut self, text: &str) -> anyhow::Result<SynthResult>;
}

/// Relative weight per phoneme class used for timing estimation.
fn phoneme_weight(p: &str) -> f32 {
    let first = p.chars().next().unwrap_or('ə');
    match first {
        // Long vowels / diphthongs.
        'a' | 'e' | 'i' | 'o' | 'u' | 'ɑ' | 'æ' | 'ɔ' | 'ɛ' | 'ɜ' | 'ɪ' | 'ʊ' | 'ʌ' | 'ə' => 1.6,
        // Punctuation / silence markers.
        '.' | ',' | '!' | '?' | ';' | ':' | ' ' | '_' => 0.8,
        // Consonants and everything else.
        _ => 1.0,
    }
}

/// Distributes `duration` seconds across `phonemes` using class weights.
/// Returns timings that are monotonically increasing and cover `[0, duration]`.
pub fn estimate_timings(phonemes: &[String], duration: f32) -> Vec<PhonemeTiming> {
    if phonemes.is_empty() || duration <= 0.0 {
        return Vec::new();
    }
    let weights: Vec<f32> = phonemes.iter().map(|p| phoneme_weight(p)).collect();
    let total: f32 = weights.iter().sum();
    let mut t = 0.0f32;
    let mut out = Vec::with_capacity(phonemes.len());
    for (p, w) in phonemes.iter().zip(&weights) {
        let d = duration * w / total;
        out.push(PhonemeTiming {
            phoneme: p.clone(),
            start: t,
            end: t + d,
        });
        t += d;
    }
    // Force the last phoneme to end exactly at `duration` (float rounding).
    if let Some(last) = out.last_mut() {
        last.end = duration;
    }
    out
}

/// Embedded backend: `piper-rs` + ONNX Runtime in-process.
#[cfg(feature = "embedded")]
pub struct EmbeddedPiper {
    inner: piper_rs::Piper,
    espeak_voice: String,
    /// Piper `length_scale` override (speed control). `None` = the model's
    /// configured value (usually 1.0). Larger = slower speech.
    length_scale: Option<f32>,
}

#[cfg(feature = "embedded")]
impl EmbeddedPiper {
    /// Loads the voice model (`model.onnx` + `model.onnx.json`).
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        let config_path = config_path_for(model_path);
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&config_path)?)?;
        let espeak_voice = config["espeak"]["voice"]
            .as_str()
            .unwrap_or("en-us")
            .to_string();
        let inner = piper_rs::Piper::new(model_path, &config_path)?;
        Ok(Self {
            inner,
            espeak_voice,
            length_scale: None,
        })
    }

    /// Sets the Piper `length_scale` (speed control): `1.0` = the model's
    /// native rate, `> 1.0` = slower, `< 1.0` = faster. `None` restores the
    /// model default.
    pub fn with_length_scale(mut self, length_scale: Option<f32>) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Phonemizes `text` with espeak-ng using the model's voice.
    fn phonemize(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let ph = espeak_rs::text_to_phonemes(text, &self.espeak_voice, None)?;
        Ok(ph
            .into_iter()
            .filter(|s| !s.is_empty())
            .flat_map(|s| s.split_whitespace().map(str::to_string).collect::<Vec<_>>())
            .collect())
    }
}

#[cfg(feature = "embedded")]
impl Synthesizer for EmbeddedPiper {
    fn synthesize(&mut self, text: &str) -> anyhow::Result<SynthResult> {
        let phonemes = self.phonemize(text)?;
        let (samples, sample_rate) =
            self.inner
                .create(text, false, None, self.length_scale, None, None)?;
        let duration = samples.len() as f32 / sample_rate as f32;
        let timings = estimate_timings(&phonemes, duration);
        Ok(SynthResult {
            samples,
            sample_rate,
            phonemes: timings,
        })
    }
}

/// Subprocess backend: shells out to the `piper` CLI (feature `subprocess`).
#[cfg(feature = "subprocess")]
pub struct SubprocessPiper {
    /// Path to the `piper` executable (default: `piper` on PATH).
    pub piper_bin: PathBuf,
    /// Voice model path (`model.onnx`).
    pub model_path: PathBuf,
    /// Piper `length_scale` override (speed control). `None` = the model's
    /// configured value. Larger = slower speech.
    pub length_scale: Option<f32>,
}

#[cfg(feature = "subprocess")]
impl SubprocessPiper {
    pub fn new(piper_bin: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            piper_bin: piper_bin.into(),
            model_path: model_path.into(),
            length_scale: None,
        }
    }

    /// Sets the Piper `length_scale` (speed control): `1.0` = the model's
    /// native rate, `> 1.0` = slower, `< 1.0` = faster.
    pub fn with_length_scale(mut self, length_scale: Option<f32>) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Phonemizes `text` by calling the `espeak-ng` CLI (best effort).
    fn phonemize(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let out = std::process::Command::new("espeak-ng")
            .args(["-q", "-x", "--sep=_", text])
            .output()?;
        if !out.status.success() {
            anyhow::bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.split_whitespace().map(str::to_string).collect())
    }
}

#[cfg(feature = "subprocess")]
impl Synthesizer for SubprocessPiper {
    fn synthesize(&mut self, text: &str) -> anyhow::Result<SynthResult> {
        let config_path = config_path_for(&self.model_path);
        let tmp = std::env::temp_dir().join(format!("avatar_piper_{}.wav", std::process::id()));
        let mut args: Vec<String> = vec![
            "-m".into(),
            self.model_path.to_str().unwrap_or_default().into(),
            "-c".into(),
            config_path.to_str().unwrap_or_default().into(),
            "-f".into(),
            tmp.to_str().unwrap_or_default().into(),
        ];
        if let Some(ls) = self.length_scale {
            args.push("--length_scale".into());
            args.push(ls.to_string());
        }
        let out = std::process::Command::new(&self.piper_bin)
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?
            .wait_with_output()?;
        if !out.status.success() {
            anyhow::bail!("piper CLI failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        let bytes = std::fs::read(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        let (samples, sample_rate) = parse_wav_i16(&bytes)?;
        let duration = samples.len() as f32 / sample_rate as f32;
        let phonemes = self.phonemize(text)?;
        let timings = estimate_timings(&phonemes, duration);
        Ok(SynthResult {
            samples,
            sample_rate,
            phonemes: timings,
        })
    }
}

/// Returns the `.onnx.json` config path next to a `.onnx` model path.
pub fn config_path_for(model_path: &Path) -> PathBuf {
    let mut s = model_path.to_path_buf().into_os_string();
    s.push(".json");
    PathBuf::from(s)
}

/// Minimal WAV parser: 16-bit PCM mono/stereo → mono f32 samples.
pub fn parse_wav_i16(bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a WAV file");
    }
    let mut pos = 12usize;
    let mut sample_rate = 0u32;
    let mut channels = 1u16;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes(body[2..4].try_into()?);
                sample_rate = u32::from_le_bytes(body[4..8].try_into()?);
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    let data = data.ok_or_else(|| anyhow::anyhow!("no data chunk"))?;
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2 * channels as usize) {
        let mut acc = 0i32;
        for c in chunk.chunks_exact(2) {
            acc += i16::from_le_bytes(c.try_into()?) as i32;
        }
        samples.push(acc as f32 / channels as f32 / i16::MAX as f32);
    }
    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the bundled test voice model, if present.
    fn test_model() -> Option<PathBuf> {
        let p = PathBuf::from(".res").join("en_US-hfc_female-medium.onnx");
        p.exists().then_some(p)
    }

    /// T3: synthesis returns non-empty WAV samples, duration > 0.
    #[test]
    fn synthesize_nonempty() {
        let Some(model) = test_model() else { return };
        let mut tts = EmbeddedPiper::new(&model).unwrap();
        let r = tts.synthesize("Hello world").unwrap();
        assert!(!r.samples.is_empty());
        assert!(r.duration() > 0.0);
        assert!(r.sample_rate > 0);
    }

    /// T3: phoneme timings are monotonic and cover the audio duration.
    #[test]
    fn timings_monotonic_cover_duration() {
        let Some(model) = test_model() else { return };
        let mut tts = EmbeddedPiper::new(&model).unwrap();
        let r = tts.synthesize("Hello world").unwrap();
        assert!(!r.phonemes.is_empty());
        let mut prev_end = 0.0f32;
        for t in &r.phonemes {
            assert!(t.start >= prev_end - 1e-4, "non-monotonic at {}", t.phoneme);
            assert!(t.end > t.start);
            prev_end = t.end;
        }
        let dur = r.duration();
        assert!((r.phonemes.last().unwrap().end - dur).abs() < 1e-3);
    }

    /// T3: per-segment synthesis — two segments give independent results.
    #[test]
    fn per_segment_synthesis() {
        let Some(model) = test_model() else { return };
        let mut tts = EmbeddedPiper::new(&model).unwrap();
        let a = tts.synthesize("Hello").unwrap();
        let b = tts.synthesize("Goodbye").unwrap();
        assert!(!a.samples.is_empty() && !b.samples.is_empty());
        assert!(a.duration() > 0.0 && b.duration() > 0.0);
    }

    /// `length_scale` controls speed: a larger value yields a longer clip
    /// (slower speech) for the same text.
    #[test]
    fn length_scale_controls_speed() {
        let Some(model) = test_model() else { return };
        let text = "Hello world, this is a speed test.";

        let mut fast = EmbeddedPiper::new(&model).unwrap().with_length_scale(Some(0.5));
        let mut slow = EmbeddedPiper::new(&model).unwrap().with_length_scale(Some(2.0));

        let d_fast = fast.synthesize(text).unwrap().duration();
        let d_slow = slow.synthesize(text).unwrap().duration();

        assert!(
            d_slow > d_fast * 1.5,
            "length_scale 2.0 should be much slower than 0.5 \
             (fast {d_fast:.2}s vs slow {d_slow:.2}s)"
        );
    }

    /// Timing estimation: monotonic, covers duration, empty input → empty.
    #[test]
    fn estimate_timings_basics() {
        assert!(estimate_timings(&[], 1.0).is_empty());
        assert!(estimate_timings(&["h".into()], 0.0).is_empty());
        let ph: Vec<String> = ["h", "ə", "l", "oʊ"].iter().map(|s| s.to_string()).collect();
        let t = estimate_timings(&ph, 1.0);
        assert_eq!(t.len(), 4);
        assert!((t[0].start - 0.0).abs() < 1e-6);
        assert!((t[3].end - 1.0).abs() < 1e-6);
        for w in t.windows(2) {
            assert!(w[1].start >= w[0].end - 1e-6);
        }
        // Vowel 'oʊ' should be longer than consonant 'l'.
        let d_l = t[2].end - t[2].start;
        let d_v = t[3].end - t[3].start;
        assert!(d_v > d_l);
    }

    /// WAV parser round-trip on a hand-built 16-bit mono WAV.
    #[test]
    fn parse_wav_roundtrip() {
        let samples: Vec<i16> = [100, -200, 300, -400].to_vec();
        let mut b = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data_len).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // PCM
        b.extend_from_slice(&1u16.to_le_bytes()); // mono
        b.extend_from_slice(&22050u32.to_le_bytes());
        b.extend_from_slice(&44100u32.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&data_len.to_le_bytes());
        for s in &samples {
            b.extend_from_slice(&s.to_le_bytes());
        }
        let (out, sr) = parse_wav_i16(&b).unwrap();
        assert_eq!(sr, 22050);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 100.0 / 32767.0).abs() < 1e-3);
        assert!((out[3] + 400.0 / 32767.0).abs() < 1e-3);
    }

    /// Config path helper.
    #[test]
    fn config_path_helper() {
        assert_eq!(
            config_path_for(Path::new("a/b.onnx")),
            PathBuf::from("a/b.onnx.json")
        );
    }
}
