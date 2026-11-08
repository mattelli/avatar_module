//! Deterministic phoneme→viseme mapping (report §3.4(b), plan task 5).
//!
//! A hardcoded table maps espeak-ng IPA phonemes to viseme weights — no ML.
//! Viseme set (K = 12), classic Preston-Blair style:
//! `sil, AI, E, O, MBP, FV, L, TH, U, WQ, etc, rest`.
//!
//! Input: `PhonemeTiming` list (task 3) + frame count (task 4, 25 fps).
//! Output: per-frame viseme weight vectors. Each frame gets the weights of the
//! phoneme(s) active at that time; a frame covered by several phonemes blends
//! them proportionally to their overlap. Weights are normalized to sum to 1.

use crate::tts::PhonemeTiming;
use std::collections::BTreeMap;

/// Canonical viseme names (index = weight vector index).
pub const VISEMES: [&str; 12] = [
    "sil", "AI", "E", "O", "MBP", "FV", "L", "TH", "U", "WQ", "etc", "rest",
];

/// Maps a phoneme (espeak-ng IPA, possibly multi-char) to viseme weights.
/// Returns `(viseme_index, weight)` pairs; multiple entries = blend.
pub fn phoneme_to_viseme(p: &str) -> Vec<(usize, f32)> {
    let p = p.trim();
    // Silence / punctuation.
    if p.is_empty()
        || matches!(p, "." | "," | "!" | "?" | ";" | ":" | " " | "_" | "-" | "…")
    {
        return vec![(0, 1.0)]; // sil
    }
    let first = p.chars().next().unwrap();
    let second = p.chars().nth(1);
    // Affricates (multi-char phonemes).
    if p == "tʃ" || p == "dʒ" {
        return vec![(10, 1.0)]; // etc
    }
    match first {
        // --- Diphthongs: blend the two vowels ---
        'a' if second == Some('ɪ') => vec![(1, 1.0)],
        'a' if second == Some('ʊ') => vec![(1, 0.7), (3, 0.3)],
        'o' if second == Some('ʊ') => vec![(3, 0.7), (8, 0.3)],
        'e' if second == Some('ɪ') => vec![(2, 0.7), (1, 0.3)],
        // --- Vowels ---
        'a' | 'ɑ' | 'æ' | 'ʌ' => vec![(1, 1.0)], // AI (open mouth)
        'e' | 'ɛ' | 'ɜ' | 'ə' => vec![(2, 1.0)], // E (mid)
        'i' | 'ɪ' => vec![(2, 1.0)],             // E (spread)
        'o' | 'ɔ' | 'ɒ' => vec![(3, 1.0)],       // O (round)
        'u' | 'ʊ' => vec![(8, 1.0)],             // U (small round)
        // --- Bilabials / nasals / plosives ---
        'm' | 'b' | 'p' => vec![(4, 1.0)], // MBP (closed lips)
        // --- Labiodentals ---
        'f' | 'v' => vec![(5, 1.0)], // FV
        // --- Lingual-alveolar L ---
        'l' => vec![(6, 1.0)], // L
        // --- Dentals / fricatives ---
        'θ' | 'ð' | 's' | 'z' | 'ʃ' | 'ʒ' | 't' | 'd' | 'n' | 'k' | 'g' | 'ŋ'
        | 'r' | 'ɹ' | 'j' | 'h' | 'x' | 'w' | 'ʍ' | 'c' | 'ç' => {
            if first == 'θ' || first == 'ð' {
                vec![(7, 1.0)] // TH (tongue between teeth)
            } else if first == 'w' || first == 'ʍ' {
                vec![(9, 1.0)] // WQ (small round)
            } else {
                vec![(10, 1.0)] // etc (generic)
            }
        }
        // Stress / linking marks and anything unknown → rest.
        _ => vec![(11, 1.0)], // rest
    }
}

/// Per-frame viseme weights: `frames × K` matrix, each row sums to 1.
///
/// `timings` come from the TTS stage (task 3); `frame_count` from the analysis
/// stage (task 4, 25 fps). A frame is attributed to the phoneme(s) whose
/// time span overlaps it; overlaps are blended proportionally.
pub fn viseme_frames(timings: &[PhonemeTiming], frame_count: usize) -> Vec<Vec<f32>> {
    let k = VISEMES.len();
    let mut out = vec![vec![0.0f32; k]; frame_count];
    if frame_count == 0 {
        return out;
    }
    if timings.is_empty() {
        // No phonemes at all → every frame is rest.
        for row in &mut out {
            row[11] = 1.0;
        }
        return out;
    }
    // Frame i covers [i*0.04, (i+1)*0.04).
    const FRAME: f32 = 0.04;
    for (i, row) in out.iter_mut().enumerate() {
        let f0 = i as f32 * FRAME;
        let f1 = f0 + FRAME;
        // Collect overlapping phonemes and their overlap duration.
        let mut overlaps: BTreeMap<usize, f32> = BTreeMap::new();
        let mut total = 0.0f32;
        for t in timings {
            let ov = (t.end.min(f1) - t.start.max(f0)).max(0.0);
            if ov > 0.0 {
                for (vi, w) in phoneme_to_viseme(&t.phoneme) {
                    *overlaps.entry(vi).or_insert(0.0) += ov * w;
                    total += ov * w;
                }
            }
        }
        if total > 0.0 {
            for (vi, w) in overlaps {
                row[vi] = w / total;
            }
        } else {
            // No phoneme covers this frame (e.g. trailing silence) → rest.
            row[11] = 1.0;
        }
    }
    out
}

/// Returns the dominant viseme index per frame (argmax).
pub fn dominant_visemes(weights: &[Vec<f32>]) -> Vec<usize> {
    weights
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::Synthesizer;

    fn timing(p: &str, start: f32, end: f32) -> PhonemeTiming {
        PhonemeTiming {
            phoneme: p.to_string(),
            start,
            end,
        }
    }

    /// T5: known phonemes map to expected visemes.
    #[test]
    fn known_phonemes_map() {
        assert_eq!(phoneme_to_viseme("m"), vec![(4, 1.0)]);
        assert_eq!(phoneme_to_viseme("b"), vec![(4, 1.0)]);
        assert_eq!(phoneme_to_viseme("p"), vec![(4, 1.0)]);
        assert_eq!(phoneme_to_viseme("f"), vec![(5, 1.0)]);
        assert_eq!(phoneme_to_viseme("v"), vec![(5, 1.0)]);
        assert_eq!(phoneme_to_viseme("l"), vec![(6, 1.0)]);
        assert_eq!(phoneme_to_viseme("θ"), vec![(7, 1.0)]);
        assert_eq!(phoneme_to_viseme("ɑ"), vec![(1, 1.0)]);
        assert_eq!(phoneme_to_viseme("ə"), vec![(2, 1.0)]);
        assert_eq!(phoneme_to_viseme("o"), vec![(3, 1.0)]);
        assert_eq!(phoneme_to_viseme("u"), vec![(8, 1.0)]);
        assert_eq!(phoneme_to_viseme("w"), vec![(9, 1.0)]);
        assert_eq!(phoneme_to_viseme("."), vec![(0, 1.0)]);
        assert_eq!(phoneme_to_viseme(""), vec![(0, 1.0)]);
    }

    /// T5: unknown phoneme → rest.
    #[test]
    fn unknown_phoneme_rest() {
        assert_eq!(phoneme_to_viseme("¿"), vec![(11, 1.0)]);
        assert_eq!(phoneme_to_viseme("ˈ"), vec![(11, 1.0)]);
    }

    /// T5: weights sum to 1 per frame.
    #[test]
    fn weights_sum_to_one() {
        let timings = vec![
            timing("h", 0.0, 0.08),
            timing("ə", 0.08, 0.2),
            timing("l", 0.2, 0.28),
            timing("oʊ", 0.28, 0.4),
        ];
        let frames = viseme_frames(&timings, 10);
        assert_eq!(frames.len(), 10);
        for row in &frames {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
        }
    }

    /// T5: frame alignment — phoneme active at frame time dominates.
    #[test]
    fn frame_alignment() {
        // 'm' from 0.0 to 0.08 (frames 0-1), 'ɑ' from 0.08 to 0.16 (frames 2-3).
        let timings = vec![timing("m", 0.0, 0.08), timing("ɑ", 0.08, 0.16)];
        let frames = viseme_frames(&timings, 4);
        let dom = dominant_visemes(&frames);
        assert_eq!(dom[0], 4); // m → MBP
        assert_eq!(dom[1], 4);
        assert_eq!(dom[2], 1); // ɑ → AI
        assert_eq!(dom[3], 1);
    }

    /// T5: overlapping phonemes blend proportionally.
    #[test]
    fn overlap_blending() {
        // 'm' covers 0.0-0.06, 'ɑ' covers 0.02-0.08 → frame 0 (0.0-0.04)
        // overlaps m by 0.04 and ɑ by 0.02.
        let timings = vec![timing("m", 0.0, 0.06), timing("ɑ", 0.02, 0.08)];
        let frames = viseme_frames(&timings, 1);
        let m = frames[0][4];
        let a = frames[0][1];
        assert!(m > a, "m {} should dominate a {}", m, a);
        assert!((m + a - 1.0).abs() < 1e-4);
    }

    /// T5: no timings → all frames rest.
    #[test]
    fn no_timings_rest() {
        let frames = viseme_frames(&[], 5);
        assert_eq!(frames.len(), 5);
        for row in &frames {
            assert_eq!(row[11], 1.0);
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    /// T5: zero frames → empty output.
    #[test]
    fn zero_frames() {
        assert!(viseme_frames(&[timing("m", 0.0, 0.1)], 0).is_empty());
    }

    /// T5: trailing silence → rest frames.
    #[test]
    fn trailing_silence_rest() {
        let timings = vec![timing("m", 0.0, 0.04)];
        let frames = viseme_frames(&timings, 3);
        assert_eq!(dominant_visemes(&frames)[0], 4);
        assert_eq!(dominant_visemes(&frames)[1], 11);
        assert_eq!(dominant_visemes(&frames)[2], 11);
    }

    /// T5: diphthong blends two vowels.
    #[test]
    fn diphthong_blend() {
        let v = phoneme_to_viseme("oʊ");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], (3, 0.7));
        assert_eq!(v[1], (8, 0.3));
    }

    /// T5: integration with TTS timings — real synthesis produces sane frames.
    #[test]
    fn integration_with_tts() {
        let model = std::path::PathBuf::from(".res").join("en_US-hfc_female-medium.onnx");
        if !model.exists() {
            return;
        }
        let mut tts = crate::tts::EmbeddedPiper::new(&model).unwrap();
        let r = tts.synthesize("Hello world").unwrap();
        let frame_count = (r.duration() / 0.04).ceil() as usize;
        let frames = viseme_frames(&r.phonemes, frame_count);
        assert_eq!(frames.len(), frame_count);
        for row in &frames {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
        // Not all frames should be rest (there is speech).
        let rest = frames.iter().filter(|r| r[11] > 0.5).count();
        assert!(rest < frame_count, "all frames rest?");
    }
}
