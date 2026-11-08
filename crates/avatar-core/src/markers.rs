//! Marker parser (report §3.1, plan task 2).
//!
//! Scans `<tag>` tokens in the input text and splits it into `(segment, mode)`
//! spans. Known modes: `neutral`, `happy`, `mad`, `sad`, `surprised`,
//! `disgusted`, `fearful`. Unknown tags are dropped silently; text without any
//! marker becomes a single neutral span.
//!
//! Span start times are known only after TTS, so this stage only produces
//! ordered spans; the TTS stage reports per-segment durations to build the
//! expression timeline (task 3 / task 9).

/// Expression modes supported by the renderer (task 6 atlas).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Neutral,
    Happy,
    Mad,
    Sad,
    Surprised,
    Disgusted,
    Fearful,
}

impl Mode {
    /// Parses a marker tag name (case-insensitive). Returns `None` for unknown tags.
    pub fn from_tag(tag: &str) -> Option<Mode> {
        match tag.to_ascii_lowercase().as_str() {
            "neutral" => Some(Mode::Neutral),
            "happy" => Some(Mode::Happy),
            "mad" | "angry" => Some(Mode::Mad),
            "sad" => Some(Mode::Sad),
            "surprised" => Some(Mode::Surprised),
            "disgusted" | "disgust" => Some(Mode::Disgusted),
            "fearful" | "fear" => Some(Mode::Fearful),
            _ => None,
        }
    }

    /// Canonical tag name for this mode.
    pub fn tag(self) -> &'static str {
        match self {
            Mode::Neutral => "neutral",
            Mode::Happy => "happy",
            Mode::Mad => "mad",
            Mode::Sad => "sad",
            Mode::Surprised => "surprised",
            Mode::Disgusted => "disgusted",
            Mode::Fearful => "fearful",
        }
    }
}

/// A span of text with the expression mode active for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Text of the segment (markers stripped, whitespace preserved).
    pub text: String,
    /// Expression mode active for this segment.
    pub mode: Mode,
}

/// Parses `text` into `(segment, mode)` spans.
///
/// - `<tag>` tokens are consumed as markers; a known tag switches the mode for
///   the following text.
/// - Unknown tags (e.g. `<xyz>`) are dropped silently.
/// - Text before any marker is neutral.
/// - Empty segments (two adjacent markers) produce no span.
/// - An unclosed `<` at the end of the text is treated as literal text.
pub fn parse(text: &str) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    let mut mode = Mode::Neutral;
    let mut buf = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '<' {
            // Try to read a tag: `<name>` up to the next '>'.
            let mut tag = String::new();
            let mut closed = false;
            while let Some(&n) = chars.peek() {
                if n == '>' {
                    chars.next();
                    closed = true;
                    break;
                }
                if n == '<' {
                    // Nested '<' — not a valid tag; treat everything as literal.
                    break;
                }
                tag.push(n);
                chars.next();
            }
            if closed {
                // Flush the pending segment before switching mode.
                if !buf.trim().is_empty() {
                    spans.push(Span {
                        text: std::mem::take(&mut buf),
                        mode,
                    });
                } else {
                    buf.clear();
                }
                if let Some(m) = Mode::from_tag(&tag) {
                    mode = m;
                }
                // Unknown tag: dropped silently, mode unchanged.
            } else {
                // Unclosed '<' — literal text.
                buf.push('<');
                buf.push_str(&tag);
            }
        } else {
            buf.push(c);
        }
    }

    if !buf.trim().is_empty() {
        spans.push(Span { text: buf, mode });
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes(text: &str) -> Vec<Mode> {
        parse(text).into_iter().map(|s| s.mode).collect()
    }

    fn texts(text: &str) -> Vec<String> {
        parse(text).into_iter().map(|s| s.text).collect()
    }

    /// T2: `"<happy> hi"` → 1 span mode=happy.
    #[test]
    fn single_marker() {
        let spans = parse("<happy> hi");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].mode, Mode::Happy);
        assert_eq!(spans[0].text.trim(), "hi");
    }

    /// T2: `"<xyz> hi"` → unknown ignored, neutral.
    #[test]
    fn unknown_marker_ignored() {
        assert_eq!(modes("<xyz> hi"), vec![Mode::Neutral]);
        assert_eq!(texts("<xyz> hi").len(), 1);
    }

    /// T2: `"<sad> a <mad> b"` → 2 spans in order.
    #[test]
    fn two_spans_in_order() {
        let spans = parse("<sad> a <mad> b");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], Span { text: " a ".into(), mode: Mode::Sad });
        assert_eq!(spans[1], Span { text: " b".into(), mode: Mode::Mad });
    }

    /// T2: no markers → single neutral span.
    #[test]
    fn no_markers_neutral() {
        let spans = parse("Hello world");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].mode, Mode::Neutral);
        assert_eq!(spans[0].text, "Hello world");
    }

    /// Adjacent markers: no empty span is emitted.
    #[test]
    fn adjacent_markers_no_empty_span() {
        let spans = parse("<happy><sad> hi");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].mode, Mode::Sad);
    }

    /// Tags are case-insensitive.
    #[test]
    fn case_insensitive() {
        assert_eq!(modes("<HAPPY> x"), vec![Mode::Happy]);
        assert_eq!(modes("<Happy> x"), vec![Mode::Happy]);
    }

    /// All known modes parse.
    #[test]
    fn all_known_modes() {
        for tag in ["neutral", "happy", "mad", "sad", "surprised", "disgusted", "fearful"] {
            let t = format!("<{}> x", tag);
            assert!(modes(&t)[0] != Mode::Neutral || tag == "neutral", "tag {}", tag);
        }
    }

    /// Unclosed '<' is literal text.
    #[test]
    fn unclosed_bracket_literal() {
        let spans = parse("a < b");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "a < b");
    }

    /// Empty input → no spans.
    #[test]
    fn empty_input() {
        assert!(parse("").is_empty());
        assert!(parse("<happy>").is_empty());
    }

    /// Mode::from_tag returns None for unknown tags.
    #[test]
    fn from_tag_unknown() {
        assert_eq!(Mode::from_tag("xyz"), None);
        assert_eq!(Mode::from_tag("angry"), Some(Mode::Mad));
    }
}
