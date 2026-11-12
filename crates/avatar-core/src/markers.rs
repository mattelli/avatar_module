//! Marker stripping (report §3.1, plan task 7).
//!
//! Expression markers (`<happy>`, `<sad>`, ...) are **removed** from the
//! pipeline: expression is fully audio-driven by ExpNet (Plan 5). The parser
//! still consumes `<tag>` tokens (so legacy marked-up text renders as a
//! single neutral clip), but no longer switches expression modes.
//!
//! The [`Mode`] enum is kept because the expression atlas
//! ([`crate::atlas`]) still defines 7 warp bands — the pipeline just never
//! switches them per marker anymore.

/// Expression modes supported by the renderer (atlas bands).
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

/// A span of text. Expression is audio-driven, so every span is neutral.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// Text of the segment (markers stripped, whitespace preserved).
    pub text: String,
    /// Expression mode active for this segment (always [`Mode::Neutral`]).
    pub mode: Mode,
}

/// Parses `text` into a single neutral span with all `<tag>` markers stripped.
///
/// - `<tag>` tokens are consumed as markers and dropped.
/// - Unknown tags (e.g. `<xyz>`) are dropped silently too.
/// - An unclosed `<` at the end of the text is treated as literal text.
/// - Empty text (or text that is only markers) yields no span.
pub fn parse(text: &str) -> Vec<Span> {
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
            if !closed {
                // Unclosed '<' — literal text.
                buf.push('<');
                buf.push_str(&tag);
            }
            // Closed tag: dropped silently (marker removal).
        } else {
            buf.push(c);
        }
    }

    if buf.trim().is_empty() {
        Vec::new()
    } else {
        vec![Span {
            text: buf,
            mode: Mode::Neutral,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T7: `parse("Hello <happy> world <sad>")` yields a single neutral span.
    #[test]
    fn markers_stripped_single_neutral_span() {
        let spans = parse("Hello <happy> world <sad>");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "Hello  world ");
        assert_eq!(spans[0].mode, Mode::Neutral);
    }

    /// No markers → a single neutral span with the text unchanged.
    #[test]
    fn no_markers_neutral() {
        let spans = parse("Hello world");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "Hello world");
        assert_eq!(spans[0].mode, Mode::Neutral);
    }

    /// Unknown tags are also stripped (all `<...>` are markers now).
    #[test]
    fn unknown_marker_stripped() {
        let spans = parse("a <xyz> b");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "a  b");
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
        assert!(parse("<happy><sad>").is_empty());
    }

    /// Mode::from_tag still works (atlas uses it).
    #[test]
    fn from_tag_unknown() {
        assert_eq!(Mode::from_tag("xyz"), None);
        assert_eq!(Mode::from_tag("angry"), Some(Mode::Mad));
    }
}
