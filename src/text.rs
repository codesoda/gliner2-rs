use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub token: String,
    pub start: usize,
    pub end: usize,
}

static TOKEN_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?xi)
        (?:https?://[^\s]+|www\.[^\s]+)
        |
        [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |
        @[a-z0-9_]+
        |
        \w+(?:[-_]\w+)*
        |
        \S
        ",
    )
    .expect("invalid token splitter regex")
});

/// Port of `gliner2.processor.WhitespaceTokenSplitter`.
///
/// Returns `(token, start, end)` where `start/end` are byte offsets into `text`.
pub fn tokenize_with_offsets(text: &str, lower: bool) -> Vec<TokenSpan> {
    TOKEN_PATTERN
        .find_iter(text)
        .map(|m| TokenSpan {
            token: if lower {
                m.as_str().to_lowercase()
            } else {
                m.as_str().to_string()
            },
            start: m.start(),
            end: m.end(),
        })
        .collect()
}
