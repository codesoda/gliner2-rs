//! Boundary-model text preparation matching pinned GLiNER2 preprocessing.
//!
//! This is intentionally independent from the legacy span-model preprocessor.
//! Boundary inference normalizes terminal punctuation before word splitting,
//! caps only the split text words, and prepends any choice prefix afterwards.

use std::{error::Error, fmt, num::NonZeroUsize, str::FromStr};

use once_cell::sync::Lazy;
use regex::Regex;

pub const DEFAULT_MAX_LEN: usize = 4096;

static PYTHON_WORD_CHAR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[\p{Letter}\p{Number}_]$").expect("valid Unicode category regex"));

/// Built-in upstream word splitters supported by boundary checkpoints.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WordSplitter {
    #[default]
    Whitespace,
    Char,
}

impl FromStr for WordSplitter {
    type Err = PreprocessingError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "whitespace" => Ok(Self::Whitespace),
            "char" => Ok(Self::Char),
            _ => Err(PreprocessingError::UnknownSplitter(value.to_owned())),
        }
    }
}

/// Validation failures for boundary preprocessing configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreprocessingError {
    ZeroMaxLen,
    UnknownSplitter(String),
}

impl fmt::Display for PreprocessingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxLen => formatter.write_str("boundary max_len must be positive"),
            Self::UnknownSplitter(value) => write!(
                formatter,
                "unknown boundary word splitter {value:?}; expected \"whitespace\" or \"char\""
            ),
        }
    }
}

impl Error for PreprocessingError {}

/// Boundary-only preprocessing settings.
///
/// Unlike the legacy span path, boundary preprocessing always has a positive
/// word cap. The checkpoint config defaults to 4096 (upstream public calls can
/// separately opt out of truncation; this Rust policy follows the config).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryPreprocessingPolicy {
    max_len: NonZeroUsize,
    splitter: WordSplitter,
}

impl Default for BoundaryPreprocessingPolicy {
    fn default() -> Self {
        Self {
            max_len: NonZeroUsize::new(DEFAULT_MAX_LEN).expect("4096 is non-zero"),
            splitter: WordSplitter::Whitespace,
        }
    }
}

impl BoundaryPreprocessingPolicy {
    pub fn new(max_len: usize, splitter: WordSplitter) -> Result<Self, PreprocessingError> {
        let max_len = NonZeroUsize::new(max_len).ok_or(PreprocessingError::ZeroMaxLen)?;
        Ok(Self { max_len, splitter })
    }

    /// Resolve an omitted model setting to the pinned checkpoint default.
    pub fn from_optional_max_len(
        max_len: Option<usize>,
        splitter: WordSplitter,
    ) -> Result<Self, PreprocessingError> {
        Self::new(max_len.unwrap_or(DEFAULT_MAX_LEN), splitter)
    }

    pub const fn max_len(self) -> usize {
        self.max_len.get()
    }

    pub const fn splitter(self) -> WordSplitter {
        self.splitter
    }

    /// Prepare encoder tokens and decoder-safe original-text mappings.
    ///
    /// `choice_prefix` is copied verbatim after original words have been
    /// capped. Thus the full encoder sequence may exceed `max_len`.
    pub fn prepare(self, text: &str, choice_prefix: &[String]) -> PreparedTokens {
        let (normalized_text, synthetic_suffix_added) = normalize_terminal_punctuation(text);
        let split = split_with_offsets(&normalized_text, self.splitter);
        let split_words = split.len();
        let retained = split.into_iter().take(self.max_len()).collect::<Vec<_>>();
        let truncated_words = split_words - retained.len();

        let mut text_tokens = Vec::with_capacity(choice_prefix.len() + retained.len());
        text_tokens.extend(choice_prefix.iter().cloned());
        text_tokens.extend(retained.iter().map(|token| token.value.clone()));

        let original_offsets = retained
            .iter()
            .map(|token| OriginalTokenOffset::from_normalized(token, text.len()))
            .collect();

        PreparedTokens {
            original_text: text.to_owned(),
            normalized_text,
            text_tokens,
            original_offsets,
            choice_prefix_words: choice_prefix.len(),
            synthetic_suffix_added,
            truncated_words,
        }
    }
}

/// How a normalized token relates to the caller's original text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OriginalCoverage {
    /// Every byte represented by the token came from the caller's text.
    Original,
    /// The token contains original bytes and absorbed the appended full stop.
    OriginalWithSyntheticSuffix,
    /// The token is the appended full stop and has no original byte range.
    SyntheticOnly,
}

/// A normalized text token's safe half-open UTF-8 range in the caller's text.
///
/// `end` is always at most `original_text.len()`. Synthetic-only punctuation
/// is represented by a zero-width range at the end of the original text; the
/// coverage marker lets decoding explicitly reject it. A URL can absorb the
/// appended full stop into the same encoder token, in which case its range is
/// clamped to the original URL and marked `OriginalWithSyntheticSuffix`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OriginalTokenOffset {
    pub start: usize,
    pub end: usize,
    pub coverage: OriginalCoverage,
}

impl OriginalTokenOffset {
    fn from_normalized(token: &SplitToken, original_len: usize) -> Self {
        let start = token.start.min(original_len);
        let end = token.end.min(original_len);
        let coverage = if token.start >= original_len {
            OriginalCoverage::SyntheticOnly
        } else if token.end > original_len {
            OriginalCoverage::OriginalWithSyntheticSuffix
        } else {
            OriginalCoverage::Original
        };
        Self {
            start,
            end,
            coverage,
        }
    }

    /// Whether this token contributes at least one original source byte.
    pub const fn has_original_text(self) -> bool {
        self.start < self.end
    }

    pub const fn contains_synthetic_suffix(self) -> bool {
        matches!(
            self.coverage,
            OriginalCoverage::OriginalWithSyntheticSuffix | OriginalCoverage::SyntheticOnly
        )
    }
}

/// Complete prepared boundary input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTokens {
    /// Exact caller input, retained separately from encoder normalization.
    pub original_text: String,
    /// Text seen by the upstream splitter (possibly with one appended `.`).
    pub normalized_text: String,
    /// Full encoder word sequence: prefix followed by capped normalized words.
    pub text_tokens: Vec<String>,
    /// Prefix-free mappings, one per retained normalized text token.
    pub original_offsets: Vec<OriginalTokenOffset>,
    pub choice_prefix_words: usize,
    pub synthetic_suffix_added: bool,
    /// Normalized words dropped by the `max_len` cap. Zero means the whole
    /// text reached the encoder.
    pub truncated_words: usize,
}

impl PreparedTokens {
    /// Start mappings suitable for a UTF-8 decoder after prefix subtraction.
    pub fn start_mappings(&self) -> Vec<usize> {
        self.original_offsets
            .iter()
            .map(|offset| offset.start)
            .collect()
    }

    /// End mappings suitable for a UTF-8 decoder after prefix subtraction.
    pub fn end_mappings(&self) -> Vec<usize> {
        self.original_offsets
            .iter()
            .map(|offset| offset.end)
            .collect()
    }

    pub fn original_token_count(&self) -> usize {
        self.original_offsets.len()
    }
}

#[derive(Clone, Debug)]
struct SplitToken {
    value: String,
    start: usize,
    end: usize,
}

fn normalize_terminal_punctuation(text: &str) -> (String, bool) {
    if text.ends_with(['.', '!', '?']) {
        (text.to_owned(), false)
    } else if text.is_empty() {
        (".".to_owned(), true)
    } else {
        let mut normalized = String::with_capacity(text.len() + 1);
        normalized.push_str(text);
        normalized.push('.');
        (normalized, true)
    }
}

fn split_with_offsets(text: &str, splitter: WordSplitter) -> Vec<SplitToken> {
    let mut output = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let ch = text[cursor..].chars().next().expect("cursor is in text");
        if python_whitespace(ch) {
            cursor += ch.len_utf8();
            continue;
        }
        let end = match splitter {
            WordSplitter::Whitespace => whitespace_match_end(text, cursor),
            WordSplitter::Char => char_match_end(text, cursor),
        };
        let source = &text[cursor..end];
        output.push(SplitToken {
            value: source.to_lowercase(),
            start: cursor,
            end,
        });
        cursor = end;
    }
    output
}

fn whitespace_match_end(text: &str, start: usize) -> usize {
    match_url_end(text, start)
        .or_else(|| match_email_end(text, start))
        .or_else(|| match_handle_end(text, start))
        .or_else(|| match_python_word_end(text, start))
        .unwrap_or_else(|| next_char_end(text, start))
}

fn char_match_end(text: &str, start: usize) -> usize {
    let mut end = start;
    for (relative, ch) in text[start..].char_indices() {
        if !ch.is_ascii_alphanumeric() && !matches!(ch, '@' | '.' | '_' | '-' | '+') {
            break;
        }
        end = start + relative + ch.len_utf8();
    }
    if end == start {
        next_char_end(text, start)
    } else {
        end
    }
}

fn match_url_end(text: &str, start: usize) -> Option<usize> {
    let after_prefix = match_ci_literal(text, start, "http")
        .and_then(|position| {
            let (ch, after) = next_char(text, position)?;
            if python_ascii_case_eq(ch, 's') {
                Some(after)
            } else {
                Some(position)
            }
        })
        .and_then(|position| match_literal(text, position, "://"))
        .or_else(|| {
            match_ci_literal(text, start, "www")
                .and_then(|position| match_literal(text, position, "."))
        })?;

    let mut end = after_prefix;
    for (relative, ch) in text[after_prefix..].char_indices() {
        if python_whitespace(ch) {
            break;
        }
        end = after_prefix + relative + ch.len_utf8();
    }
    (end > after_prefix).then_some(end)
}

fn match_email_end(text: &str, start: usize) -> Option<usize> {
    let local_end = consume_while(text, start, email_local_char);
    if local_end == start || !text[local_end..].starts_with('@') {
        return None;
    }
    let domain_start = local_end + 1;
    let domain_run_end = consume_while(text, domain_start, email_domain_char);
    if domain_run_end == domain_start {
        return None;
    }

    // `[a-z0-9.-]+` is greedy, then backtracks to the rightmost dot from
    // which `[a-z]{2,}` can match. The TLD itself need not reach a word
    // boundary (for example, `x@y.com2` matches through `com`).
    let dots = text[domain_start..domain_run_end]
        .char_indices()
        .filter_map(|(relative, ch)| (ch == '.').then_some(domain_start + relative))
        .collect::<Vec<_>>();
    for dot in dots.into_iter().rev() {
        if dot == domain_start {
            continue;
        }
        let tld_start = dot + 1;
        let tld_end = consume_while(text, tld_start, python_ascii_letter);
        if text[tld_start..tld_end].chars().count() >= 2 {
            return Some(tld_end);
        }
    }
    None
}

fn match_handle_end(text: &str, start: usize) -> Option<usize> {
    if !text[start..].starts_with('@') {
        return None;
    }
    let body_start = start + 1;
    let end = consume_while(text, body_start, |ch| {
        python_ascii_letter(ch) || ch.is_ascii_digit() || ch == '_'
    });
    (end > body_start).then_some(end)
}

fn match_python_word_end(text: &str, start: usize) -> Option<usize> {
    let mut end = consume_while(text, start, python_word_char);
    if end == start {
        return None;
    }
    while let Some((separator, after_separator)) = next_char(text, end) {
        if !matches!(separator, '-' | '_') {
            break;
        }
        let after_word = consume_while(text, after_separator, python_word_char);
        if after_word == after_separator {
            break;
        }
        end = after_word;
    }
    Some(end)
}

fn consume_while(text: &str, start: usize, predicate: impl Fn(char) -> bool) -> usize {
    let mut end = start;
    for (relative, ch) in text[start..].char_indices() {
        if !predicate(ch) {
            break;
        }
        end = start + relative + ch.len_utf8();
    }
    end
}

fn next_char(text: &str, position: usize) -> Option<(char, usize)> {
    let ch = text.get(position..)?.chars().next()?;
    Some((ch, position + ch.len_utf8()))
}

fn next_char_end(text: &str, start: usize) -> usize {
    next_char(text, start)
        .expect("start points at a character")
        .1
}

fn match_literal(text: &str, mut position: usize, expected: &str) -> Option<usize> {
    for expected_char in expected.chars() {
        let (actual, after) = next_char(text, position)?;
        if actual != expected_char {
            return None;
        }
        position = after;
    }
    Some(position)
}

fn match_ci_literal(text: &str, mut position: usize, expected: &str) -> Option<usize> {
    for expected_char in expected.chars() {
        let (actual, after) = next_char(text, position)?;
        if !python_ascii_case_eq(actual, expected_char) {
            return None;
        }
        position = after;
    }
    Some(position)
}

fn python_ascii_case_eq(actual: char, expected_lowercase: char) -> bool {
    actual.eq_ignore_ascii_case(&expected_lowercase)
        || matches!(
            (expected_lowercase, actual),
            ('i', 'İ' | 'ı') | ('s', 'ſ') | ('k', 'K')
        )
}

fn python_ascii_letter(ch: char) -> bool {
    ch.is_ascii_alphabetic() || matches!(ch, 'İ' | 'ı' | 'ſ' | 'K')
}

fn email_local_char(ch: char) -> bool {
    python_ascii_letter(ch) || ch.is_ascii_digit() || matches!(ch, '.' | '_' | '%' | '+' | '-')
}

fn email_domain_char(ch: char) -> bool {
    python_ascii_letter(ch) || ch.is_ascii_digit() || matches!(ch, '.' | '-')
}

fn python_word_char(ch: char) -> bool {
    let mut encoded = [0; 4];
    PYTHON_WORD_CHAR.is_match(ch.encode_utf8(&mut encoded))
}

fn python_whitespace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{001c}'..='\u{001f}')
}
