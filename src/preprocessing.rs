use crate::text::{TokenSpan, tokenize_with_offsets};

/// Shared word-level preprocessing policy used before encoder formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreprocessingPolicy {
    max_len: Option<usize>,
}

impl PreprocessingPolicy {
    pub const fn new(max_len: Option<usize>) -> Self {
        Self { max_len }
    }

    pub const fn max_len(self) -> Option<usize> {
        self.max_len
    }

    /// Tokenize text and cap tokens and byte offsets together.
    pub fn tokenize(self, text: &str, lower: bool) -> Vec<TokenSpan> {
        let mut spans = tokenize_with_offsets(text, lower);
        if let Some(max_len) = self.max_len {
            spans.truncate(max_len);
        }
        spans
    }

    /// Cap caller-provided original text tokens for raw inference.
    pub fn truncate_tokens(self, tokens: &[String]) -> &[String] {
        match self.max_len {
            Some(max_len) => &tokens[..tokens.len().min(max_len)],
            None => tokens,
        }
    }

    /// Prepend structural choice tokens after the original words have already
    /// been capped. Prefix tokens are never included in the word limit.
    pub fn prepend_prefix(self, prefix: &[String], original: &[String]) -> Vec<String> {
        let original = self.truncate_tokens(original);
        let mut combined = Vec::with_capacity(prefix.len() + original.len());
        combined.extend_from_slice(prefix);
        combined.extend_from_slice(original);
        combined
    }
}
