//! Public result types and model-free byte-to-word mapping for explicit spans.

use anyhow::{Context, Result, bail, ensure};

use super::preprocessing::{OriginalCoverage, PreparedTokens};

/// A score for one caller-supplied span.
///
/// `start` and `end` are half-open UTF-8 byte offsets into the original input
/// text. `confidence` is sigmoid-calibrated from `logit`; it is not normalized
/// across labels or spans.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitSpanScore {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub logit: f32,
    pub confidence: f32,
}

/// Explicit scores for one label query.
///
/// Groups are returned in caller label order and `spans` are returned in caller
/// span order, including duplicates. Explicit scoring performs no thresholding,
/// abstention, overlap handling, or deduplication.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitSpanScores {
    pub label: String,
    pub spans: Vec<ExplicitSpanScore>,
}

/// Map original-text UTF-8 byte spans to half-open encoder word boundaries.
///
/// Every requested endpoint must exactly match a retained original-word
/// endpoint. The choice prefix is included in returned coordinates. No endpoint
/// is derived from normalized text, and no request is snapped or trimmed.
pub(crate) fn map_byte_spans(
    prepared: &PreparedTokens,
    spans: &[[usize; 2]],
) -> Result<Vec<[i64; 2]>> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }

    validate_prepared(prepared)?;

    let mut mapped = Vec::with_capacity(spans.len());
    for (span_index, &[start, end]) in spans.iter().enumerate() {
        if start >= end {
            bail!("explicit span {span_index} [{start}, {end}) must be nonempty with start < end");
        }
        if start > prepared.original_text.len() || end > prepared.original_text.len() {
            bail!(
                "explicit span {span_index} [{start}, {end}) is out of range for original text of {} bytes",
                prepared.original_text.len()
            );
        }
        if !prepared.original_text.is_char_boundary(start)
            || !prepared.original_text.is_char_boundary(end)
        {
            bail!(
                "explicit span {span_index} [{start}, {end}) does not use UTF-8 character boundaries"
            );
        }

        let start_word = prepared
            .original_offsets
            .iter()
            .position(|offset| offset.has_original_text() && offset.start == start);
        let end_word = prepared
            .original_offsets
            .iter()
            .position(|offset| offset.has_original_text() && offset.end == end);
        let (Some(start_word), Some(end_word)) = (start_word, end_word) else {
            bail!(
                "explicit span {span_index} [{start}, {end}) is not aligned to retained original-word boundaries (it may be whitespace, partial, synthetic, or truncated)"
            );
        };
        if start_word > end_word {
            bail!(
                "explicit span {span_index} [{start}, {end}) is not aligned to an ordered range of retained original words"
            );
        }

        let token_start = prepared
            .choice_prefix_words
            .checked_add(start_word)
            .with_context(|| format!("explicit span {span_index} start token index overflow"))?;
        let word_end_boundary = end_word
            .checked_add(1)
            .with_context(|| format!("explicit span {span_index} end word boundary overflow"))?;
        let token_end = prepared
            .choice_prefix_words
            .checked_add(word_end_boundary)
            .with_context(|| format!("explicit span {span_index} end token index overflow"))?;
        mapped.push([
            i64::try_from(token_start)
                .with_context(|| format!("explicit span {span_index} start exceeds i64"))?,
            i64::try_from(token_end)
                .with_context(|| format!("explicit span {span_index} end exceeds i64"))?,
        ]);
    }
    Ok(mapped)
}

fn validate_prepared(prepared: &PreparedTokens) -> Result<()> {
    let expected_tokens = prepared
        .choice_prefix_words
        .checked_add(prepared.original_offsets.len())
        .context("prepared token layout length overflow")?;
    ensure!(
        prepared.text_tokens.len() == expected_tokens,
        "inconsistent prepared token layout: text_tokens has length {}, expected choice prefix {} + original offsets {} = {expected_tokens}",
        prepared.text_tokens.len(),
        prepared.choice_prefix_words,
        prepared.original_offsets.len()
    );

    let original_len = prepared.original_text.len();
    let mut previous_end = 0;
    for (offset_index, offset) in prepared.original_offsets.iter().enumerate() {
        ensure!(
            offset.start <= offset.end && offset.end <= original_len,
            "prepared original offset {offset_index} [{}, {}) is out of range for original text of {original_len} bytes",
            offset.start,
            offset.end
        );
        ensure!(
            prepared.original_text.is_char_boundary(offset.start)
                && prepared.original_text.is_char_boundary(offset.end),
            "prepared original offset {offset_index} [{}, {}) does not use UTF-8 character boundaries",
            offset.start,
            offset.end
        );
        ensure!(
            offset.start >= previous_end,
            "prepared original offsets are not monotonic at index {offset_index}: start {} precedes prior end {previous_end}",
            offset.start
        );

        match offset.coverage {
            OriginalCoverage::Original => ensure!(
                offset.has_original_text(),
                "prepared original offset {offset_index} marked Original is empty"
            ),
            OriginalCoverage::OriginalWithSyntheticSuffix => ensure!(
                offset.has_original_text()
                    && offset.end == original_len
                    && prepared.synthetic_suffix_added,
                "prepared original offset {offset_index} has inconsistent OriginalWithSyntheticSuffix coverage"
            ),
            OriginalCoverage::SyntheticOnly => ensure!(
                offset.start == original_len
                    && offset.end == original_len
                    && prepared.synthetic_suffix_added,
                "prepared original offset {offset_index} has inconsistent SyntheticOnly coverage"
            ),
        }
        previous_end = offset.end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::preprocessing::{
        BoundaryPreprocessingPolicy, OriginalTokenOffset, WordSplitter,
    };

    fn prepare(text: &str, max_len: usize, splitter: WordSplitter) -> PreparedTokens {
        BoundaryPreprocessingPolicy::new(max_len, splitter)
            .unwrap()
            .prepare(text, &[])
    }

    fn error(prepared: &PreparedTokens, spans: &[[usize; 2]]) -> String {
        map_byte_spans(prepared, spans).unwrap_err().to_string()
    }

    #[test]
    fn maps_ascii_words_and_multiword_ranges() {
        let prepared = prepare("one two three", 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[[0, 3], [4, 7], [0, 13], [4, 13]]).unwrap(),
            vec![[0, 1], [1, 2], [0, 3], [1, 3]]
        );
    }

    #[test]
    fn preserves_unsorted_duplicate_requests() {
        let prepared = prepare("one two three", 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[[8, 13], [0, 3], [8, 13], [4, 7]]).unwrap(),
            vec![[2, 3], [0, 1], [2, 3], [1, 2]]
        );
    }

    #[test]
    fn maps_unicode_with_whitespace_splitter() {
        let text = "γειά 😀 世界";
        let prepared = prepare(text, 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[[0, 8], [9, 13], [14, 20], [0, 20]]).unwrap(),
            vec![[0, 1], [1, 2], [2, 3], [0, 3]]
        );
    }

    #[test]
    fn maps_unicode_with_char_splitter() {
        let text = "α😀中 文";
        let prepared = prepare(text, 20, WordSplitter::Char);
        assert_eq!(
            map_byte_spans(&prepared, &[[0, 9], [10, 13], [2, 6]]).unwrap(),
            vec![[0, 3], [3, 4], [1, 2]]
        );
    }

    #[test]
    fn maps_caller_trimmed_literal_newline_bounds() {
        let text = "\nAlice\nBob\n";
        let prepared = prepare(text, 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[[1, 6], [7, 10], [1, 10]]).unwrap(),
            vec![[0, 1], [1, 2], [0, 2]]
        );
        assert!(error(&prepared, &[[0, 6]]).contains("not aligned"));
        assert!(error(&prepared, &[[1, 11]]).contains("not aligned"));
    }

    #[test]
    fn rejects_invalid_requested_ranges_without_panicking() {
        let prepared = prepare("café next", 20, WordSplitter::Whitespace);

        assert!(error(&prepared, &[[1, 5]]).contains("not aligned"));
        assert!(error(&prepared, &[[0, 4]]).contains("UTF-8"));
        assert!(error(&prepared, &[[5, 5]]).contains("nonempty"));
        assert!(error(&prepared, &[[5, 0]]).contains("nonempty"));
        assert!(error(&prepared, &[[0, 100]]).contains("out of range"));
    }

    #[test]
    fn rejects_words_removed_by_max_len_truncation() {
        let prepared = prepare("one two three", 2, WordSplitter::Whitespace);
        assert_eq!(map_byte_spans(&prepared, &[[0, 7]]).unwrap(), vec![[0, 2]]);
        assert!(error(&prepared, &[[8, 13]]).contains("retained"));
        assert!(error(&prepared, &[[0, 13]]).contains("retained"));
    }

    #[test]
    fn shifts_for_choice_prefix_including_verbatim_multiword_entries() {
        let prefix = vec!["[P]".to_owned(), "new york".to_owned()];
        let prepared = BoundaryPreprocessingPolicy::default().prepare("one two", &prefix);
        assert_eq!(
            prepared.text_tokens[..2],
            ["[P]".to_owned(), "new york".to_owned()]
        );
        assert_eq!(
            map_byte_spans(&prepared, &[[0, 3], [4, 7], [0, 7]]).unwrap(),
            vec![[2, 3], [3, 4], [2, 4]]
        );
    }

    #[test]
    fn empty_requests_bypass_even_for_synthetic_only_text() {
        let prepared = prepare("", 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[]).unwrap(),
            Vec::<[i64; 2]>::new()
        );
        assert!(error(&prepared, &[[0, 0]]).contains("nonempty"));
    }

    #[test]
    fn maps_original_punctuation_but_never_synthetic_punctuation() {
        let prepared = prepare("Hello, world!", 20, WordSplitter::Whitespace);
        assert_eq!(
            map_byte_spans(&prepared, &[[0, 6], [7, 13]]).unwrap(),
            vec![[0, 2], [2, 4]]
        );

        let trailing_space = prepare("hello ", 20, WordSplitter::Whitespace);
        assert!(error(&trailing_space, &[[0, 6]]).contains("not aligned"));
    }

    #[test]
    fn accepts_original_part_of_url_token_with_absorbed_synthetic_suffix() {
        let text = "https://example.com/path";
        let prepared = prepare(text, 20, WordSplitter::Whitespace);
        assert_eq!(
            prepared.original_offsets[0].coverage,
            OriginalCoverage::OriginalWithSyntheticSuffix
        );
        assert_eq!(
            map_byte_spans(&prepared, &[[0, text.len()]]).unwrap(),
            vec![[0, 1]]
        );
    }

    #[test]
    fn rejects_inconsistent_prepared_layout() {
        let mut prepared = prepare("one", 20, WordSplitter::Whitespace);
        prepared.text_tokens.pop();
        assert!(error(&prepared, &[[0, 3]]).contains("inconsistent prepared token layout"));
    }

    #[test]
    fn rejects_bad_prepared_offsets() {
        let mut out_of_bounds = prepare("é", 20, WordSplitter::Whitespace);
        out_of_bounds.original_offsets[0].end = 3;
        assert!(error(&out_of_bounds, &[[0, 2]]).contains("out of range"));

        let mut invalid_utf8 = prepare("é", 20, WordSplitter::Whitespace);
        invalid_utf8.original_offsets[0].end = 1;
        assert!(error(&invalid_utf8, &[[0, 2]]).contains("UTF-8"));

        let mut non_monotonic = prepare("one two", 20, WordSplitter::Whitespace);
        non_monotonic.original_offsets[1] = OriginalTokenOffset {
            start: 2,
            end: 6,
            coverage: OriginalCoverage::Original,
        };
        assert!(error(&non_monotonic, &[[0, 3]]).contains("not monotonic"));
    }
}
