//! Pure Rust decoding helpers for GLiNER2.5 boundary candidates.
//!
//! This module deliberately has no model or pipeline dependency. It accepts
//! scorer outputs, applies the pinned upstream threshold/abstention and overlap
//! rules, and maps half-open word spans back to UTF-8 byte offsets.

use std::{cmp::Ordering, str::FromStr};

use anyhow::{Context, Result, bail, ensure};

/// One thresholded, half-open token span.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredSpan {
    pub confidence: f32,
    pub start: usize,
    pub end: usize,
}

/// Scorer inputs for one query. All candidate vectors must have equal length.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryScores {
    pub indices: Vec<[usize; 2]>,
    pub valid_mask: Vec<bool>,
    pub pair_logits: Vec<f32>,
    pub query_valid: bool,
    pub threshold: f32,
    pub null_logit: Option<f32>,
}

/// A word's half-open UTF-8 byte range in the caller's original text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WordOffset {
    pub start: usize,
    pub end: usize,
}

/// A decoded span. Byte boundaries continue to cover the untrimmed source
/// slice, while `text` matches upstream's trimmed surface string.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedSpan {
    pub confidence: f32,
    pub token_start: usize,
    pub token_end: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldDtype {
    List,
    Scalar,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodedValue {
    List(Vec<DecodedSpan>),
    Scalar(Option<DecodedSpan>),
}

/// Canonical overlap policies from `gliner2.inference.overlap`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OverlapPolicy {
    Allow,
    Nested,
    Disallow,
    Longest,
}

impl FromStr for OverlapPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let key = value.trim().to_ascii_lowercase().replace('-', "_");
        match key.as_str() {
            "allow" | "all" | "none" => Ok(Self::Allow),
            "nested" | "allow_nested" => Ok(Self::Nested),
            "flat" | "disallow" | "no_overlap" | "non_overlapping" => Ok(Self::Disallow),
            "longest" | "keep_longest" => Ok(Self::Longest),
            _ => bail!(
                "unknown overlap policy {value:?}; expected allow, nested, flat/disallow, or longest"
            ),
        }
    }
}

/// Resolve an optional policy through an architecture-specific default.
pub fn normalize_overlap_policy(
    policy: Option<&str>,
    default: Option<&str>,
) -> Result<OverlapPolicy> {
    let selected = policy.or(default).ok_or_else(|| {
        anyhow::anyhow!("overlap policy is absent and no architecture default was provided")
    })?;
    selected.parse()
}

fn validate_probability(value: f32, name: &str) -> Result<()> {
    ensure!(value.is_finite(), "{name} must be finite, got {value}");
    ensure!(
        (0.0..=1.0).contains(&value),
        "{name} must be in [0,1], got {value}"
    );
    Ok(())
}

/// Float32 sigmoid after float32 temperature scaling, as used by the scorer.
pub fn sigmoid_probability(logit: f32, temperature: f32) -> Result<f32> {
    ensure!(logit.is_finite(), "logit must be finite, got {logit}");
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be finite and positive, got {temperature}"
    );
    let scaled = logit / temperature;
    let probability = if scaled >= 0.0 {
        1.0 / (1.0 + (-scaled).exp())
    } else {
        let exponential = scaled.exp();
        exponential / (1.0 + exponential)
    };
    ensure!(
        probability.is_finite(),
        "sigmoid produced a non-finite probability from logit {logit} and temperature {temperature}"
    );
    Ok(probability)
}

/// Confidence gating is inclusive: equality to the threshold is retained.
pub fn passes_confidence_threshold(probability: f32, threshold: f32) -> Result<bool> {
    validate_probability(probability, "probability")?;
    validate_probability(threshold, "confidence threshold")?;
    Ok(probability >= threshold)
}

/// Null abstention is strict: equality to the threshold does not abstain.
pub fn should_abstain(null_probability: f32, threshold: f32) -> Result<bool> {
    validate_probability(null_probability, "null probability")?;
    validate_probability(threshold, "abstention threshold")?;
    Ok(null_probability > threshold)
}

/// Threshold scorer outputs for each query while preserving candidate order.
///
/// Count-head output is intentionally absent: supported checkpoints have
/// adaptive thresholding disabled, so normal decoding must not consult it.
pub fn group_scored_candidates(
    queries: &[QueryScores],
    pair_temperature: f32,
    abstention_threshold: f32,
) -> Result<Vec<Vec<ScoredSpan>>> {
    ensure!(
        pair_temperature.is_finite() && pair_temperature > 0.0,
        "pair temperature must be finite and positive, got {pair_temperature}"
    );
    validate_probability(abstention_threshold, "abstention threshold")?;

    queries
        .iter()
        .enumerate()
        .map(|(query_index, query)| {
            validate_probability(query.threshold, "query confidence threshold")
                .with_context(|| format!("query {query_index}"))?;
            let candidate_count = query.indices.len();
            ensure!(
                query.valid_mask.len() == candidate_count,
                "query {query_index}: valid_mask length {} differs from indices length {candidate_count}",
                query.valid_mask.len()
            );
            ensure!(
                query.pair_logits.len() == candidate_count,
                "query {query_index}: pair_logits length {} differs from indices length {candidate_count}",
                query.pair_logits.len()
            );

            if let Some(null_logit) = query.null_logit {
                let null_probability = sigmoid_probability(null_logit, 1.0)
                    .with_context(|| format!("query {query_index} null logit"))?;
                if should_abstain(null_probability, abstention_threshold)? {
                    return Ok(Vec::new());
                }
            }
            if !query.query_valid {
                return Ok(Vec::new());
            }

            let mut retained = Vec::new();
            for candidate_index in 0..candidate_count {
                let logit = query.pair_logits[candidate_index];
                ensure!(
                    logit.is_finite(),
                    "query {query_index} candidate {candidate_index}: pair logit must be finite, got {logit}"
                );
                if !query.valid_mask[candidate_index] {
                    continue;
                }
                let [start, end] = query.indices[candidate_index];
                ensure!(
                    start < end,
                    "query {query_index} candidate {candidate_index}: malformed half-open span [{start},{end})"
                );
                let confidence = sigmoid_probability(logit, pair_temperature).with_context(|| {
                    format!("query {query_index} candidate {candidate_index}")
                })?;
                if passes_confidence_threshold(confidence, query.threshold)? {
                    retained.push(ScoredSpan {
                        confidence,
                        start,
                        end,
                    });
                }
            }
            Ok(retained)
        })
        .collect()
}

fn rank_cmp(
    left_index: usize,
    left: &ScoredSpan,
    right_index: usize,
    right: &ScoredSpan,
) -> Ordering {
    // Values are finite, so partial comparison matches Python float ordering
    // (including treating -0.0 and +0.0 as equal).
    right
        .confidence
        .partial_cmp(&left.confidence)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.start.cmp(&right.start))
        .then_with(|| left.end.cmp(&right.end))
        .then_with(|| left_index.cmp(&right_index))
}

fn validate_spans(spans: &[ScoredSpan]) -> Result<()> {
    for (index, span) in spans.iter().enumerate() {
        validate_probability(span.confidence, "span confidence")
            .with_context(|| format!("span {index}"))?;
        ensure!(
            span.start < span.end,
            "span {index} has malformed half-open bounds [{},{})",
            span.start,
            span.end
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct IndexedSpan<'a> {
    input_index: usize,
    span: &'a ScoredSpan,
}

fn indexed_rank_cmp(left: &IndexedSpan<'_>, right: &IndexedSpan<'_>) -> Ordering {
    rank_cmp(left.input_index, left.span, right.input_index, right.span)
}

fn overlaps(left: &ScoredSpan, right: &ScoredSpan) -> bool {
    left.start < right.end && right.start < left.end
}

fn contains(left: &ScoredSpan, right: &ScoredSpan) -> bool {
    left.start <= right.start && right.end <= left.end
}

fn selection_cmp(left: &[usize], right: &[usize], by_end: &[IndexedSpan<'_>]) -> Ordering {
    let mut left_rows: Vec<_> = left.iter().map(|&index| by_end[index]).collect();
    let mut right_rows: Vec<_> = right.iter().map(|&index| by_end[index]).collect();
    left_rows.sort_by(indexed_rank_cmp);
    right_rows.sort_by(indexed_rank_cmp);
    for (left_row, right_row) in left_rows.iter().zip(&right_rows) {
        let ordering = indexed_rank_cmp(left_row, right_row);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left_rows.len().cmp(&right_rows.len())
}

/// Deduplicate exact coordinates and apply an upstream overlap policy.
///
/// `Disallow` uses weighted interval scheduling. Its sums are f64 additions of
/// f32 confidences, matching Python's `float(score)` arithmetic. Exact sum ties
/// prefer larger cardinality, then lexicographically better global ranking.
pub fn resolve_overlaps(spans: &[ScoredSpan], policy: OverlapPolicy) -> Result<Vec<ScoredSpan>> {
    validate_spans(spans)?;
    if spans.is_empty() {
        return Ok(Vec::new());
    }

    let mut ranked: Vec<_> = spans
        .iter()
        .enumerate()
        .map(|(input_index, span)| IndexedSpan { input_index, span })
        .collect();
    ranked.sort_by(indexed_rank_cmp);
    let mut distinct = Vec::with_capacity(ranked.len());
    for row in ranked {
        if distinct.iter().any(|existing: &IndexedSpan<'_>| {
            existing.span.start == row.span.start && existing.span.end == row.span.end
        }) {
            continue;
        }
        distinct.push(row);
    }

    let mut selected = match policy {
        OverlapPolicy::Allow => distinct,
        OverlapPolicy::Nested => {
            let mut kept: Vec<IndexedSpan<'_>> = Vec::new();
            for candidate in distinct {
                let crossing = kept.iter().any(|existing| {
                    overlaps(candidate.span, existing.span)
                        && !contains(candidate.span, existing.span)
                        && !contains(existing.span, candidate.span)
                });
                if !crossing {
                    kept.push(candidate);
                }
            }
            kept
        }
        OverlapPolicy::Longest => distinct
            .iter()
            .copied()
            .filter(|candidate| {
                !distinct.iter().any(|other| {
                    (other.span.start < candidate.span.start || candidate.span.end < other.span.end)
                        && contains(other.span, candidate.span)
                })
            })
            .collect(),
        OverlapPolicy::Disallow => {
            let mut by_end = distinct;
            by_end.sort_by(|left, right| {
                left.span
                    .end
                    .cmp(&right.span.end)
                    .then_with(|| left.span.start.cmp(&right.span.start))
                    .then_with(|| {
                        right
                            .span
                            .confidence
                            .partial_cmp(&left.span.confidence)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| left.input_index.cmp(&right.input_index))
            });
            let ends: Vec<_> = by_end.iter().map(|row| row.span.end).collect();
            let predecessors: Vec<_> = by_end
                .iter()
                .enumerate()
                .map(|(index, row)| ends[..index].partition_point(|&end| end <= row.span.start))
                .collect();
            let mut best: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
            for (index, row) in by_end.iter().enumerate() {
                let (previous_score, previous_selection) = &best[predecessors[index]];
                let mut with_selection = previous_selection.clone();
                with_selection.push(index);
                let with_score = *previous_score + f64::from(row.span.confidence);
                let (without_score, without_selection) = &best[index];
                let take_with = match with_score.partial_cmp(without_score) {
                    Some(Ordering::Greater) => true,
                    Some(Ordering::Less) => false,
                    Some(Ordering::Equal) => {
                        match with_selection.len().cmp(&without_selection.len()) {
                            Ordering::Greater => true,
                            Ordering::Less => false,
                            Ordering::Equal => {
                                selection_cmp(&with_selection, without_selection, &by_end)
                                    == Ordering::Less
                            }
                        }
                    }
                    None => unreachable!("validated finite confidences have finite f64 sums"),
                };
                best.push(if take_with {
                    (with_score, with_selection)
                } else {
                    (*without_score, without_selection.clone())
                });
            }
            best.pop()
                .expect("weighted interval table is initialized")
                .1
                .into_iter()
                .map(|index| by_end[index])
                .collect()
        }
    };
    selected.sort_by(indexed_rank_cmp);
    Ok(selected.into_iter().map(|row| row.span.clone()).collect())
}

fn validate_word_offsets(text: &str, offsets: &[WordOffset]) -> Result<()> {
    let mut previous_start = 0;
    let mut previous_end = 0;
    for (index, offset) in offsets.iter().enumerate() {
        ensure!(
            offset.start <= offset.end,
            "word offset {index} is reversed: [{},{})",
            offset.start,
            offset.end
        );
        ensure!(
            offset.end <= text.len(),
            "word offset {index} ends at byte {}, beyond original text length {}",
            offset.end,
            text.len()
        );
        ensure!(
            text.is_char_boundary(offset.start) && text.is_char_boundary(offset.end),
            "word offset {index} [{},{}) is not on UTF-8 character boundaries",
            offset.start,
            offset.end
        );
        if index > 0 {
            ensure!(
                offset.start >= previous_start && offset.end >= previous_end,
                "word offsets are not monotonic at index {index}: [{},{}) follows [{previous_start},{previous_end})",
                offset.start,
                offset.end
            );
        }
        previous_start = offset.start;
        previous_end = offset.end;
    }
    Ok(())
}

/// Convert an in-range half-open word span to original-text UTF-8 bytes.
pub fn map_half_open_utf8(
    text: &str,
    offsets: &[WordOffset],
    start: usize,
    end: usize,
) -> Result<DecodedSpan> {
    validate_word_offsets(text, offsets)?;
    ensure!(
        start < end,
        "half-open token span requires start < end, got [{start},{end})"
    );
    ensure!(
        end <= offsets.len(),
        "token span [{start},{end}) is out of range for {} original words",
        offsets.len()
    );
    let byte_start = offsets[start].start;
    let byte_end = offsets[end - 1].end;
    ensure!(
        byte_start <= byte_end,
        "mapped byte span is reversed: [{byte_start},{byte_end})"
    );
    let surface = text
        .get(byte_start..byte_end)
        .ok_or_else(|| {
            anyhow::anyhow!("mapped bytes [{byte_start},{byte_end}) are not valid UTF-8 boundaries")
        })?
        .trim()
        .to_owned();
    Ok(DecodedSpan {
        confidence: 0.0,
        token_start: start,
        token_end: end,
        start: byte_start,
        end: byte_end,
        text: surface,
    })
}

/// Resolve overlaps in scorer coordinates, then remove a choice prefix and map
/// retained spans into the caller's original text. Prefix/synthetic-suffix
/// candidates are filtered before indexing; malformed offset tables are errors.
pub fn decode_to_utf8(
    text: &str,
    offsets: &[WordOffset],
    candidates: &[ScoredSpan],
    policy: OverlapPolicy,
    choice_prefix_words: usize,
) -> Result<Vec<DecodedSpan>> {
    validate_word_offsets(text, offsets)?;
    let resolved = resolve_overlaps(candidates, policy)?;
    let mut decoded = Vec::new();
    for candidate in resolved {
        let Some(token_start) = candidate.start.checked_sub(choice_prefix_words) else {
            continue;
        };
        let Some(token_end) = candidate.end.checked_sub(choice_prefix_words) else {
            continue;
        };
        if token_start >= token_end || token_end > offsets.len() {
            continue;
        }
        let mut mapped = map_half_open_utf8(text, offsets, token_start, token_end)?;
        if mapped.text.is_empty() {
            continue;
        }
        mapped.confidence = candidate.confidence;
        decoded.push(mapped);
    }
    Ok(decoded)
}

/// Keep every retained span for lists; scalar fields take the first ranked span.
pub fn select_field_value(spans: Vec<DecodedSpan>, dtype: FieldDtype) -> DecodedValue {
    match dtype {
        FieldDtype::List => DecodedValue::List(spans),
        FieldDtype::Scalar => DecodedValue::Scalar(spans.into_iter().next()),
    }
}
