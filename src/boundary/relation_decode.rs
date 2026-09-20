use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct RelationMention {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelationEdge {
    pub head: RelationMention,
    pub tail: RelationMention,
    pub score: f32,
}

/// Collapse contained, repeated-coordinate, and semantically duplicate relation edges.
///
/// Mention coordinates are half-open UTF-8 byte offsets into `text`. Selection lengths
/// and gaps are measured in Unicode scalar values to match Python string coordinates.
pub fn deduplicate_relation_edges(text: &str, edges: &[RelationEdge]) -> Result<Vec<RelationEdge>> {
    let codepoints = CodepointOffsets::new(text);
    for (edge_index, edge) in edges.iter().enumerate() {
        validate_mention(text, &edge.head, edge_index, "head")?;
        validate_mention(text, &edge.tail, edge_index, "tail")?;
        ensure!(
            edge.score.is_finite() && (0.0..=1.0).contains(&edge.score),
            "edge {edge_index} score must be finite and in [0, 1], got {}",
            edge.score
        );
    }
    if edges.len() < 2 {
        return Ok(edges.to_vec());
    }

    let head_canonical = canonical_mentions(edges, true, &codepoints);
    let tail_canonical = canonical_mentions(edges, false, &codepoints);

    // Python dictionaries retain the first insertion position when a value is replaced.
    let mut exact_positions = HashMap::<(usize, usize, usize, usize), usize>::new();
    let mut exact = Vec::<RelationEdge>::new();
    for edge in edges {
        let head = head_canonical[&(edge.head.start, edge.head.end)].clone();
        let tail = tail_canonical[&(edge.tail.start, edge.tail.end)].clone();
        let key = (head.start, head.end, tail.start, tail.end);
        let normalized = RelationEdge {
            head,
            tail,
            score: edge.score,
        };
        if let Some(&position) = exact_positions.get(&key) {
            if edge.score > exact[position].score {
                exact[position] = normalized;
            }
        } else {
            exact_positions.insert(key, exact.len());
            exact.push(normalized);
        }
    }

    let mut semantic_positions = HashMap::<(String, String), usize>::new();
    let mut semantic = Vec::<RelationEdge>::new();
    for edge in exact {
        let key = (
            semantic_text(&edge.head.text),
            semantic_text(&edge.tail.text),
        );
        if let Some(&position) = semantic_positions.get(&key) {
            if semantic_rank_is_lower(&edge, &semantic[position], &codepoints) {
                semantic[position] = edge;
            }
        } else {
            semantic_positions.insert(key, semantic.len());
            semantic.push(edge);
        }
    }

    let token_sets: Vec<_> = semantic
        .iter()
        .map(|edge| {
            (
                semantic_tokens(&edge.head.text),
                semantic_tokens(&edge.tail.text),
            )
        })
        .collect();
    let mut kept = Vec::new();
    for (index, edge) in semantic.into_iter().enumerate() {
        let (head_tokens, tail_tokens) = &token_sets[index];
        let dominated =
            token_sets
                .iter()
                .enumerate()
                .any(|(other_index, (other_head, other_tail))| {
                    other_index != index
                        && ((head_tokens.is_subset(other_head)
                            && head_tokens != other_head
                            && tail_tokens == other_tail)
                            || (tail_tokens.is_subset(other_tail)
                                && tail_tokens != other_tail
                                && head_tokens == other_head))
                });
        if !dominated {
            kept.push(edge);
        }
    }

    kept.sort_by(|left, right| {
        left.head
            .start
            .cmp(&right.head.start)
            .then_with(|| left.tail.start.cmp(&right.tail.start))
            .then_with(|| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
            })
    });
    Ok(kept)
}

fn validate_mention(
    text: &str,
    mention: &RelationMention,
    edge_index: usize,
    side: &str,
) -> Result<()> {
    if mention.start >= mention.end {
        bail!(
            "edge {edge_index} {side} span must be nonempty, got {}..{}",
            mention.start,
            mention.end
        );
    }
    ensure!(
        mention.end <= text.len(),
        "edge {edge_index} {side} span {}..{} is outside text byte length {}",
        mention.start,
        mention.end,
        text.len()
    );
    ensure!(
        text.is_char_boundary(mention.start) && text.is_char_boundary(mention.end),
        "edge {edge_index} {side} span {}..{} is not on UTF-8 boundaries",
        mention.start,
        mention.end
    );
    let surface = text[mention.start..mention.end].trim_matches(is_python_whitespace);
    ensure!(
        mention.text == surface,
        "edge {edge_index} {side} surface {:?} does not match trimmed source {:?}",
        mention.text,
        surface
    );
    Ok(())
}

fn canonical_mentions(
    edges: &[RelationEdge],
    head: bool,
    codepoints: &CodepointOffsets,
) -> HashMap<(usize, usize), RelationMention> {
    let mut positions = HashMap::<(usize, usize), usize>::new();
    let mut mentions = Vec::<RelationMention>::new();
    for edge in edges {
        let mention = if head { &edge.head } else { &edge.tail };
        let key = (mention.start, mention.end);
        if let Some(&position) = positions.get(&key) {
            mentions[position] = mention.clone();
        } else {
            positions.insert(key, mentions.len());
            mentions.push(mention.clone());
        }
    }

    let mut canonical = HashMap::with_capacity(mentions.len());
    for mention in &mentions {
        let mut best: Option<&RelationMention> = None;
        for candidate in &mentions {
            if candidate.start <= mention.start && candidate.end >= mention.end {
                let replace = best.is_none_or(|current| {
                    let candidate_width = codepoints.width(candidate.start, candidate.end);
                    let current_width = codepoints.width(current.start, current.end);
                    candidate_width > current_width
                        || (candidate_width == current_width
                            && codepoints.at(candidate.start) < codepoints.at(current.start))
                });
                if replace {
                    best = Some(candidate);
                }
            }
        }
        canonical.insert(
            (mention.start, mention.end),
            best.expect("each valid mention contains itself").clone(),
        );
    }
    canonical
}

fn semantic_rank_is_lower(
    candidate: &RelationEdge,
    previous: &RelationEdge,
    codepoints: &CodepointOffsets,
) -> bool {
    let candidate_gap = codepoint_gap(candidate, codepoints);
    let previous_gap = codepoint_gap(previous, codepoints);
    candidate_gap < previous_gap
        || (candidate_gap == previous_gap
            && (candidate.score > previous.score
                || (candidate.score == previous.score
                    && (candidate.head.start < previous.head.start
                        || (candidate.head.start == previous.head.start
                            && candidate.tail.start < previous.tail.start)))))
}

fn codepoint_gap(edge: &RelationEdge, codepoints: &CodepointOffsets) -> usize {
    let head_start = codepoints.at(edge.head.start);
    let head_end = codepoints.at(edge.head.end);
    let tail_start = codepoints.at(edge.tail.start);
    let tail_end = codepoints.at(edge.tail.end);
    head_start
        .saturating_sub(tail_end)
        .max(tail_start.saturating_sub(head_end))
}

fn semantic_text(value: &str) -> String {
    let folded = super::choice_unicode::casefold(value);
    folded
        .split(is_python_whitespace)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn semantic_tokens(value: &str) -> HashSet<String> {
    semantic_text(value)
        .split(' ')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn is_python_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000D}'
            | '\u{001C}'..='\u{001F}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

struct CodepointOffsets {
    by_byte: HashMap<usize, usize>,
}

impl CodepointOffsets {
    fn new(text: &str) -> Self {
        let mut by_byte: HashMap<_, _> = text
            .char_indices()
            .enumerate()
            .map(|(codepoint, (byte, _))| (byte, codepoint))
            .collect();
        by_byte.insert(text.len(), text.chars().count());
        Self { by_byte }
    }

    fn at(&self, byte: usize) -> usize {
        self.by_byte[&byte]
    }

    fn width(&self, start: usize, end: usize) -> usize {
        self.at(end) - self.at(start)
    }
}
