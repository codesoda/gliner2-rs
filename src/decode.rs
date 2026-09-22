use ndarray::ArrayView2;

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredSpan {
    pub token_start: usize,
    pub token_end: usize, // exclusive
    pub start: usize,     // byte offset in original text
    pub end: usize,       // byte offset in original text
    pub text: String,
    pub score: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn find_valid_spans(
    field_logits: ArrayView2<'_, f32>,
    threshold: f32,
    text: &str,
    token_starts: &[usize],
    token_ends: &[usize],
    text_tokens: &[String],
) -> Vec<ScoredSpan> {
    let (text_len, max_width) = field_logits.dim();

    let has_offsets = token_starts.len() == text_len && token_ends.len() == text_len;

    let mut spans = Vec::new();
    for token_start in 0..text_len {
        for width in 0..max_width {
            let token_end = token_start + width + 1;
            if token_end > text_len {
                continue;
            }

            let score = sigmoid(field_logits[[token_start, width]]);
            if score < threshold {
                continue;
            }

            let (start, end, span_text) = if has_offsets {
                let start = token_starts[token_start];
                let end = token_ends[token_end - 1];
                let span_text = text.get(start..end).unwrap_or_default().trim().to_string();
                (start, end, span_text)
            } else {
                let span_text = text_tokens[token_start..token_end].join(" ");
                (0usize, 0usize, span_text.trim().to_string())
            };

            if span_text.is_empty() {
                continue;
            }

            spans.push(ScoredSpan {
                token_start,
                token_end,
                start,
                end,
                text: span_text,
                score,
            });
        }
    }

    spans
}

pub fn greedy_non_overlapping(mut spans: Vec<ScoredSpan>) -> Vec<ScoredSpan> {
    spans.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut selected: Vec<ScoredSpan> = Vec::new();
    'outer: for span in spans {
        for sel in &selected {
            let disjoint = span.token_end <= sel.token_start || span.token_start >= sel.token_end;
            if !disjoint {
                continue 'outer;
            }
        }
        selected.push(span);
    }

    // Ensure deterministic ordering for equal scores.
    selected.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.token_start.cmp(&b.token_start))
            .then_with(|| a.token_end.cmp(&b.token_end))
    });

    selected
}
