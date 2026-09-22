use crate::{Result, tokenizer::RuntimeTokenizer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Schema,
    Sep,
    Text,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping {
    pub segment: Segment,
    pub orig_idx: usize,
    pub schema_idx: usize,
}

#[derive(Debug, Clone)]
pub struct FormattedInput {
    pub subwords: Vec<String>,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub mapped_indices: Vec<Mapping>,
}

/// Combine schema tokens + [SEP_TEXT] + text tokens into a flat subword list,
/// along with the mapping used for pooling and schema extraction.
pub fn format_input_with_mapping(
    tokenizer: &RuntimeTokenizer,
    schema_tokens_list: &[Vec<String>],
    text_tokens: &[String],
) -> Result<FormattedInput> {
    let mut combined: Vec<String> = Vec::new();
    for structure_tokens in schema_tokens_list {
        combined.extend(structure_tokens.iter().cloned());
        combined.push("[SEP_STRUCT]".to_string());
    }
    if !combined.is_empty() {
        combined.pop(); // drop trailing [SEP_STRUCT]
    }
    combined.push("[SEP_TEXT]".to_string());
    combined.extend(text_tokens.iter().cloned());

    let num_schemas = schema_tokens_list.len();
    let text_schema_idx = num_schemas;

    let mut subwords = Vec::new();
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut mapped = Vec::new();

    let mut current_schema_idx = 0usize;
    let mut found_sep_text = false;

    for (orig_idx, token) in combined.iter().enumerate() {
        let (segment, schema_idx) = if token == "[SEP_TEXT]" {
            found_sep_text = true;
            (Segment::Sep, text_schema_idx)
        } else if !found_sep_text {
            let idx = current_schema_idx;
            if token == "[SEP_STRUCT]" {
                current_schema_idx += 1;
            }
            (Segment::Schema, idx)
        } else {
            (Segment::Text, text_schema_idx)
        };

        let pieces = tokenizer.tokenize_token(token)?;
        let ids: Vec<i64> = pieces
            .iter()
            .map(|p| {
                tokenizer
                    .id_for(p)
                    .or(tokenizer.id_for("[UNK]"))
                    .map(|id| id as i64)
                    .ok_or_else(|| anyhow::anyhow!("token '{}' missing id", p))
            })
            .collect::<Result<_>>()?;

        mapped.extend(std::iter::repeat_n(
            Mapping {
                segment: segment.clone(),
                orig_idx,
                schema_idx,
            },
            pieces.len(),
        ));
        subwords.extend(pieces.iter().cloned());
        input_ids.extend(ids);
        attention_mask.extend(std::iter::repeat_n(1i64, pieces.len()));
    }

    Ok(FormattedInput {
        subwords,
        input_ids,
        attention_mask,
        mapped_indices: mapped,
    })
}
