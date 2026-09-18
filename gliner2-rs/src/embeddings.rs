use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::{
    Result,
    schema::{FormattedInput, Segment},
};

#[derive(Debug)]
pub struct ExtractedEmbeddings {
    pub schema_embs: Vec<Vec<Array1<f32>>>,
    pub text_emb: Array2<f32>,
}

/// Extract special token embeddings per schema and pooled text embeddings.
/// Assumes `hidden` shape [1, seq, hidden].
pub fn extract_embeddings(
    hidden: &Array3<f32>,
    formatted: &FormattedInput,
    num_schemas: usize,
) -> Result<ExtractedEmbeddings> {
    let (_, seq_len, hidden_size) = hidden.dim();
    assert_eq!(
        formatted.mapped_indices.len(),
        seq_len,
        "mapping length must match sequence length"
    );

    let view = hidden.index_axis(Axis(0), 0);

    let special_set = ["[P]", "[C]", "[E]", "[R]", "[L]"];

    let mut schema_embs: Vec<Vec<Array1<f32>>> = vec![Vec::new(); num_schemas];
    let mut text_buckets: Vec<Vec<Array1<f32>>> = Vec::new();
    let mut last_text_orig: Option<usize> = None;
    let mut current_bucket: Vec<Array1<f32>> = Vec::new();

    for (idx, mapping) in formatted.mapped_indices.iter().enumerate() {
        let emb = view.slice(s![idx, ..]).to_owned();
        match mapping.segment {
            Segment::Schema => {
                let token = &formatted.subwords[idx];
                if special_set.contains(&token.as_str()) && mapping.schema_idx < schema_embs.len() {
                    schema_embs[mapping.schema_idx].push(emb);
                }
            }
            Segment::Text => {
                if last_text_orig != Some(mapping.orig_idx) {
                    if !current_bucket.is_empty() {
                        text_buckets.push(current_bucket);
                        current_bucket = Vec::new();
                    }
                    last_text_orig = Some(mapping.orig_idx);
                }
                current_bucket.push(emb);
            }
            Segment::Sep => {}
        }
    }
    if !current_bucket.is_empty() {
        text_buckets.push(current_bucket);
    }

    let mut text_rows: Vec<f32> = Vec::with_capacity(text_buckets.len() * hidden_size);
    for bucket in text_buckets {
        // first pooling
        let first = bucket
            .first()
            .expect("empty bucket")
            .as_slice()
            .expect("slice");
        text_rows.extend_from_slice(first);
    }
    let text_emb = Array2::from_shape_vec((text_rows.len() / hidden_size, hidden_size), text_rows)?;

    Ok(ExtractedEmbeddings {
        schema_embs,
        text_emb,
    })
}
