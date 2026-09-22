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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    Entity,
    Content,
    Relation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryMetadata {
    pub schema_idx: usize,
    pub field_idx: usize,
    pub kind: QueryKind,
}

#[derive(Debug)]
pub struct BoundaryQueryEmbeddings {
    pub query_emb: Array2<f32>,
    pub metadata: Vec<QueryMetadata>,
}

/// Enumerate boundary extraction queries in schema order.
///
/// Exactly one query is emitted for each `[E]`, `[C]`, or `[R]` marker.
/// `[P]` and classification-only `[L]` markers are intentionally excluded.
pub fn extract_boundary_queries(
    hidden: &Array3<f32>,
    formatted: &FormattedInput,
    num_schemas: usize,
) -> Result<BoundaryQueryEmbeddings> {
    let (batch, seq_len, hidden_size) = hidden.dim();
    if batch != 1 {
        return Err(anyhow::anyhow!(
            "boundary query extraction expects hidden batch size 1, got {batch}"
        ));
    }
    if formatted.mapped_indices.len() != seq_len || formatted.subwords.len() != seq_len {
        return Err(anyhow::anyhow!(
            "boundary query extraction shape mismatch: hidden sequence {seq_len}, mappings {}, subwords {}",
            formatted.mapped_indices.len(),
            formatted.subwords.len()
        ));
    }

    let view = hidden.index_axis(Axis(0), 0);
    let mut rows = Vec::new();
    let mut metadata = Vec::new();
    let mut field_counts = vec![0usize; num_schemas];

    for (idx, mapping) in formatted.mapped_indices.iter().enumerate() {
        if mapping.segment != Segment::Schema {
            continue;
        }

        let kind = match formatted.subwords[idx].as_str() {
            "[E]" => QueryKind::Entity,
            "[C]" => QueryKind::Content,
            "[R]" => QueryKind::Relation,
            _ => continue,
        };
        if mapping.schema_idx >= num_schemas {
            return Err(anyhow::anyhow!(
                "query marker at sequence index {idx} references schema {}, but only {num_schemas} schemas were provided",
                mapping.schema_idx
            ));
        }

        let field_idx = field_counts[mapping.schema_idx];
        field_counts[mapping.schema_idx] += 1;
        rows.extend(view.slice(s![idx, ..]).iter().copied());
        metadata.push(QueryMetadata {
            schema_idx: mapping.schema_idx,
            field_idx,
            kind,
        });
    }

    let query_emb = Array2::from_shape_vec((metadata.len(), hidden_size), rows)?;
    Ok(BoundaryQueryEmbeddings {
        query_emb,
        metadata,
    })
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
