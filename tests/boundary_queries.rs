use gliner2_rs::{
    embeddings::{QueryKind, extract_boundary_queries, extract_embeddings},
    schema::{FormattedInput, Mapping, Segment},
};
use ndarray::Array3;

fn formatted(subwords: &[&str], schemas: &[usize]) -> FormattedInput {
    FormattedInput {
        subwords: subwords.iter().map(|value| (*value).to_string()).collect(),
        input_ids: vec![0; subwords.len()],
        attention_mask: vec![1; subwords.len()],
        mapped_indices: schemas
            .iter()
            .enumerate()
            .map(|(index, schema_idx)| Mapping {
                segment: if subwords[index] == "text" {
                    Segment::Text
                } else {
                    Segment::Schema
                },
                orig_idx: index,
                schema_idx: *schema_idx,
            })
            .collect(),
    }
}

#[test]
fn routes_mixed_markers_once_in_schema_order() {
    let input = formatted(
        &["[P]", "[E]", "[L]", "[C]", "[R]", "[R]", "text"],
        &[0, 0, 0, 0, 1, 1, 2],
    );
    let hidden =
        Array3::from_shape_vec((1, 7, 2), (0..14).map(|value| value as f32).collect()).unwrap();

    let queries = extract_boundary_queries(&hidden, &input, 2).unwrap();
    assert_eq!(queries.query_emb.shape(), &[4, 2]);
    assert_eq!(queries.query_emb.row(0).to_vec(), vec![2.0, 3.0]);
    assert_eq!(queries.query_emb.row(1).to_vec(), vec![6.0, 7.0]);
    assert_eq!(queries.query_emb.row(2).to_vec(), vec![8.0, 9.0]);
    assert_eq!(queries.query_emb.row(3).to_vec(), vec![10.0, 11.0]);
    assert_eq!(queries.metadata[0].schema_idx, 0);
    assert_eq!(queries.metadata[0].field_idx, 0);
    assert_eq!(queries.metadata[0].kind, QueryKind::Entity);
    assert_eq!(queries.metadata[1].field_idx, 1);
    assert_eq!(queries.metadata[1].kind, QueryKind::Content);
    assert_eq!(queries.metadata[2].schema_idx, 1);
    assert_eq!(queries.metadata[2].field_idx, 0);
    assert_eq!(queries.metadata[2].kind, QueryKind::Relation);
    assert_eq!(queries.metadata[3].field_idx, 1);

    // Legacy extraction continues to gather [P], [E], [L], [C], and [R].
    let legacy = extract_embeddings(&hidden, &input, 2).unwrap();
    assert_eq!(legacy.schema_embs[0].len(), 4);
    assert_eq!(legacy.schema_embs[1].len(), 2);
}

#[test]
fn empty_query_set_has_valid_zero_row_shape() {
    let input = formatted(&["[P]", "[L]", "text"], &[0, 0, 1]);
    let hidden = Array3::zeros((1, 3, 5));

    let queries = extract_boundary_queries(&hidden, &input, 1).unwrap();
    assert_eq!(queries.query_emb.shape(), &[0, 5]);
    assert!(queries.metadata.is_empty());
}

#[test]
fn invalid_shapes_and_schema_indices_return_errors() {
    let input = formatted(&["[E]"], &[0]);
    let wrong_sequence = Array3::zeros((1, 2, 3));
    assert!(extract_boundary_queries(&wrong_sequence, &input, 1).is_err());

    let wrong_batch = Array3::zeros((2, 1, 3));
    assert!(extract_boundary_queries(&wrong_batch, &input, 1).is_err());

    let bad_schema = formatted(&["[R]"], &[1]);
    let hidden = Array3::zeros((1, 1, 3));
    assert!(extract_boundary_queries(&hidden, &bad_schema, 1).is_err());
}
