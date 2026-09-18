use ndarray::Array3;

use gliner2_rs::{
    embeddings::extract_embeddings,
    schema::{FormattedInput, Mapping, Segment},
};

#[test]
fn extracts_schema_and_text_embeddings_first_pool() {
    // Build a simple sequence: [P], person, [SEP_TEXT], Alice
    // Mapping length matches seq_len = 4.
    let formatted = FormattedInput {
        subwords: vec![
            "[P]".to_string(),
            "person".to_string(),
            "[SEP_TEXT]".to_string(),
            "Alice".to_string(),
        ],
        input_ids: vec![0, 1, 2, 3],
        attention_mask: vec![1, 1, 1, 1],
        mapped_indices: vec![
            Mapping {
                segment: Segment::Schema,
                orig_idx: 0,
                schema_idx: 0,
            },
            Mapping {
                segment: Segment::Schema,
                orig_idx: 1,
                schema_idx: 0,
            },
            Mapping {
                segment: Segment::Sep,
                orig_idx: 2,
                schema_idx: 1,
            },
            Mapping {
                segment: Segment::Text,
                orig_idx: 3,
                schema_idx: 1,
            },
        ],
    };

    // Hidden shape [1, 4, 2]; use distinct values per position.
    let data = vec![
        // [P]
        1.0, 1.1, //
        // person
        2.0, 2.2, //
        // sep
        3.0, 3.3, //
        // Alice
        4.0, 4.4,
    ];
    let hidden = Array3::from_shape_vec((1, 4, 2), data).unwrap();

    let extracted = extract_embeddings(&hidden, &formatted, 1).unwrap();

    assert_eq!(extracted.schema_embs.len(), 1);
    assert_eq!(extracted.schema_embs[0].len(), 1); // only [P] is kept
    assert_eq!(extracted.schema_embs[0][0].as_slice().unwrap(), &[1.0, 1.1]);

    assert_eq!(extracted.text_emb.shape(), &[1, 2]);
    assert_eq!(extracted.text_emb.row(0).to_vec(), vec![4.0, 4.4]);
}
