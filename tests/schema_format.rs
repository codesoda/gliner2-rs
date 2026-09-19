use gliner2_rs::{
    Result,
    schema::{Segment, format_input_with_mapping},
    tokenizer::RuntimeTokenizer,
};

mod common;
use common::{artifacts_available, model_root};

#[test]
fn mapping_segments_are_correct() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");

    if !artifacts_available(&[&model_dir])? {
        return Ok(());
    }

    let tokenizer = RuntimeTokenizer::from_dir(&model_dir)?;

    let schema_tokens_list = vec![vec!["[P]".to_string(), "person".to_string()]];
    let text_tokens = vec!["Alice".to_string(), "works".to_string()];

    let formatted = format_input_with_mapping(&tokenizer, &schema_tokens_list, &text_tokens)?;

    // Combined tokens: [P], person, [SEP_TEXT], Alice, works
    assert_eq!(formatted.mapped_indices.len(), formatted.subwords.len());
    assert_eq!(formatted.input_ids.len(), formatted.subwords.len());

    // First two are schema, third is sep, rest are text.
    let segments: Vec<_> = formatted
        .mapped_indices
        .iter()
        .map(|m| (&m.segment, m.schema_idx))
        .collect();

    // Because of subword splitting, we check prefixes.
    for (pos, expected) in [
        (Segment::Schema, 0), // [P]
        (Segment::Schema, 0), // person
        (Segment::Sep, 1),    // [SEP_TEXT] uses text_schema_idx = num_schemas
        (Segment::Text, 1),   // Alice
        (Segment::Text, 1),   // works
    ]
    .into_iter()
    .enumerate()
    {
        assert!(pos < segments.len());
        assert_eq!(segments[pos].0, &expected.0);
        assert_eq!(segments[pos].1, expected.1);
    }

    Ok(())
}
