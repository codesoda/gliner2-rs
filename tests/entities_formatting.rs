use gliner2_rs::{
    entities::{EntitySpan, FormattedEntitySpan, FormattedEntityValue, format_entity_spans},
    schema_spec::FieldDtype,
};

#[test]
fn entity_formatting_respects_dtype_flags_and_deduping() {
    let spans = vec![
        EntitySpan {
            start: 0,
            end: 5,
            text: "Alice".to_string(),
            score: 0.9,
        },
        // Case-insensitive duplicate; should be removed.
        EntitySpan {
            start: 10,
            end: 15,
            text: "alice".to_string(),
            score: 0.8,
        },
    ];

    let out = format_entity_spans(&spans, FieldDtype::List, false, false);
    assert_eq!(
        out,
        FormattedEntityValue::List(vec![FormattedEntitySpan::Text("Alice".to_string())])
    );

    let out = format_entity_spans(&spans, FieldDtype::List, true, false);
    assert_eq!(
        out,
        FormattedEntityValue::List(vec![FormattedEntitySpan::TextWithConfidence {
            text: "Alice".to_string(),
            confidence: 0.9,
        }])
    );

    let out = format_entity_spans(&spans, FieldDtype::List, false, true);
    assert_eq!(
        out,
        FormattedEntityValue::List(vec![FormattedEntitySpan::TextWithSpans {
            text: "Alice".to_string(),
            start: 0,
            end: 5,
        }])
    );

    let out = format_entity_spans(&spans, FieldDtype::Str, true, true);
    assert_eq!(
        out,
        FormattedEntityValue::Single(Some(FormattedEntitySpan::TextWithConfidenceAndSpans {
            text: "Alice".to_string(),
            confidence: 0.9,
            start: 0,
            end: 5,
        }))
    );
}
