use gliner2_rs::{
    Result,
    classification::FormattedClassification,
    entities::FormattedEntityValue,
    pipeline::Gliner2Pipeline,
    schema_spec::{FieldDtype, SchemaBuilder, StructureFieldSpec},
};

mod common;
use common::{artifacts_available, model_root};

#[test]
fn tutorial_4_combined_schema_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");
    let classifier_onnx = root.join("onnx/gliner2-base-v1/classifier.onnx");

    if !artifacts_available(&[&model_dir, &encoder_onnx, &extractor_onnx, &classifier_onnx])? {
        return Ok(());
    }

    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?
        .with_classifier(classifier_onnx)?;

    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string()])
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .relations(vec!["works_for".to_string()])
        .structure("product")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("price").dtype(FieldDtype::Str))
        .finish()
        .build();

    let text = "Tim Cook works for Apple. The iPhone 15 costs $999. This is exciting!";
    let out = pipeline.extract(text, &schema, 0.5)?;

    assert_eq!(
        out.classifications.get("sentiment"),
        Some(&FormattedClassification::Single("positive".to_string()))
    );

    match out.entities.get("company") {
        Some(FormattedEntityValue::List(values)) => assert!(!values.is_empty()),
        other => panic!("expected company as non-empty list, got {other:?}"),
    }
    match out.entities.get("person") {
        Some(FormattedEntityValue::List(values)) => assert!(!values.is_empty()),
        other => panic!("expected person as non-empty list, got {other:?}"),
    }

    let products = out
        .structures
        .get("product")
        .expect("missing product structure");
    assert!(!products.is_empty());
    let first = &products[0];
    assert!(first.contains_key("name"));
    assert!(first.contains_key("price"));

    assert!(out.relations.contains_key("works_for"));

    Ok(())
}
