use std::{collections::BTreeMap, path::PathBuf};

use gliner2_rs::{
    Result,
    classification::FormattedClassification,
    pipeline::Gliner2Pipeline,
    schema_spec::{QuickClassificationTask, SchemaBuilder},
};

fn model_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn tutorial_1_single_label_classification_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");
    let classifier_onnx = root.join("onnx/gliner2-base-v1/classifier.onnx");

    if !model_dir.exists() || !encoder_onnx.exists() || !extractor_onnx.exists() || !classifier_onnx.exists() {
        eprintln!("SKIP: missing model/onnx artifacts");
        return Ok(());
    }

    // Mirrors tutorial/1-classification.md "Single-Label Classification" basic example.
    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?
        .with_classifier(classifier_onnx)?;

    let labels = vec![
        "positive".to_string(),
        "negative".to_string(),
        "neutral".to_string(),
    ];
    let text = "This product exceeded my expectations! Absolutely love it.";

    let schema = SchemaBuilder::new()
        .classification("sentiment", labels.clone())
        .build();

    let out = pipeline.extract(text, &schema, 0.5)?;
    assert_eq!(
        out.classifications.get("sentiment"),
        Some(&FormattedClassification::Single("positive".to_string()))
    );

    let out = pipeline.extract_with_confidence(text, &schema, 0.5)?;
    match out.classifications.get("sentiment") {
        Some(FormattedClassification::SingleWithConfidence { label, confidence }) => {
            assert_eq!(label, "positive");
            assert!(*confidence >= 0.0 && *confidence <= 1.0);
        }
        other => panic!("expected SingleWithConfidence, got {other:?}"),
    }

    let tasks: BTreeMap<String, QuickClassificationTask> =
        BTreeMap::from([("sentiment".to_string(), labels.into())]);
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    assert_eq!(
        out.get("sentiment"),
        Some(&FormattedClassification::Single("positive".to_string()))
    );

    Ok(())
}
