use gliner2_rs::{
    Result, entities::build_entities_schema_tokens, pipeline::Gliner2Pipeline,
    schema_spec::SchemaBuilder,
};
use std::fs;
use tempfile::tempdir;

mod common;
use common::{artifacts_available, model_root};

#[test]
fn configured_cap_reaches_raw_and_both_classification_paths() -> Result<()> {
    let root = model_root();
    let tokenizer = root.join("models/gliner2-base-v1/tokenizer.json");
    let bundle = root.join("onnx/gliner2-base-v1");
    let encoder = bundle.join("encoder.onnx");
    let extractor = bundle.join("extractor_padded.onnx");
    let classifier = bundle.join("classifier.onnx");
    if !artifacts_available(&[&tokenizer, &encoder, &extractor, &classifier])? {
        return Ok(());
    }

    let capped_metadata = tempdir()?;
    fs::copy(&tokenizer, capped_metadata.path().join("tokenizer.json"))?;
    fs::write(
        capped_metadata.path().join("config.json"),
        r#"{"architecture":"span","max_len":2}"#,
    )?;
    let pipeline = Gliner2Pipeline::new(capped_metadata.path(), &encoder, &extractor)?
        .with_classifier(&classifier)?;
    let short = "Alice works";
    let long = "Alice works at Acme in Paris.";
    let labels = vec!["positive".to_string(), "negative".to_string()];
    assert_eq!(
        pipeline.classify(short, "sentiment", &labels, false, 0.5)?,
        pipeline.classify(long, "sentiment", &labels, false, 0.5)?
    );
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "organization".to_string()])
        .classification("sentiment", labels)
        .build();
    assert_eq!(
        pipeline.extract_with_confidence_and_spans(short, &schema, 0.5)?,
        pipeline.extract_with_confidence_and_spans(long, &schema, 0.5)?
    );

    let schemas = vec![build_entities_schema_tokens(&["person".to_string()], None)];
    let short_tokens = vec!["alice".to_string(), "works".to_string()];
    let mut long_tokens = short_tokens.clone();
    long_tokens.extend(["at".to_string(), "acme".to_string()]);
    let short_raw = pipeline.infer_raw(&schemas, &short_tokens)?;
    let long_raw = pipeline.infer_raw(&schemas, &long_tokens)?;
    assert_eq!(short_raw.count_logits, long_raw.count_logits);
    assert_eq!(short_raw.span_scores, long_raw.span_scores);
    assert_eq!(long_raw.span_scores.shape()[2], 2);
    Ok(())
}
