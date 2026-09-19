use gliner2_rs::{Result, pipeline::Gliner2Pipeline};

mod common;
use common::{artifacts_available, model_root};

#[test]
fn extract_entities_finds_simple_spans() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !artifacts_available(&[&model_dir, &encoder_onnx, &extractor_onnx])? {
        return Ok(());
    }

    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?;

    let text = "Alice works at Acme in Paris.";
    let labels = vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
    ];

    let out = pipeline.extract_entities(text, &labels, 0.5)?;

    let person = out.iter().find(|m| m.label == "person").unwrap();
    assert!(
        person
            .spans
            .iter()
            .any(|s| s.text == "Alice" && s.start == 0 && s.end == 5)
    );

    let org = out.iter().find(|m| m.label == "organization").unwrap();
    assert!(org.spans.iter().any(|s| s.text == "Acme"));

    let loc = out.iter().find(|m| m.label == "location").unwrap();
    assert!(loc.spans.iter().any(|s| s.text == "Paris"));

    Ok(())
}
