use std::path::PathBuf;

use gliner2_rs::{Result, pipeline::Gliner2Pipeline};

fn model_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn tutorial_2_entity_extraction_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !model_dir.exists() || !encoder_onnx.exists() || !extractor_onnx.exists() {
        eprintln!("SKIP: missing model/onnx artifacts");
        return Ok(());
    }

    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?;

    // Mirrors tutorial/2-ner.md "Basic Entity Extraction" example.
    let text = "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino.";
    let labels = vec![
        "company".to_string(),
        "person".to_string(),
        "product".to_string(),
        "location".to_string(),
    ];

    let out = pipeline.extract_entities(text, &labels, 0.5)?;

    let company = out.iter().find(|m| m.label == "company").unwrap();
    assert!(company.spans.iter().any(|s| s.text.contains("Apple")));

    let person = out.iter().find(|m| m.label == "person").unwrap();
    assert!(person.spans.iter().any(|s| s.text.contains("Tim Cook")));

    let product = out.iter().find(|m| m.label == "product").unwrap();
    assert!(product.spans.iter().any(|s| s.text.contains("iPhone")));

    let location = out.iter().find(|m| m.label == "location").unwrap();
    assert!(location.spans.iter().any(|s| s.text.contains("Cupertino")));

    Ok(())
}

