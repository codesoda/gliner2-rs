use std::path::PathBuf;

use gliner2_rs::{Result, pipeline::Gliner2Pipeline};

fn model_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn tutorial_6_relation_extraction_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !model_dir.exists() || !encoder_onnx.exists() || !extractor_onnx.exists() {
        eprintln!("SKIP: missing model/onnx artifacts");
        return Ok(());
    }

    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?;

    let text = "John works for Apple Inc. and lives in San Francisco.";
    let relation_types = vec!["works_for".to_string(), "lives_in".to_string()];

    // Ensure all requested relation types are always present, even if empty.
    let out = pipeline.extract_relations(text, &relation_types, 1.1)?;
    for rel in &relation_types {
        let pairs = out.get(rel).expect("missing requested relation type");
        assert!(pairs.is_empty());
    }

    // Smoke-test decoding at a realistic threshold (contents may vary by model).
    let out = pipeline.extract_relations(text, &relation_types, 0.5)?;
    for rel in &relation_types {
        assert!(out.contains_key(rel));
    }
    for pairs in out.values() {
        for (head, tail) in pairs {
            assert!(!head.trim().is_empty());
            assert!(!tail.trim().is_empty());
        }
    }

    // Batch API mirrors the Python tutorial surface (currently loops internally).
    let texts = vec![
        "John works for Microsoft and lives in Seattle.",
        "Sarah founded TechStartup in 2020.",
        "Bob reports to Alice at Google.",
    ];
    let batch = pipeline.batch_extract_relations(&texts, &relation_types, 1.1, 2)?;
    assert_eq!(batch.len(), texts.len());
    for item in batch {
        for rel in &relation_types {
            assert!(item.contains_key(rel));
        }
    }

    Ok(())
}

