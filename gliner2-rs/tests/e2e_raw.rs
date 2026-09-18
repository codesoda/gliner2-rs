use std::path::PathBuf;

use gliner2_rs::{Result, pipeline::Gliner2Pipeline};

fn model_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn end_to_end_raw_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !encoder_onnx.exists() || !extractor_onnx.exists() {
        eprintln!("SKIP: missing onnx artifacts");
        return Ok(());
    }

    let pipeline = Gliner2Pipeline::new(model_dir, encoder_onnx, extractor_onnx)?;

    let schema_tokens_list = vec![vec![
        "[P]".to_string(),
        "[E]".to_string(),
        "person".to_string(),
    ]];
    let text_tokens = vec![
        "alice".to_string(),
        "works".to_string(),
        "at".to_string(),
        "acme".to_string(),
    ];

    let out = pipeline.infer_raw(&schema_tokens_list, &text_tokens)?;

    assert_eq!(out.count_logits.shape(), &[1, 20]);
    assert_eq!(out.span_scores.ndim(), 4);

    Ok(())
}
