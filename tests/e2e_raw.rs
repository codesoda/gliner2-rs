use gliner2_rs::{Result, pipeline::Gliner2Pipeline};

mod common;
use common::{artifacts_available, model_root};

#[test]
fn end_to_end_raw_runs() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let encoder_onnx = root.join("onnx/gliner2-base-v1/encoder.onnx");
    let extractor_onnx = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !artifacts_available(&[&model_dir, &encoder_onnx, &extractor_onnx])? {
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
