use gliner2_rs::{Result, pipeline::Gliner2Pipeline};
mod common;
use common::model_paths_from_args;

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2-base-v1");

    let pipeline = Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?;

    // Minimal tokens (not yet full SchemaTransformer parity).
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

    println!("count_logits shape: {:?}", out.count_logits.shape());
    println!("span_scores shape: {:?}", out.span_scores.shape());
    Ok(())
}
