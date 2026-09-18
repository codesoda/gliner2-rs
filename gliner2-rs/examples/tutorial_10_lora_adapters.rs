use std::time::Instant;

use gliner2_rs::{Result, pipeline::Gliner2Pipeline};
mod common;
use common::{model_paths_from_args, repo_root};

fn main() -> Result<()> {
    let root = repo_root();
    let paths = model_paths_from_args("onnx/gliner2-base-v1");

    let load_start = Instant::now();
    let mut pipeline = Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?;
    println!(
        "base model load took: {:.2?} (onnx={})",
        load_start.elapsed(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // Mirrors tutorial/10-lora_adapters.md "Loading and Swapping Adapters".
    let legal_adapter = root.join("adapters/legal");
    if !legal_adapter.join("encoder.onnx").exists() {
        println!(
            "SKIP: missing adapter bundle at {} (expected `encoder.onnx`).\n\
Export an adapter-specific encoder ONNX (merged weights) and place it there.",
            legal_adapter.display()
        );
        return Ok(());
    }

    let start = Instant::now();
    pipeline.load_adapter(&legal_adapter)?;
    println!("adapter load took: {:.2?}", start.elapsed());
    println!("adapter_config: {:#?}", pipeline.adapter_config());
    println!("-----------------");

    let text = "Apple Inc. filed a lawsuit against Samsung Electronics.";
    let labels = vec!["company".to_string(), "legal_action".to_string()];
    let start = Instant::now();
    let out = pipeline.extract_entities(text, &labels, 0.5)?;
    println!("text: {text}");
    println!("entities: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
