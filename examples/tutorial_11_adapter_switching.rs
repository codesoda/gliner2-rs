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

    // Mirrors tutorial/11-adapter_switching.md "Quick Start".
    let legal_adapter = root.join("adapters/legal");
    let medical_adapter = root.join("adapters/medical");
    if !legal_adapter.join("encoder.onnx").exists()
        || !medical_adapter.join("encoder.onnx").exists()
    {
        println!(
            "SKIP: missing adapter bundles.\n\
Expected:\n\
- {}\n\
- {}\n\
(each containing `encoder.onnx`)",
            legal_adapter.display(),
            medical_adapter.display(),
        );
        return Ok(());
    }

    // Legal domain
    let start = Instant::now();
    pipeline.load_adapter(&legal_adapter)?;
    println!("legal adapter load took: {:.2?}", start.elapsed());
    println!("has_adapter: {}", pipeline.has_adapter());
    println!("adapter_config: {:#?}", pipeline.adapter_config());
    let start = Instant::now();
    let legal = pipeline.extract_entities("Apple sued Google", &["company".to_string()], 0.5)?;
    println!("legal_result: {legal:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Medical domain (previous adapter auto-swapped)
    let start = Instant::now();
    pipeline.load_adapter(&medical_adapter)?;
    println!("medical adapter load took: {:.2?}", start.elapsed());
    println!("has_adapter: {}", pipeline.has_adapter());
    println!("adapter_config: {:#?}", pipeline.adapter_config());
    let start = Instant::now();
    let medical =
        pipeline.extract_entities("Patient has diabetes", &["disease".to_string()], 0.5)?;
    println!("medical_result: {medical:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Use base model (no adapter)
    let start = Instant::now();
    pipeline.unload_adapter()?;
    println!("unload_adapter took: {:.2?}", start.elapsed());
    println!("has_adapter: {}", pipeline.has_adapter());
    println!("adapter_config: {:#?}", pipeline.adapter_config());
    let start = Instant::now();
    let base = pipeline.extract_entities("Some text", &["entity".to_string()], 0.5)?;
    println!("base_result: {base:#?}");
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
