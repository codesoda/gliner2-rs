use gliner2_rs::{Result, pipeline::Gliner2Pipeline};
mod common;
use common::model_paths_from_args;

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2-base-v1");

    let pipeline = Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?;

    let text = "Alice works at Acme in Paris.";
    let labels = vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
    ];

    let out = pipeline.extract_entities(text, &labels, 0.5)?;
    for m in out {
        println!("{}: {:?}", m.label, m.spans);
    }

    Ok(())
}
