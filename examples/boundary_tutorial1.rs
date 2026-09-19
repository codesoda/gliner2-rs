use std::{env, path::PathBuf};

use gliner2_rs::{Result, pipeline::BoundaryPipeline, schema_spec::SchemaBuilder};

fn main() -> Result<()> {
    let bundle = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("onnx/gliner2.5-base-v1"));
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let text = "Ada Lovelace worked with Charles Babbage in London.";
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_owned(), "location".to_owned()])
        .build();
    let result = pipeline.extract_with_confidence_and_spans(text, &schema, 0.5)?;
    println!("{:#?}", result.entities);
    Ok(())
}
