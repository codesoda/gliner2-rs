use std::{env, path::PathBuf};

use gliner2_rs::{Result, pipeline::BoundaryPipeline, schema_spec::SchemaBuilder};

fn main() -> Result<()> {
    let bundle = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("onnx/gliner2.5-base-v1"));
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let text = "Urgent: the customer cannot log in after the security update.";
    let schema = SchemaBuilder::new()
        .classification("priority", vec!["urgent".to_owned(), "normal".to_owned()])
        .classification(
            "department",
            vec![
                "support".to_owned(),
                "sales".to_owned(),
                "billing".to_owned(),
            ],
        )
        .build();
    let result = pipeline.extract_with_confidence(text, &schema, 0.5)?;
    println!("{:#?}", result.classifications);
    Ok(())
}
