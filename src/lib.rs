pub mod adapters;
pub mod api;
pub mod boundary;
pub mod classification;
pub mod classifier;
pub mod config;
pub mod decode;
pub mod embeddings;
pub mod encoder;
pub mod entities;
pub mod extractor;
pub mod json;
pub mod pipeline;
pub mod preprocessing;
pub mod relations;
mod runtime;
pub mod schema;
pub mod schema_spec;
pub mod spans;
pub mod structures;
pub mod text;
pub mod tokenizer;
pub mod training;
pub mod validators;

/// Architecture-aware high-level extractor. The low-level ONNX span head
/// remains available as [`extractor::Extractor`].
pub type Extractor = pipeline::AutoPipeline;

pub type Result<T> = anyhow::Result<T>;
