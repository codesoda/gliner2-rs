pub mod assignment;
#[rustfmt::skip] // Deterministic generated tables; checked against the generator.
mod choice_unicode;
mod choices;
pub mod classification;
pub mod classification_pipeline;
pub mod config;
pub mod decode;
pub mod explicit;
pub mod explicit_spans;
pub mod marginals;
pub mod pipeline;
pub mod pool;
pub mod preprocessing;
pub mod record_decode;
mod record_prepare;
pub mod record_schema;
pub mod records;
pub mod relation_decode;
pub mod relation_pairs;
pub mod relations;
pub mod scorer;

pub use classification_pipeline::ClassificationPipeline;
pub use explicit_spans::{ExplicitSpanScore, ExplicitSpanScores};
pub use pipeline::BoundaryPipeline;
