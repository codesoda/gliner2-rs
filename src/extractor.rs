use std::path::Path;

use ndarray::{Array1, Array2, Array4, Ix2, Ix4};

use crate::Result;
use crate::options::RuntimeOptions;
use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &["text_emb", "schema_emb_padded", "schema_mask", "spans_idx"];
const OUTPUTS: &[&str] = &["count_logits", "span_scores"];

pub struct Extractor {
    session: RuntimeSession,
}

#[derive(Debug)]
pub struct ExtractorOutput {
    pub count_logits: Array2<f32>,
    pub span_scores: Array4<f32>,
}

impl Extractor {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_options(model_path, RuntimeOptions::default())
    }

    pub fn new_with_options(model_path: impl AsRef<Path>, options: RuntimeOptions) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load_with(model_path, "extractor", INPUTS, OUTPUTS, options)?,
        })
    }

    pub fn infer(
        &self,
        text_emb: Array2<f32>,
        schema_emb_padded: Array2<f32>,
        schema_mask: Array1<bool>,
        spans_idx: Array2<i64>,
    ) -> Result<ExtractorOutput> {
        let inputs = ort::inputs! {
            "text_emb" => tensor(&text_emb)?,
            "schema_emb_padded" => tensor(&schema_emb_padded)?,
            "schema_mask" => tensor(&schema_mask)?,
            "spans_idx" => tensor(&spans_idx)?,
        };
        self.session.run(inputs, |outputs| {
            let count_logits = extract::<f32, Ix2>(outputs, "count_logits")?;
            let span_scores = extract::<f32, Ix4>(outputs, "span_scores")?;
            Ok(ExtractorOutput {
                count_logits,
                span_scores,
            })
        })
    }
}
