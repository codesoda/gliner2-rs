use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::Path;

use anyhow::{Context, anyhow};
use ndarray::{Array1, Array2, Array4, Ix2, Ix4};
use once_cell::sync::Lazy;
use orp::{
    model::Model,
    params::RuntimeParameters,
    pipeline::{Pipeline, PostProcessor, PreProcessor},
};
use ort::session::SessionOutputs;

use crate::Result;

pub struct Extractor {
    model: Model,
}

#[derive(Debug)]
pub struct ExtractorOutput {
    pub count_logits: Array2<f32>,
    pub span_scores: Array4<f32>,
}

impl Extractor {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::new(model_path, RuntimeParameters::default())
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn infer(
        &self,
        text_emb: Array2<f32>,
        schema_emb_padded: Array2<f32>,
        schema_mask: Array1<bool>,
        spans_idx: Array2<i64>,
    ) -> Result<ExtractorOutput> {
        self.model
            .inference(
                ExtractorInput {
                    text_emb,
                    schema_emb_padded,
                    schema_mask,
                    spans_idx,
                },
                &ExtractorPipeline,
                &(),
            )
            .map_err(|e| anyhow!(e.to_string()))
    }
}

struct ExtractorInput {
    text_emb: Array2<f32>,
    schema_emb_padded: Array2<f32>,
    schema_mask: Array1<bool>,
    spans_idx: Array2<i64>,
}

struct ExtractorPipeline;

impl<'a> Pipeline<'a> for ExtractorPipeline {
    type Input = ExtractorInput;
    type Output = ExtractorOutput;
    type Context = ();
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        |input: ExtractorInput| {
            let inputs = ort::inputs! {
                "text_emb" => input.text_emb,
                "schema_emb_padded" => input.schema_emb_padded,
                "schema_mask" => input.schema_mask,
                "spans_idx" => input.spans_idx,
            }?;
            Ok((inputs.into(), ()))
        }
    }

    fn post_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        |(outputs, _ctx): (SessionOutputs<'_, '_>, ())| {
            let count = outputs
                .get("count_logits")
                .context("missing count_logits")?
                .try_extract_tensor::<f32>()?;
            let count = count
                .into_dimensionality::<Ix2>()
                .map_err(|e| anyhow!("unexpected count_logits shape: {e}"))?
                .to_owned();

            let spans = outputs
                .get("span_scores")
                .context("missing span_scores")?
                .try_extract_tensor::<f32>()?;
            let spans = spans
                .into_dimensionality::<Ix4>()
                .map_err(|e| anyhow!("unexpected span_scores shape: {e}"))?
                .to_owned();

            Ok(ExtractorOutput {
                count_logits: count,
                span_scores: spans,
            })
        }
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        static INPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["text_emb", "schema_emb_padded", "schema_mask", "spans_idx"]));
        Some(&INPUTS)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        static OUTPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["count_logits", "span_scores"]));
        Some(&OUTPUTS)
    }
}
