use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2};
use once_cell::sync::Lazy;
use orp::{
    model::Model,
    params::RuntimeParameters,
    pipeline::{Pipeline, PostProcessor, PreProcessor},
};
use ort::session::SessionOutputs;
use std::collections::HashSet;
use std::iter::FromIterator;
mod common;
use common::model_paths_from_args;

/// Simple encoder smoke pipeline: build dummy inputs and return the output shape.
struct EncoderPipeline;

#[derive(Clone, Copy)]
struct EncoderInput {
    batch: usize,
    seq: usize,
}

impl<'a> Pipeline<'a> for EncoderPipeline {
    type Input = EncoderInput;
    type Output = Vec<usize>;
    type Context = ();
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        |input: EncoderInput| {
            let ids = Array2::<i64>::ones((input.batch, input.seq));
            let mask = Array2::<i64>::ones((input.batch, input.seq));
            let inputs = ort::inputs! {
                "input_ids" => ids,
                "attention_mask" => mask,
            }?;
            Ok((inputs.into(), ()))
        }
    }

    fn post_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        |(outputs, _ctx): (SessionOutputs<'_, '_>, ())| {
            let value = outputs
                .get("last_hidden_state")
                .context("missing last_hidden_state")?;
            let view = value.try_extract_tensor::<f32>()?;
            Ok(view.shape().to_vec())
        }
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        static INPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["input_ids", "attention_mask"]));
        Some(&INPUTS)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        static OUTPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["last_hidden_state"]));
        Some(&OUTPUTS)
    }
}

/// Extractor smoke pipeline: build dummy embeddings and span indices and return output shapes.
struct ExtractorPipeline {
    hidden_size: usize,
    max_width: usize,
}

#[derive(Clone, Copy)]
struct ExtractorInput {
    text_len: usize,
    max_fields: usize,
}

impl<'a> Pipeline<'a> for ExtractorPipeline {
    type Input = ExtractorInput;
    type Output = (Vec<usize>, Vec<usize>);
    type Context = ();
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        let hidden = self.hidden_size;
        let max_width = self.max_width;
        move |input: ExtractorInput| {
            let text_emb = Array2::<f32>::zeros((input.text_len, hidden));
            let schema_emb_padded = Array2::<f32>::zeros((1 + input.max_fields, hidden));
            let schema_mask = Array1::<bool>::from_elem(input.max_fields, true);
            let spans_idx = build_spans(input.text_len, max_width);

            let inputs = ort::inputs! {
                "text_emb" => text_emb,
                "schema_emb_padded" => schema_emb_padded,
                "schema_mask" => schema_mask,
                "spans_idx" => spans_idx,
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
            let spans = outputs
                .get("span_scores")
                .context("missing span_scores")?
                .try_extract_tensor::<f32>()?;
            Ok((count.shape().to_vec(), spans.shape().to_vec()))
        }
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        static INPUTS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
            HashSet::from_iter(["text_emb", "schema_emb_padded", "schema_mask", "spans_idx"])
        });
        Some(&INPUTS)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        static OUTPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["count_logits", "span_scores"]));
        Some(&OUTPUTS)
    }
}

fn build_spans(text_len: usize, max_width: usize) -> Array2<i64> {
    let num_spans = text_len * max_width;
    let mut spans = Array2::<i64>::zeros((num_spans, 2));
    let mut idx = 0;
    for start in 0..text_len {
        for w in 0..max_width {
            if start + w < text_len {
                spans[[idx, 0]] = start as i64;
                spans[[idx, 1]] = (start + w) as i64;
            } else {
                spans[[idx, 0]] = -1;
                spans[[idx, 1]] = -1;
            }
            idx += 1;
        }
    }
    spans
}

fn main() -> Result<()> {
    // Paths are relative to the crate root when running `cargo run --example smoke` inside gliner2-rs.
    let paths = model_paths_from_args("../onnx/gliner2-base-v1");
    let encoder_path = paths.encoder;
    let extractor_path = paths.extractor;

    // Encoder
    let encoder = Model::new(&encoder_path, RuntimeParameters::default())
        .map_err(|e| anyhow!(e.to_string()))?;
    let encoder_out = encoder
        .inference(EncoderInput { batch: 2, seq: 8 }, &EncoderPipeline, &())
        .map_err(|e| anyhow!(e.to_string()))?;
    println!("encoder last_hidden_state shape: {:?}", encoder_out);

    // Extractor
    let extractor = Model::new(extractor_path, RuntimeParameters::default())
        .map_err(|e| anyhow!(e.to_string()))?;
    let extractor_out = extractor
        .inference(
            ExtractorInput {
                text_len: 6,
                max_fields: 64,
            },
            &ExtractorPipeline {
                hidden_size: 768,
                max_width: 8,
            },
            &(),
        )
        .map_err(|e| anyhow!(e.to_string()))?;
    println!(
        "extractor count_logits shape: {:?}, span_scores shape: {:?}",
        extractor_out.0, extractor_out.1
    );

    Ok(())
}
