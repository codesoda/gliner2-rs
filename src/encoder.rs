use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::Path;

use anyhow::{Context, anyhow};
use ndarray::{Array2, Array3, Ix3};
use once_cell::sync::Lazy;
use orp::{
    model::Model,
    params::RuntimeParameters,
    pipeline::{Pipeline, PostProcessor, PreProcessor},
};
use ort::session::SessionOutputs;

use crate::Result;

/// Thin wrapper around the encoder ONNX model.
pub struct Encoder {
    model: Model,
}

impl Encoder {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::new(model_path, RuntimeParameters::default())
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(Self { model })
    }

    /// Run encoder with already-built tensors.
    pub fn infer(
        &self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
    ) -> Result<Array3<f32>> {
        let out = self
            .model
            .inference(
                EncoderInput {
                    input_ids,
                    attention_mask,
                },
                &EncoderPipeline,
                &(),
            )
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(out)
    }
}

struct EncoderInput {
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
}

struct EncoderPipeline;

impl<'a> Pipeline<'a> for EncoderPipeline {
    type Input = EncoderInput;
    type Output = Array3<f32>;
    type Context = ();
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        |input: EncoderInput| {
            let inputs = ort::inputs! {
                "input_ids" => input.input_ids,
                "attention_mask" => input.attention_mask,
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
            let arr = view
                .into_dimensionality::<Ix3>()
                .map_err(|e| anyhow!("unexpected encoder output shape: {e}"))?
                .to_owned();
            Ok(arr)
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
