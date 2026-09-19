use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::Path;

use anyhow::{Context, anyhow};
use ndarray::{Array1, Array2, Ix1};
use once_cell::sync::Lazy;
use orp::{
    model::Model,
    params::RuntimeParameters,
    pipeline::{Pipeline, PostProcessor, PreProcessor},
};
use ort::session::SessionOutputs;

use crate::Result;

pub struct Classifier {
    model: Model,
}

impl Classifier {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::new(model_path, RuntimeParameters::default())
            .map_err(|e| anyhow!(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn infer(&self, cls_embeds: Array2<f32>) -> Result<Array1<f32>> {
        self.model
            .inference(ClassifierInput { cls_embeds }, &ClassifierPipeline, &())
            .map_err(|e| anyhow!(e.to_string()))
    }
}

struct ClassifierInput {
    cls_embeds: Array2<f32>,
}

struct ClassifierPipeline;

impl<'a> Pipeline<'a> for ClassifierPipeline {
    type Input = ClassifierInput;
    type Output = Array1<f32>;
    type Context = ();
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        |input: ClassifierInput| {
            let inputs = ort::inputs! {
                "cls_embeds" => input.cls_embeds,
            }?;
            Ok((inputs.into(), ()))
        }
    }

    fn post_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        |(outputs, _ctx): (SessionOutputs<'_, '_>, ())| {
            let logits = outputs
                .get("logits")
                .context("missing logits")?
                .try_extract_tensor::<f32>()?;
            let logits = logits
                .into_dimensionality::<Ix1>()
                .map_err(|e| anyhow!("unexpected logits shape: {e}"))?
                .to_owned();
            Ok(logits)
        }
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        static INPUTS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| HashSet::from_iter(["cls_embeds"]));
        Some(&INPUTS)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        static OUTPUTS: Lazy<HashSet<&'static str>> = Lazy::new(|| HashSet::from_iter(["logits"]));
        Some(&OUTPUTS)
    }
}
