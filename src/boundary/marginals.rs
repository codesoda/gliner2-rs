use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::Path;

use anyhow::{Context, anyhow, ensure};
use ndarray::{Array2, Array3, Ix2, Ix3};
use once_cell::sync::Lazy;
use orp::{
    model::Model,
    params::RuntimeParameters,
    pipeline::{Pipeline, PostProcessor, PreProcessor},
};
use ort::session::SessionOutputs;

use crate::Result;

/// Typed inputs for `boundary_marginals.onnx`.
#[derive(Debug)]
pub struct MarginalInput {
    pub text_states: Array3<f32>,
    pub text_mask: Array2<bool>,
    pub query_states: Array3<f32>,
    pub query_mask: Array2<bool>,
}

/// Boundary encoding, query marginals, and shared-pool endpoint projections.
#[derive(Debug)]
pub struct MarginalOutput {
    pub boundary_states: Array3<f32>,
    pub boundary_mask: Array2<bool>,
    pub start_logits: Array3<f32>,
    pub end_logits: Array3<f32>,
    pub inside_logits: Array3<f32>,
    pub inside_prefix: Array3<f32>,
    pub inside_prefix_mean: Array3<f32>,
    pub start_all: Array3<f32>,
    pub end_all: Array3<f32>,
}

/// Low-level ONNX wrapper for the GLiNER2.5 marginal graph.
pub struct MarginalModel {
    model: Model,
}

impl MarginalModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model = Model::new(model_path, RuntimeParameters::default())
            .map_err(|error| anyhow!(error.to_string()))?;
        Ok(Self { model })
    }

    pub fn infer(&self, input: MarginalInput) -> Result<MarginalOutput> {
        validate_input(&input)?;
        self.model
            .inference(input, &MarginalPipeline, &())
            .map_err(|error| anyhow!(error.to_string()))
    }
}

fn validate_input(input: &MarginalInput) -> Result<()> {
    let [text_batch, text_length, text_width] = input
        .text_states
        .shape()
        .try_into()
        .map_err(|_| anyhow!("text_states must have shape [B,L,H]"))?;
    let [query_batch, query_count, query_width] = input
        .query_states
        .shape()
        .try_into()
        .map_err(|_| anyhow!("query_states must have shape [B,Q,H]"))?;

    ensure!(text_batch > 0, "marginal batch must be non-empty");
    // ORT 1.20 can crash on the empty sequence. Reject before entering native
    // inference; the high-level pipeline normalizes empty text to ".".
    ensure!(text_length > 0, "marginal text length must be non-zero");
    // Do not cap L here: the pipeline truncates original words before adding
    // choice prefixes, so the graph's text-state length may exceed max_len.
    ensure!(text_width > 0, "text hidden width must be non-zero");
    ensure!(
        text_batch == query_batch,
        "text/query batch mismatch: {text_batch} != {query_batch}"
    );
    ensure!(
        text_width == query_width,
        "text/query hidden-width mismatch: {text_width} != {query_width}"
    );
    ensure!(
        query_count > 0,
        "Q=0 classification-only inputs must bypass the marginal graph"
    );
    ensure!(
        input.text_mask.shape() == [text_batch, text_length],
        "text_mask shape {:?} does not match [{text_batch},{text_length}]",
        input.text_mask.shape()
    );
    ensure!(
        input.query_mask.shape() == [query_batch, query_count],
        "query_mask shape {:?} does not match [{query_batch},{query_count}]",
        input.query_mask.shape()
    );
    ensure!(
        input.text_states.iter().all(|value| value.is_finite()),
        "text_states contains non-finite values"
    );
    ensure!(
        input.query_states.iter().all(|value| value.is_finite()),
        "query_states contains non-finite values"
    );
    Ok(())
}

struct MarginalPipeline;

impl<'a> Pipeline<'a> for MarginalPipeline {
    type Input = MarginalInput;
    type Output = MarginalOutput;
    type Context = (usize, usize, usize);
    type Parameters = ();

    fn pre_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        |input: MarginalInput| {
            let batch = input.text_states.shape()[0];
            let text_length = input.text_states.shape()[1];
            let query_count = input.query_states.shape()[1];
            let inputs = ort::inputs! {
                "text_states" => input.text_states,
                "text_mask" => input.text_mask,
                "query_states" => input.query_states,
                "query_mask" => input.query_mask,
            }?;
            Ok((inputs.into(), (batch, text_length, query_count)))
        }
    }

    fn post_processor(
        &self,
        _: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        |(outputs, (batch, text_length, query_count)): (
            SessionOutputs<'_, '_>,
            (usize, usize, usize),
        )| {
            let boundary_states = extract_f32_3(&outputs, "boundary_states")?;
            let boundary_mask = extract_bool_2(&outputs, "boundary_mask")?;
            let start_logits = extract_f32_3(&outputs, "start_logits")?;
            let end_logits = extract_f32_3(&outputs, "end_logits")?;
            let inside_logits = extract_f32_3(&outputs, "inside_logits")?;
            let inside_prefix = extract_f32_3(&outputs, "inside_prefix")?;
            let inside_prefix_mean = extract_f32_3(&outputs, "inside_prefix_mean")?;
            let start_all = extract_f32_3(&outputs, "start_all")?;
            let end_all = extract_f32_3(&outputs, "end_all")?;

            validate_output(
                batch,
                text_length,
                query_count,
                &boundary_states,
                &boundary_mask,
                &start_logits,
                &end_logits,
                &inside_logits,
                &inside_prefix,
                &inside_prefix_mean,
                &start_all,
                &end_all,
            )?;

            Ok(MarginalOutput {
                boundary_states,
                boundary_mask,
                start_logits,
                end_logits,
                inside_logits,
                inside_prefix,
                inside_prefix_mean,
                start_all,
                end_all,
            })
        }
    }

    fn expected_inputs(&self) -> Option<&HashSet<&str>> {
        static INPUTS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
            HashSet::from_iter(["text_states", "text_mask", "query_states", "query_mask"])
        });
        Some(&INPUTS)
    }

    fn expected_outputs(&self) -> Option<&HashSet<&str>> {
        static OUTPUTS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
            HashSet::from_iter([
                "boundary_states",
                "boundary_mask",
                "start_logits",
                "end_logits",
                "inside_logits",
                "inside_prefix",
                "inside_prefix_mean",
                "start_all",
                "end_all",
            ])
        });
        Some(&OUTPUTS)
    }
}

fn extract_f32_3(outputs: &SessionOutputs<'_, '_>, name: &str) -> Result<Array3<f32>> {
    outputs
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .try_extract_tensor::<f32>()?
        .into_dimensionality::<Ix3>()
        .map_err(|error| anyhow!("unexpected {name} shape: {error}"))
        .map(|value| value.to_owned())
}

fn extract_bool_2(outputs: &SessionOutputs<'_, '_>, name: &str) -> Result<Array2<bool>> {
    outputs
        .get(name)
        .with_context(|| format!("missing {name}"))?
        .try_extract_tensor::<bool>()?
        .into_dimensionality::<Ix2>()
        .map_err(|error| anyhow!("unexpected {name} shape: {error}"))
        .map(|value| value.to_owned())
}

#[allow(clippy::too_many_arguments)]
fn validate_output(
    batch: usize,
    text_length: usize,
    query_count: usize,
    boundary_states: &Array3<f32>,
    boundary_mask: &Array2<bool>,
    start_logits: &Array3<f32>,
    end_logits: &Array3<f32>,
    inside_logits: &Array3<f32>,
    inside_prefix: &Array3<f32>,
    inside_prefix_mean: &Array3<f32>,
    start_all: &Array3<f32>,
    end_all: &Array3<f32>,
) -> Result<()> {
    let boundary_count = text_length
        .checked_add(1)
        .context("boundary count overflow")?;
    ensure!(
        boundary_states.shape()[0..2] == [batch, boundary_count],
        "boundary_states shape {:?}, expected [{batch},{boundary_count},D]",
        boundary_states.shape()
    );
    let boundary_width = boundary_states.shape()[2];
    ensure!(boundary_width > 0, "boundary width must be non-zero");
    ensure_shape_2(
        "boundary_mask",
        boundary_mask.shape(),
        [batch, boundary_count],
    )?;
    ensure_shape_3(
        "start_logits",
        start_logits.shape(),
        [batch, query_count, boundary_count],
    )?;
    ensure_shape_3(
        "end_logits",
        end_logits.shape(),
        [batch, query_count, boundary_count],
    )?;
    ensure_shape_3(
        "inside_logits",
        inside_logits.shape(),
        [batch, query_count, text_length],
    )?;
    ensure_shape_3(
        "inside_prefix",
        inside_prefix.shape(),
        [batch, query_count, boundary_count],
    )?;
    ensure_shape_3(
        "inside_prefix_mean",
        inside_prefix_mean.shape(),
        [batch, query_count, 1],
    )?;
    ensure_shape_3(
        "start_all",
        start_all.shape(),
        [batch, boundary_count, boundary_width],
    )?;
    ensure_shape_3(
        "end_all",
        end_all.shape(),
        [batch, boundary_count, boundary_width],
    )?;

    for (name, values) in [
        ("boundary_states", boundary_states),
        ("start_logits", start_logits),
        ("end_logits", end_logits),
        ("inside_logits", inside_logits),
        ("inside_prefix", inside_prefix),
        ("inside_prefix_mean", inside_prefix_mean),
        ("start_all", start_all),
        ("end_all", end_all),
    ] {
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "{name} contains non-finite values"
        );
    }
    Ok(())
}

fn ensure_shape_2(name: &str, actual: &[usize], expected: [usize; 2]) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} shape {actual:?}, expected {expected:?}"
    );
    Ok(())
}

fn ensure_shape_3(name: &str, actual: &[usize], expected: [usize; 3]) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} shape {actual:?}, expected {expected:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(text_length: usize, query_count: usize) -> MarginalInput {
        MarginalInput {
            text_states: Array3::zeros((1, text_length, 1)),
            text_mask: Array2::from_elem((1, text_length), true),
            query_states: Array3::zeros((1, query_count, 1)),
            query_mask: Array2::from_elem((1, query_count), true),
        }
    }

    #[test]
    fn rejects_empty_text_before_native_inference() {
        let error = validate_input(&input(0, 1)).unwrap_err();
        assert!(error.to_string().contains("text length must be non-zero"));
    }

    #[test]
    fn permits_choice_prefix_beyond_original_word_limit() {
        validate_input(&input(4097, 1)).unwrap();
    }

    #[test]
    fn classification_only_must_bypass_graph() {
        let error = validate_input(&input(1, 0)).unwrap_err();
        assert!(error.to_string().contains("must bypass"));
    }
}
