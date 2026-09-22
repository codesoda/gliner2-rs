use std::path::Path;

use anyhow::{Context, anyhow, ensure};
use ndarray::{Array2, Array3, Ix2, Ix3};

use crate::Result;
use crate::options::RuntimeOptions;
use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &["text_states", "text_mask", "query_states", "query_mask"];
const OUTPUTS: &[&str] = &[
    "boundary_states",
    "boundary_mask",
    "start_logits",
    "end_logits",
    "inside_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "start_all",
    "end_all",
];

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
    session: RuntimeSession,
}

impl MarginalModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_options(model_path, RuntimeOptions::default())
    }

    pub fn new_with_options(model_path: impl AsRef<Path>, options: RuntimeOptions) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load_with(
                model_path,
                "boundary marginal",
                INPUTS,
                OUTPUTS,
                options,
            )?,
        })
    }

    pub fn infer(&self, input: MarginalInput) -> Result<MarginalOutput> {
        validate_input(&input)?;
        let batch = input.text_states.shape()[0];
        let text_length = input.text_states.shape()[1];
        let query_count = input.query_states.shape()[1];
        let inputs = ort::inputs! {
            "text_states" => tensor(&input.text_states)?,
            "text_mask" => tensor(&input.text_mask)?,
            "query_states" => tensor(&input.query_states)?,
            "query_mask" => tensor(&input.query_mask)?,
        };
        self.session.run(inputs, |outputs| {
            let boundary_states = extract::<f32, Ix3>(outputs, "boundary_states")?;
            let boundary_mask = extract::<bool, Ix2>(outputs, "boundary_mask")?;
            let start_logits = extract::<f32, Ix3>(outputs, "start_logits")?;
            let end_logits = extract::<f32, Ix3>(outputs, "end_logits")?;
            let inside_logits = extract::<f32, Ix3>(outputs, "inside_logits")?;
            let inside_prefix = extract::<f32, Ix3>(outputs, "inside_prefix")?;
            let inside_prefix_mean = extract::<f32, Ix3>(outputs, "inside_prefix_mean")?;
            let start_all = extract::<f32, Ix3>(outputs, "start_all")?;
            let end_all = extract::<f32, Ix3>(outputs, "end_all")?;

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
        })
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
