use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array2, Array3, Array4, Ix3};

use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &[
    "boundary_states",
    "text_states",
    "text_mask",
    "query_states",
    "query_mask",
    "start_logits",
    "end_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "candidate_indices",
    "candidate_mask",
];
const OUTPUTS: &[&str] = &["pair_logits", "compatibility", "legal_mask"];

const MASK_LOGIT: f32 = -10_000.0;

/// Inputs for the query-specific explicit-span scorer.
#[derive(Debug)]
pub struct ExplicitInput {
    pub boundary_states: Array3<f32>,
    pub text_states: Array3<f32>,
    pub text_mask: Array2<bool>,
    pub query_states: Array3<f32>,
    pub query_mask: Array2<bool>,
    pub start_logits: Array3<f32>,
    pub end_logits: Array3<f32>,
    pub inside_prefix: Array3<f32>,
    pub inside_prefix_mean: Array3<f32>,
    /// Caller-supplied half-open spans, `[B,Q,C,2]`.
    pub candidate_indices: Array4<i64>,
    pub candidate_mask: Array3<bool>,
}

impl ExplicitInput {
    /// Validate tensor ranks, dimensions, and finite inputs without loading ORT.
    ///
    /// Candidate coordinates are deliberately not range-validated. The graph
    /// computes legality itself and safely clamps every gather, so negative,
    /// reversed, out-of-range, and masked coordinates are valid ABI inputs.
    pub fn validate(&self) -> Result<()> {
        validate_input(self).map(|_| ())
    }
}

/// Learned explicit-span scores and their graph-computed legality.
#[derive(Debug)]
pub struct ExplicitOutput {
    pub pair_logits: Array3<f32>,
    pub compatibility: Array3<f32>,
    pub legal_mask: Array3<bool>,
}

/// Low-level ONNX wrapper for `boundary_explicit_scorer.onnx`.
pub struct ExplicitModel {
    session: RuntimeSession,
}

impl ExplicitModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load(model_path, "boundary explicit scorer", INPUTS, OUTPUTS)?,
        })
    }

    pub fn infer(&self, input: ExplicitInput) -> Result<ExplicitOutput> {
        let expected_legal = validate_input(&input)?;
        let inputs = ort::inputs! {
            "boundary_states" => tensor(&input.boundary_states)?,
            "text_states" => tensor(&input.text_states)?,
            "text_mask" => tensor(&input.text_mask)?,
            "query_states" => tensor(&input.query_states)?,
            "query_mask" => tensor(&input.query_mask)?,
            "start_logits" => tensor(&input.start_logits)?,
            "end_logits" => tensor(&input.end_logits)?,
            "inside_prefix" => tensor(&input.inside_prefix)?,
            "inside_prefix_mean" => tensor(&input.inside_prefix_mean)?,
            "candidate_indices" => tensor(&input.candidate_indices)?,
            "candidate_mask" => tensor(&input.candidate_mask)?,
        };
        self.session.run(inputs, |outputs| {
            let pair_logits = extract::<f32, Ix3>(outputs, "pair_logits")?;
            let compatibility = extract::<f32, Ix3>(outputs, "compatibility")?;
            let legal_mask = extract::<bool, Ix3>(outputs, "legal_mask")?;
            validate_output(&pair_logits, &compatibility, &legal_mask, &expected_legal)?;

            Ok(ExplicitOutput {
                pair_logits,
                compatibility,
                legal_mask,
            })
        })
    }
}

fn validate_input(input: &ExplicitInput) -> Result<Array3<bool>> {
    let [batch, boundary_count, boundary_width] =
        shape3("boundary_states", &input.boundary_states)?;
    let [text_batch, text_length, text_width] = shape3("text_states", &input.text_states)?;
    let [query_batch, query_count, query_width] = shape3("query_states", &input.query_states)?;
    let [
        candidate_batch,
        candidate_queries,
        candidate_count,
        index_width,
    ] = shape4("candidate_indices", &input.candidate_indices)?;

    // These dimensions are caller bypasses. ORT 1.20 rejects at least Q=0 and
    // C=0 in native reshape kernels, so none may reach the session.
    ensure!(
        batch > 0,
        "B=0 inputs must bypass the explicit scorer graph"
    );
    ensure!(
        text_length > 0,
        "L=0 inputs must bypass the explicit scorer graph"
    );
    ensure!(
        query_count > 0,
        "Q=0 inputs must bypass the explicit scorer graph"
    );
    ensure!(
        candidate_count > 0,
        "C=0 inputs must bypass the explicit scorer graph"
    );
    ensure!(
        text_width > 0,
        "H=0 explicit scorer hidden width is invalid"
    );

    let expected_boundaries = text_length
        .checked_add(1)
        .context("boundary count overflow")?;
    ensure!(
        boundary_count == expected_boundaries,
        "boundary count N={boundary_count} must equal L+1={expected_boundaries}"
    );
    ensure!(
        boundary_width == 128,
        "boundary_states width must be 128, got {boundary_width}"
    );
    ensure!(
        query_width == text_width,
        "text/query hidden-width mismatch: {text_width} != {query_width}"
    );
    ensure!(
        text_batch == batch && query_batch == batch && candidate_batch == batch,
        "input batch mismatch: boundary={batch}, text={text_batch}, query={query_batch}, candidates={candidate_batch}"
    );
    ensure!(
        candidate_queries == query_count,
        "candidate/query count mismatch: candidates={candidate_queries}, queries={query_count}"
    );
    ensure!(
        index_width == 2,
        "candidate_indices must have shape [B,Q,C,2], got {:?}",
        input.candidate_indices.shape()
    );

    ensure_shape2("text_mask", input.text_mask.shape(), [batch, text_length])?;
    ensure_shape2("query_mask", input.query_mask.shape(), [batch, query_count])?;
    for (name, values) in [
        ("start_logits", &input.start_logits),
        ("end_logits", &input.end_logits),
        ("inside_prefix", &input.inside_prefix),
    ] {
        ensure_shape3(name, values.shape(), [batch, query_count, boundary_count])?;
    }
    ensure_shape3(
        "inside_prefix_mean",
        input.inside_prefix_mean.shape(),
        [batch, query_count, 1],
    )?;
    ensure_shape3(
        "candidate_mask",
        input.candidate_mask.shape(),
        [batch, query_count, candidate_count],
    )?;

    for (name, values) in [
        ("boundary_states", &input.boundary_states),
        ("text_states", &input.text_states),
        ("query_states", &input.query_states),
        ("start_logits", &input.start_logits),
        ("end_logits", &input.end_logits),
        ("inside_prefix", &input.inside_prefix),
        ("inside_prefix_mean", &input.inside_prefix_mean),
    ] {
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "{name} contains non-finite values"
        );
    }

    // Match the source method: text length is the count of true text-mask
    // entries, rather than the padded L dimension. Comparisons only are used;
    // no coordinate subtraction can overflow for i64::MIN/MAX inputs.
    let mut text_lengths = Vec::with_capacity(batch);
    for batch_index in 0..batch {
        let length = input
            .text_mask
            .row(batch_index)
            .iter()
            .filter(|&&value| value)
            .count();
        text_lengths.push(i64::try_from(length).context("text length exceeds i64")?);
    }
    let legal = Array3::from_shape_fn(
        (batch, query_count, candidate_count),
        |(batch_index, query, candidate)| {
            let start = input.candidate_indices[(batch_index, query, candidate, 0)];
            let end = input.candidate_indices[(batch_index, query, candidate, 1)];
            start >= 0
                && end > start
                && end <= text_lengths[batch_index]
                && input.query_mask[(batch_index, query)]
                && input.candidate_mask[(batch_index, query, candidate)]
        },
    );
    Ok(legal)
}

fn shape3<T>(name: &str, values: &Array3<T>) -> Result<[usize; 3]> {
    values
        .shape()
        .try_into()
        .map_err(|_| anyhow!("{name} must have rank 3"))
}

fn shape4<T>(name: &str, values: &Array4<T>) -> Result<[usize; 4]> {
    values
        .shape()
        .try_into()
        .map_err(|_| anyhow!("{name} must have rank 4"))
}

fn validate_output(
    pair_logits: &Array3<f32>,
    compatibility: &Array3<f32>,
    legal_mask: &Array3<bool>,
    expected_legal: &Array3<bool>,
) -> Result<()> {
    let shape = [
        expected_legal.shape()[0],
        expected_legal.shape()[1],
        expected_legal.shape()[2],
    ];
    ensure_shape3("pair_logits", pair_logits.shape(), shape)?;
    ensure_shape3("compatibility", compatibility.shape(), shape)?;
    ensure_shape3("legal_mask", legal_mask.shape(), shape)?;
    ensure_finite("pair_logits", pair_logits.iter())?;
    ensure_finite("compatibility", compatibility.iter())?;
    ensure!(
        legal_mask == expected_legal,
        "legal_mask differs from the explicit scorer input contract"
    );
    for (index, &legal) in legal_mask.indexed_iter() {
        if !legal {
            ensure!(
                compatibility[index].to_bits() == 0.0_f32.to_bits(),
                "illegal compatibility at {index:?} is not exact +0"
            );
            ensure!(
                pair_logits[index].to_bits() == MASK_LOGIT.to_bits(),
                "illegal pair logit at {index:?} is not exact {MASK_LOGIT}"
            );
        }
    }
    Ok(())
}

fn ensure_finite<'a>(name: &str, values: impl Iterator<Item = &'a f32>) -> Result<()> {
    ensure!(
        values.into_iter().all(|value| value.is_finite()),
        "{name} contains non-finite values"
    );
    Ok(())
}

fn ensure_shape2(name: &str, actual: &[usize], expected: [usize; 2]) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} shape {actual:?}, expected {expected:?}"
    );
    Ok(())
}

fn ensure_shape3(name: &str, actual: &[usize], expected: [usize; 3]) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} shape {actual:?}, expected {expected:?}"
    );
    Ok(())
}
