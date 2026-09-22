use std::path::Path;

use anyhow::{Context, anyhow, ensure};
use ndarray::{Array2, Array3, Ix2, Ix3};

use crate::Result;
use crate::options::RuntimeOptions;
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
    "candidate_compat",
];
const OUTPUTS: &[&str] = &[
    "pair_logits",
    "candidate_states",
    "null_logits",
    "count_log_rates",
];

/// Explicit shared-pool inputs for `boundary_scorer.onnx`.
#[derive(Debug)]
pub struct ScorerInput {
    pub boundary_states: Array3<f32>,
    pub text_states: Array3<f32>,
    pub text_mask: Array2<bool>,
    pub query_states: Array3<f32>,
    pub query_mask: Array2<bool>,
    pub start_logits: Array3<f32>,
    pub end_logits: Array3<f32>,
    pub inside_prefix: Array3<f32>,
    pub inside_prefix_mean: Array3<f32>,
    pub candidate_indices: Array3<i64>,
    pub candidate_mask: Array2<bool>,
    pub candidate_compat: Array2<f32>,
}

impl ScorerInput {
    /// Check shapes, finite values, and gather indices without loading a graph.
    pub fn validate(&self) -> Result<()> {
        validate_input(self)
    }
}

/// Learned scores and candidate representations from the shared scorer.
#[derive(Debug)]
pub struct ScorerOutput {
    /// Query-major pair scores, `[B,Q,C]`.
    pub pair_logits: Array3<f32>,
    /// Query-independent candidate states, `[B,C,H]`.
    pub candidate_states: Array3<f32>,
    pub null_logits: Array2<f32>,
    pub count_log_rates: Array2<f32>,
}

/// Low-level ONNX wrapper for the GLiNER2.5 shared boundary scorer graph.
pub struct ScorerModel {
    session: RuntimeSession,
}

impl ScorerModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_options(model_path, RuntimeOptions::default())
    }

    pub fn new_with_options(model_path: impl AsRef<Path>, options: RuntimeOptions) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load_with(
                model_path,
                "boundary scorer",
                INPUTS,
                OUTPUTS,
                options,
            )?,
        })
    }

    pub fn infer(&self, input: ScorerInput) -> Result<ScorerOutput> {
        validate_input(&input)?;
        let batch = input.boundary_states.shape()[0];
        let hidden = input.text_states.shape()[2];
        let queries = input.query_states.shape()[1];
        let candidates = input.candidate_indices.shape()[1];
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
            "candidate_compat" => tensor(&input.candidate_compat)?,
        };
        self.session.run(inputs, |outputs| {
            let pair_logits = extract::<f32, Ix3>(outputs, "pair_logits")?;
            let candidate_states = extract::<f32, Ix3>(outputs, "candidate_states")?;
            let null_logits = extract::<f32, Ix2>(outputs, "null_logits")?;
            let count_log_rates = extract::<f32, Ix2>(outputs, "count_log_rates")?;

            ensure_shape3(
                "pair_logits",
                pair_logits.shape(),
                [batch, queries, candidates],
            )?;
            ensure_shape3(
                "candidate_states",
                candidate_states.shape(),
                [batch, candidates, hidden],
            )?;
            ensure_shape2("null_logits", null_logits.shape(), [batch, queries])?;
            ensure_shape2("count_log_rates", count_log_rates.shape(), [batch, queries])?;
            ensure_finite("pair_logits", pair_logits.iter())?;
            ensure_finite("candidate_states", candidate_states.iter())?;
            ensure_finite("null_logits", null_logits.iter())?;
            ensure_finite("count_log_rates", count_log_rates.iter())?;

            Ok(ScorerOutput {
                pair_logits,
                candidate_states,
                null_logits,
                count_log_rates,
            })
        })
    }
}

/// Validate every shape and every value which could reach native ORT.
///
/// Exposed through `ScorerInput::validate` for model-free preflight checks.
fn validate_input(input: &ScorerInput) -> Result<()> {
    let [batch, boundary_count, boundary_width] =
        shape3("boundary_states", &input.boundary_states)?;
    let [text_batch, text_length, text_width] = shape3("text_states", &input.text_states)?;
    let [query_batch, query_count, query_width] = shape3("query_states", &input.query_states)?;
    let [candidate_batch, candidate_count, index_width] =
        shape3("candidate_indices", &input.candidate_indices)?;

    ensure!(batch > 0, "scorer batch must be non-empty");
    // ORT 1.20 may crash on these zero dimensions. They are defined caller
    // bypasses, and must never enter native inference.
    ensure!(
        text_length > 0,
        "L=0 inputs must bypass the scorer graph before native inference"
    );
    ensure!(
        query_count > 0,
        "Q=0 inputs must bypass the scorer graph before native inference"
    );
    ensure!(
        candidate_count > 0,
        "C=0 inputs must bypass the scorer graph before native inference"
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
    ensure!(text_width > 0, "text hidden width must be non-zero");
    ensure!(
        query_width == text_width,
        "text/query hidden-width mismatch: {text_width} != {query_width}"
    );
    ensure!(
        text_batch == batch && query_batch == batch && candidate_batch == batch,
        "input batch mismatch: boundary={batch}, text={text_batch}, query={query_batch}, candidates={candidate_batch}"
    );
    ensure!(
        index_width == 2,
        "candidate_indices must have shape [B,C,2], got {:?}",
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
    ensure_shape2(
        "candidate_mask",
        input.candidate_mask.shape(),
        [batch, candidate_count],
    )?;
    ensure_shape2(
        "candidate_compat",
        input.candidate_compat.shape(),
        [batch, candidate_count],
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
    ensure!(
        input.candidate_compat.iter().all(|value| value.is_finite()),
        "candidate_compat contains non-finite values"
    );

    // The graph gathers every row, including padding. Consequently even a
    // masked candidate must contain in-range endpoints. Only live candidates
    // are required to encode a non-empty half-open span.
    let boundary_limit = i64::try_from(boundary_count).context("boundary count exceeds i64")?;
    for batch_index in 0..batch {
        for candidate in 0..candidate_count {
            let start = input.candidate_indices[(batch_index, candidate, 0)];
            let end = input.candidate_indices[(batch_index, candidate, 1)];
            ensure!(
                (0..boundary_limit).contains(&start) && (0..boundary_limit).contains(&end),
                "candidate_indices[{batch_index},{candidate}] = [{start},{end}] is outside 0..N ({boundary_count})"
            );
            if input.candidate_mask[(batch_index, candidate)] {
                ensure!(
                    start < end && end <= i64::try_from(text_length)?,
                    "valid candidate [{start},{end}) at [{batch_index},{candidate}] must satisfy 0 <= start < end <= L ({text_length})"
                );
            }
        }
    }
    Ok(())
}

fn shape3<T>(name: &str, values: &Array3<T>) -> Result<[usize; 3]> {
    values
        .shape()
        .try_into()
        .map_err(|_| anyhow!("{name} must have rank 3"))
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
