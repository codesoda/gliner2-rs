use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array1, Array2, Array3, Ix1, Ix2, Ix3, arr0};

use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &[
    "field_query_states",
    "field_candidate_states",
    "field_candidate_mask",
    "context_states",
    "context_mask",
    "seed_states",
    "seed_object_logits",
    "mode",
];
const OUTPUTS: &[&str] = &["instance_states", "object_logits", "assignment_logits"];

/// Positive-dimension, batch-free inputs for `boundary_records.onnx`.
#[derive(Debug)]
pub struct RecordInput {
    /// Per-field query states, `[F,H]`.
    pub field_query_states: Array2<f32>,
    /// Padded field candidates, `[F,C,H]`.
    pub field_candidate_states: Array3<f32>,
    /// Live candidate columns, `[F,C]`.
    pub field_candidate_mask: Array2<bool>,
    /// Candidate context used by anchorless attention, `[M,H]`.
    pub context_states: Array2<f32>,
    pub context_mask: Array1<bool>,
    /// Natural or latent instance seeds, `[Ni,H]`. Anchorless callers supply one dummy row.
    pub seed_states: Array2<f32>,
    pub seed_object_logits: Array1<f32>,
    /// Natural=0, latent=1, anchorless=2.
    pub mode: i64,
}

impl RecordInput {
    /// Validate the graph ABI without loading ONNX Runtime.
    pub fn validate(&self) -> Result<()> {
        validate_input(self)
    }
}

/// Learned record-instance and field-assignment outputs.
#[derive(Debug)]
pub struct RecordOutput {
    /// `[J,H]`; `J=32` in anchorless mode and `J=Ni` otherwise.
    pub instance_states: Array2<f32>,
    pub object_logits: Array1<f32>,
    /// `[F,J,C+1]`, with the ABSENT alternative at column zero.
    pub assignment_logits: Array3<f32>,
}

/// Low-level ONNX wrapper for the GLiNER2.5 `RecordHead.forward_group` kernel.
pub struct RecordModel {
    session: RuntimeSession,
}

impl RecordModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load(model_path, "boundary records", INPUTS, OUTPUTS)?,
        })
    }

    pub fn infer(&self, input: &RecordInput) -> Result<RecordOutput> {
        validate_input(input)?;
        let fields = input.field_query_states.shape()[0];
        let hidden = input.field_query_states.shape()[1];
        let candidates = input.field_candidate_states.shape()[1];
        let seeds = input.seed_states.shape()[0];
        let mode = arr0(input.mode);
        let inputs = ort::inputs! {
            "field_query_states" => tensor(&input.field_query_states)?,
            "field_candidate_states" => tensor(&input.field_candidate_states)?,
            "field_candidate_mask" => tensor(&input.field_candidate_mask)?,
            "context_states" => tensor(&input.context_states)?,
            "context_mask" => tensor(&input.context_mask)?,
            "seed_states" => tensor(&input.seed_states)?,
            "seed_object_logits" => tensor(&input.seed_object_logits)?,
            "mode" => tensor(&mode)?,
        };
        self.session.run(inputs, |outputs| {
            let instance_states = extract::<f32, Ix2>(outputs, "instance_states")?;
            let object_logits = extract::<f32, Ix1>(outputs, "object_logits")?;
            let assignment_logits = extract::<f32, Ix3>(outputs, "assignment_logits")?;
            let instances = if input.mode == 2 { 32 } else { seeds };
            let assignment_width = candidates
                .checked_add(1)
                .context("assignment width overflow")?;

            ensure_shape2(
                "instance_states",
                instance_states.shape(),
                [instances, hidden],
            )?;
            ensure_shape1("object_logits", object_logits.shape(), [instances])?;
            ensure_shape3(
                "assignment_logits",
                assignment_logits.shape(),
                [fields, instances, assignment_width],
            )?;
            ensure_finite("instance_states", instance_states.iter())?;
            ensure_finite("object_logits", object_logits.iter())?;
            ensure_finite("assignment_logits", assignment_logits.iter())?;

            Ok(RecordOutput {
                instance_states,
                object_logits,
                assignment_logits,
            })
        })
    }
}

fn validate_input(input: &RecordInput) -> Result<()> {
    let [fields, hidden] = shape2("field_query_states", &input.field_query_states)?;
    let [candidate_fields, candidates, candidate_hidden] =
        shape3("field_candidate_states", &input.field_candidate_states)?;
    let [context_count, context_hidden] = shape2("context_states", &input.context_states)?;
    let [seed_count, seed_hidden] = shape2("seed_states", &input.seed_states)?;

    // ORT 1.20 can fail in native code on these zero dimensions. Natural and
    // latent Ni=0 are caller bypasses. Anchorless uses one masked zero context
    // row and one dummy seed row rather than passing zero-sized tensors.
    ensure!(
        fields > 0,
        "F=0 inputs must bypass the record graph before native inference"
    );
    ensure!(
        candidates > 0,
        "C=0 inputs must bypass the record graph before native inference"
    );
    ensure!(
        context_count > 0,
        "M=0 inputs must use one masked zero context row"
    );
    ensure!(
        seed_count > 0,
        "Ni=0 inputs must bypass natural/latent record inference; anchorless requires a dummy seed row"
    );
    ensure!(hidden > 0, "H=0 record hidden width is invalid");

    ensure!(
        candidate_fields == fields,
        "field count mismatch: query={fields}, candidates={candidate_fields}"
    );
    ensure!(
        candidate_hidden == hidden && context_hidden == hidden && seed_hidden == hidden,
        "record hidden-width mismatch: query={hidden}, candidates={candidate_hidden}, context={context_hidden}, seeds={seed_hidden}"
    );
    ensure_shape2(
        "field_candidate_mask",
        input.field_candidate_mask.shape(),
        [fields, candidates],
    )?;
    ensure_shape1("context_mask", input.context_mask.shape(), [context_count])?;
    ensure_shape1(
        "seed_object_logits",
        input.seed_object_logits.shape(),
        [seed_count],
    )?;
    ensure!(
        (0..=2).contains(&input.mode),
        "record mode must be 0 (natural), 1 (latent), or 2 (anchorless), got {}",
        input.mode
    );

    for (name, finite) in [
        (
            "field_query_states",
            input
                .field_query_states
                .iter()
                .all(|value| value.is_finite()),
        ),
        (
            "field_candidate_states",
            input
                .field_candidate_states
                .iter()
                .all(|value| value.is_finite()),
        ),
        (
            "context_states",
            input.context_states.iter().all(|value| value.is_finite()),
        ),
        (
            "seed_states",
            input.seed_states.iter().all(|value| value.is_finite()),
        ),
        (
            "seed_object_logits",
            input
                .seed_object_logits
                .iter()
                .all(|value| value.is_finite()),
        ),
    ] {
        ensure!(finite, "{name} contains non-finite values");
    }
    Ok(())
}

fn shape2<T>(name: &str, values: &Array2<T>) -> Result<[usize; 2]> {
    values
        .shape()
        .try_into()
        .map_err(|_| anyhow!("{name} must have rank 2"))
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

fn ensure_shape1(name: &str, actual: &[usize], expected: [usize; 1]) -> Result<()> {
    ensure!(
        actual == expected,
        "{name} shape {actual:?}, expected {expected:?}"
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
