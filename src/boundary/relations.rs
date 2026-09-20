use std::path::Path;

use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array3, Ix1};

use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &[
    "text_states",
    "relation_query_states",
    "batch_index",
    "relation_index",
    "head_start",
    "head_end",
    "tail_start",
    "tail_end",
    "pair_mask",
];
const OUTPUTS: &[&str] = &["relation_logits"];

/// Typed, owned inputs for the learned `boundary_relations.onnx` graph.
///
/// `text_states` are the actual encoder text states. Despite the upstream
/// scorer's parameter name, they are not boundary-encoder states. Coordinates
/// are half-open word indices. Every pair must have valid routing and endpoints,
/// including pairs disabled by `pair_mask`.
#[derive(Debug)]
pub struct RelationInput {
    /// Encoder text states, `[B,L,H]`.
    pub text_states: Array3<f32>,
    /// Directional relation query states, `[B,R,2H]`, concatenated head then tail.
    pub relation_query_states: Array3<f32>,
    /// Flattened pair routing, each `[P]`.
    pub batch_index: Array1<i64>,
    pub relation_index: Array1<i64>,
    /// Half-open word coordinates, each `[P]`.
    pub head_start: Array1<i64>,
    pub head_end: Array1<i64>,
    pub tail_start: Array1<i64>,
    pub tail_end: Array1<i64>,
    /// False pairs are retained but receive an exact positive-zero logit.
    pub pair_mask: Array1<bool>,
}

impl RelationInput {
    /// Validate the public graph contract without loading ONNX Runtime.
    pub fn validate(&self) -> Result<()> {
        validate_input(self)
    }
}

/// Owned learned relation scores in flattened pair order.
#[derive(Debug)]
pub struct RelationOutput {
    /// Raw logits, `[P]`. A false `pair_mask` entry is exactly `+0.0`.
    pub relation_logits: Array1<f32>,
}

/// Low-level direct-ORT wrapper for the learned GLiNER2.5 relation scorer.
pub struct RelationModel {
    session: RuntimeSession,
}

impl RelationModel {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load(model_path, "boundary relations", INPUTS, OUTPUTS)?,
        })
    }

    pub fn infer(&self, input: &RelationInput) -> Result<RelationOutput> {
        validate_input(input)?;
        let pair_count = input.batch_index.len();
        let inputs = ort::inputs! {
            "text_states" => tensor(&input.text_states)?,
            "relation_query_states" => tensor(&input.relation_query_states)?,
            "batch_index" => tensor(&input.batch_index)?,
            "relation_index" => tensor(&input.relation_index)?,
            "head_start" => tensor(&input.head_start)?,
            "head_end" => tensor(&input.head_end)?,
            "tail_start" => tensor(&input.tail_start)?,
            "tail_end" => tensor(&input.tail_end)?,
            "pair_mask" => tensor(&input.pair_mask)?,
        };
        self.session.run(inputs, |outputs| {
            let relation_logits = extract::<f32, Ix1>(outputs, "relation_logits")?;
            ensure!(
                relation_logits.shape() == [pair_count],
                "relation_logits shape {:?}, expected [{pair_count}]",
                relation_logits.shape()
            );
            ensure!(
                relation_logits.iter().all(|value| value.is_finite()),
                "relation_logits contains non-finite values"
            );
            for (pair, (&enabled, &logit)) in input
                .pair_mask
                .iter()
                .zip(relation_logits.iter())
                .enumerate()
            {
                if !enabled {
                    ensure!(
                        logit.to_bits() == 0.0_f32.to_bits(),
                        "masked relation_logits[{pair}] must be exact +0.0, got {logit}"
                    );
                }
            }
            Ok(RelationOutput { relation_logits })
        })
    }
}

fn validate_input(input: &RelationInput) -> Result<()> {
    let [batch, length, hidden]: [usize; 3] = input
        .text_states
        .shape()
        .try_into()
        .context("text_states must have shape [B,L,H]")?;
    let [relation_batch, relations, relation_width]: [usize; 3] = input
        .relation_query_states
        .shape()
        .try_into()
        .context("relation_query_states must have shape [B,R,2H]")?;
    let pair_count = input.batch_index.len();

    // P=0 and no-query calls are public caller bypasses. Keeping every graph
    // axis positive avoids native-runtime empty-axis behavior.
    ensure!(batch > 0, "B=0 must bypass relation inference before ORT");
    ensure!(length > 0, "L=0 must bypass relation inference before ORT");
    ensure!(hidden > 0, "H=0 relation hidden width is invalid");
    ensure!(
        relations > 0,
        "R=0 must bypass relation inference before ORT"
    );
    ensure!(
        pair_count > 0,
        "P=0 must bypass relation inference before ORT"
    );
    ensure!(
        relation_batch == batch,
        "relation batch mismatch: text={batch}, relation={relation_batch}"
    );
    let expected_relation_width = hidden
        .checked_mul(2)
        .context("relation query width overflow")?;
    ensure!(
        relation_width == expected_relation_width,
        "relation query width must equal 2H: got {relation_width}, expected {expected_relation_width}"
    );

    for (name, length) in [
        ("relation_index", input.relation_index.len()),
        ("head_start", input.head_start.len()),
        ("head_end", input.head_end.len()),
        ("tail_start", input.tail_start.len()),
        ("tail_end", input.tail_end.len()),
        ("pair_mask", input.pair_mask.len()),
    ] {
        ensure!(
            length == pair_count,
            "{name} length {length}, expected pair count {pair_count}"
        );
    }

    ensure!(
        input.text_states.iter().all(|value| value.is_finite()),
        "text_states contains non-finite values"
    );
    ensure!(
        input
            .relation_query_states
            .iter()
            .all(|value| value.is_finite()),
        "relation_query_states contains non-finite values"
    );

    for pair in 0..pair_count {
        let batch_index = input.batch_index[pair];
        ensure!(
            batch_index >= 0 && usize::try_from(batch_index).is_ok_and(|value| value < batch),
            "batch_index[{pair}]={batch_index} is outside [0,{batch})"
        );
        let relation_index = input.relation_index[pair];
        ensure!(
            relation_index >= 0
                && usize::try_from(relation_index).is_ok_and(|value| value < relations),
            "relation_index[{pair}]={relation_index} is outside [0,{relations})"
        );
        validate_span(
            "head",
            pair,
            input.head_start[pair],
            input.head_end[pair],
            length,
        )?;
        validate_span(
            "tail",
            pair,
            input.tail_start[pair],
            input.tail_end[pair],
            length,
        )?;
    }
    Ok(())
}

fn validate_span(name: &str, pair: usize, start: i64, end: i64, length: usize) -> Result<()> {
    let valid =
        start >= 0 && end > start && usize::try_from(end).is_ok_and(|value| value <= length);
    ensure!(
        valid,
        "{name} coordinates at pair {pair} must satisfy 0 <= start < end <= L={length}, got [{start},{end})"
    );
    Ok(())
}
