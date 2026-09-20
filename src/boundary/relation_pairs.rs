//! Pure typed relation-pair proposals for batch-one boundary inference.
//!
//! Candidate endpoints are selected independently for each relation type from
//! raw scorer logits, then paired with a fixed-size Cartesian-product cap. This
//! module performs no native inference and accepts nonnegative-stride and
//! broadcast ndarray views without materializing a query-expanded candidate
//! pool. Negative-stride logit views are rejected because pinned PyTorch does
//! not expose an equivalent tensor layout for a normative sigmoid ordering.

use std::cmp::Ordering;

use anyhow::{Context, Result, ensure};
use ndarray::{ArrayView1, ArrayView2, ArrayView3};

/// Checkpoint relation-proposal settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelationProposalConfig {
    pub heads_per_relation: usize,
    pub tails_per_relation: usize,
    pub pair_cap: usize,
    pub argument_threshold: f32,
}

impl Default for RelationProposalConfig {
    fn default() -> Self {
        Self {
            heads_per_relation: 32,
            tails_per_relation: 32,
            pair_cap: 64,
            argument_threshold: 0.2,
        }
    }
}

/// Query membership and self-pair policy for one relation type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationTypeSpec {
    pub relation_type: String,
    pub head_query_ids: Vec<i64>,
    pub tail_query_ids: Vec<i64>,
    pub allow_self: bool,
}

/// One compact relation pair, in relation-major proposal rank order.
#[derive(Clone, Debug, PartialEq)]
pub struct RelationPair {
    pub relation_index: usize,
    pub head_query_id: usize,
    pub tail_query_id: usize,
    pub head: [usize; 2],
    pub tail: [usize; 2],
    pub head_probability: f32,
    pub tail_probability: f32,
}

#[derive(Clone, Copy, Debug)]
struct Endpoint {
    query_id: usize,
    span: [usize; 2],
    probability: f32,
    flat_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct PairProposal {
    head: Endpoint,
    tail: Endpoint,
    score: f32,
    cartesian_index: usize,
}

fn descending_f32(left: f32, right: f32) -> Ordering {
    // Inputs are finite. Equal f32 probabilities, including sigmoid saturation,
    // remain ties rather than falling back to raw logits.
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

// Pinned PyTorch 2.8 CPU evaluates contiguous f32 sigmoid in eight-value
// Vectorized blocks. Its AArch64 build uses SLEEF expf u10 (submodule commit
// 5a1d179df9cf652951b59010a2d2075372d67f68, Boost Software License 1.0); a
// scalar platform exp differs by
// an ulp for some live proposal logits and can therefore change stable ranks.
// Keep this local to relation proposals: entity/record confidence behavior is a
// separate accepted contract.
fn sleef_expf_u10(value: f32) -> f32 {
    if value < -104.0 {
        return 0.0;
    }
    if value > 100.0 {
        return f32::INFINITY;
    }

    const R_LN2: f32 = f32::from_bits(0x3fb8_aa3b);
    const L2_UPPER: f32 = f32::from_bits(0x3f31_7200);
    const L2_LOWER: f32 = f32::from_bits(0x35bf_be8e);

    let exponent = (value * R_LN2).round_ties_even() as i32;
    let exponent_f32 = exponent as f32;
    let mut reduced = exponent_f32.mul_add(-L2_UPPER, value);
    reduced = exponent_f32.mul_add(-L2_LOWER, reduced);

    let mut polynomial = 0.000_198_527_62_f32;
    polynomial = polynomial.mul_add(reduced, 0.001_393_043_6);
    polynomial = polynomial.mul_add(reduced, 0.008_333_361);
    polynomial = polynomial.mul_add(reduced, 0.041_666_485);
    polynomial = polynomial.mul_add(reduced, 0.166_666_67);
    polynomial = polynomial.mul_add(reduced, 0.5);
    polynomial = (reduced * reduced).mul_add(polynomial, reduced);
    polynomial += 1.0;

    // SLEEF's vldexp2 splits the exponent so both powers remain normal.
    let upper_exponent = exponent >> 1;
    let lower_exponent = exponent - upper_exponent;
    let power_of_two = |power: i32| {
        f32::from_bits(u32::try_from(power + 0x7f).expect("split exponent is nonnegative") << 23)
    };
    (polynomial * power_of_two(upper_exponent)) * power_of_two(lower_exponent)
}

fn scalar_source_sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

fn vector_source_sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + sleef_expf_u10(-logit))
}

fn proposal_probabilities(pair_logits: ArrayView2<'_, f32>) -> Vec<f32> {
    // TensorIterator preserves dense row/column layouts and traverses their
    // physical order. Gapped layouts start a fresh loop on the contiguous axis;
    // broadcast axes follow the standard output layout. cpu_kernel_vec handles
    // two four-lane vectors per iteration and sends each loop tail to scalar exp.
    const PYTORCH_VECTOR_BLOCK: usize = 8;
    let (query_count, candidate_count) = pair_logits.dim();
    let source_count = pair_logits.len();
    let query_stride = pair_logits.strides()[0];
    let candidate_stride = pair_logits.strides()[1];

    enum Traversal {
        GlobalRowMajor,
        GlobalColumnMajor,
        Rows,
        Columns,
        Scalar,
    }

    let traversal =
        if pair_logits.is_standard_layout() || (query_stride == 0 && candidate_stride == 0) {
            Traversal::GlobalRowMajor
        } else if query_stride == 1 && candidate_stride == query_count as isize {
            Traversal::GlobalColumnMajor
        } else if matches!(candidate_stride, 0 | 1) {
            Traversal::Rows
        } else if query_stride == 1 {
            Traversal::Columns
        } else {
            Traversal::Scalar
        };

    let global_vectorized = source_count / PYTORCH_VECTOR_BLOCK * PYTORCH_VECTOR_BLOCK;
    let row_vectorized = candidate_count / PYTORCH_VECTOR_BLOCK * PYTORCH_VECTOR_BLOCK;
    let column_vectorized = query_count / PYTORCH_VECTOR_BLOCK * PYTORCH_VECTOR_BLOCK;
    let mut probabilities = Vec::with_capacity(source_count);
    for query in 0..query_count {
        for candidate in 0..candidate_count {
            let uses_vector_kernel = match traversal {
                Traversal::GlobalRowMajor => {
                    query * candidate_count + candidate < global_vectorized
                }
                Traversal::GlobalColumnMajor => candidate * query_count + query < global_vectorized,
                Traversal::Rows => candidate < row_vectorized,
                Traversal::Columns => query < column_vectorized,
                Traversal::Scalar => false,
            };
            let logit = pair_logits[(query, candidate)];
            probabilities.push(if uses_vector_kernel {
                vector_source_sigmoid(logit)
            } else {
                scalar_source_sigmoid(logit)
            });
        }
    }
    probabilities
}

fn validate_config(config: RelationProposalConfig) -> Result<()> {
    ensure!(
        config.heads_per_relation > 0,
        "heads_per_relation must be positive"
    );
    ensure!(
        config.tails_per_relation > 0,
        "tails_per_relation must be positive"
    );
    ensure!(config.pair_cap > 0, "pair_cap must be positive");
    ensure!(
        config.argument_threshold.is_finite() && (0.0..=1.0).contains(&config.argument_threshold),
        "argument_threshold must be a finite probability in [0,1], got {}",
        config.argument_threshold
    );
    config
        .heads_per_relation
        .checked_mul(config.tails_per_relation)
        .ok_or_else(|| anyhow::anyhow!("relation endpoint Cartesian product overflows usize"))?;
    Ok(())
}

fn validate_inputs(
    indices: ArrayView3<'_, i64>,
    valid_mask: ArrayView2<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
    pair_logits: ArrayView2<'_, f32>,
    relation_count: usize,
    config: RelationProposalConfig,
) -> Result<(usize, usize)> {
    let (queries, candidates, coordinates) = indices.dim();
    ensure!(
        coordinates == 2,
        "indices must have shape [Q,C,2], got {:?}",
        indices.dim()
    );
    ensure!(
        valid_mask.dim() == (queries, candidates),
        "valid_mask must have shape [Q,C]=[{queries},{candidates}], got {:?}",
        valid_mask.dim()
    );
    ensure!(
        query_mask.len() == queries,
        "query_mask must have shape [Q]=[{queries}], got [{}]",
        query_mask.len()
    );
    ensure!(
        pair_logits.dim() == (queries, candidates),
        "pair_logits must have shape [Q,C]=[{queries},{candidates}], got {:?}",
        pair_logits.dim()
    );
    ensure!(
        queries > 0,
        "relation pair generation requires at least one query"
    );
    ensure!(
        candidates > 0,
        "relation pair generation requires at least one candidate slot"
    );
    ensure!(
        pair_logits.iter().all(|value| value.is_finite()),
        "pair_logits contains a non-finite value"
    );
    ensure!(
        pair_logits.strides().iter().all(|&stride| stride >= 0),
        "negative-stride pair_logits are unsupported because pinned PyTorch has no equivalent tensor layout"
    );

    let source_count = queries
        .checked_mul(candidates)
        .ok_or_else(|| anyhow::anyhow!("flattened relation candidate count overflows usize"))?;
    relation_count
        .checked_mul(config.pair_cap)
        .ok_or_else(|| anyhow::anyhow!("maximum compact relation output size overflows usize"))?;

    // Only rows which can participate are required to contain legal spans.
    // Padding and query-masked rows may retain arbitrary sentinel coordinates.
    for query in 0..queries {
        if !query_mask[query] {
            continue;
        }
        for candidate in 0..candidates {
            if !valid_mask[(query, candidate)] {
                continue;
            }
            let start = indices[(query, candidate, 0)];
            let end = indices[(query, candidate, 1)];
            ensure!(
                start >= 0 && end > start,
                "query {query} candidate {candidate} has malformed live half-open span [{start},{end})"
            );
            usize::try_from(start).with_context(|| {
                format!("query {query} candidate {candidate} start does not fit usize")
            })?;
            usize::try_from(end).with_context(|| {
                format!("query {query} candidate {candidate} end does not fit usize")
            })?;
        }
    }

    Ok((queries, source_count))
}

fn membership(query_ids: &[i64], query_count: usize) -> Vec<bool> {
    let mut members = vec![false; query_count];
    for &query_id in query_ids {
        if let Ok(query_id) = usize::try_from(query_id)
            && query_id < query_count
        {
            members[query_id] = true;
        }
    }
    members
}

fn select_endpoints(
    indices: ArrayView3<'_, i64>,
    valid_mask: ArrayView2<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
    probabilities: &[f32],
    members: &[bool],
    requested: usize,
    argument_threshold: f32,
) -> Vec<Endpoint> {
    let (_, candidates, _) = indices.dim();
    let mut selected = Vec::new();
    for query in 0..members.len() {
        if !members[query] || !query_mask[query] {
            continue;
        }
        for candidate in 0..candidates {
            if !valid_mask[(query, candidate)] {
                continue;
            }
            let flat_index = query * candidates + candidate;
            let probability = probabilities[flat_index];
            if probability < argument_threshold {
                continue;
            }
            selected.push(Endpoint {
                query_id: query,
                span: [
                    usize::try_from(indices[(query, candidate, 0)])
                        .expect("live spans were validated"),
                    usize::try_from(indices[(query, candidate, 1)])
                        .expect("live spans were validated"),
                ],
                probability,
                flat_index,
            });
        }
    }
    selected.sort_by(|left, right| {
        descending_f32(left.probability, right.probability)
            .then_with(|| left.span[0].cmp(&right.span[0]))
            .then_with(|| left.span[1].cmp(&right.span[1]))
            .then_with(|| left.flat_index.cmp(&right.flat_index))
    });
    selected.truncate(requested);
    selected
}

/// Generate compact typed relation pairs from one batch item's candidate rows.
///
/// Inputs use query-major shapes: indices `[Q,C,2]`, valid/logits `[Q,C]`, and
/// query mask `[Q]`. Raw pair logits are converted with an untempered sigmoid
/// matching the pinned macOS arm64 PyTorch 2.8 reference operation order.
/// Positive-stride and broadcast logit views are supported; negative-stride
/// logits are rejected because that reference cannot represent them.
/// Invalid query IDs in a relation spec are ignored.
pub fn generate_relation_pairs(
    indices: ArrayView3<'_, i64>,
    valid_mask: ArrayView2<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
    pair_logits: ArrayView2<'_, f32>,
    specs: &[RelationTypeSpec],
    config: RelationProposalConfig,
) -> Result<Vec<RelationPair>> {
    validate_config(config)?;
    if specs.is_empty() {
        return Ok(Vec::new());
    }

    let (query_count, source_count) = validate_inputs(
        indices,
        valid_mask,
        query_mask,
        pair_logits,
        specs.len(),
        config,
    )?;
    let probabilities = proposal_probabilities(pair_logits);
    debug_assert_eq!(probabilities.len(), source_count);
    debug_assert_eq!(query_count, pair_logits.nrows());

    // Do not reserve `relation_count * pair_cap`: valid settings may use a
    // very large cap while the actual compact candidate product stays small.
    let mut output = Vec::new();

    for (relation_index, spec) in specs.iter().enumerate() {
        let head_members = membership(&spec.head_query_ids, query_count);
        let tail_members = membership(&spec.tail_query_ids, query_count);
        let heads = select_endpoints(
            indices,
            valid_mask,
            query_mask,
            &probabilities,
            &head_members,
            config.heads_per_relation,
            config.argument_threshold,
        );
        let tails = select_endpoints(
            indices,
            valid_mask,
            query_mask,
            &probabilities,
            &tail_members,
            config.tails_per_relation,
            config.argument_threshold,
        );

        let pair_count = heads.len().checked_mul(tails.len()).ok_or_else(|| {
            anyhow::anyhow!("relation {relation_index} pair count overflows usize")
        })?;
        let mut proposals = Vec::with_capacity(pair_count);
        for (head_index, &head) in heads.iter().enumerate() {
            for (tail_index, &tail) in tails.iter().enumerate() {
                if !spec.allow_self && head.span == tail.span {
                    continue;
                }
                proposals.push(PairProposal {
                    head,
                    tail,
                    score: head.probability * tail.probability,
                    cartesian_index: head_index * config.tails_per_relation + tail_index,
                });
            }
        }
        proposals.sort_by(|left, right| {
            descending_f32(left.score, right.score)
                .then_with(|| left.cartesian_index.cmp(&right.cartesian_index))
        });
        proposals.truncate(config.pair_cap);

        output.extend(proposals.into_iter().map(|proposal| RelationPair {
            relation_index,
            head_query_id: proposal.head.query_id,
            tail_query_id: proposal.tail.query_id,
            head: proposal.head.span,
            tail: proposal.tail.span,
            head_probability: proposal.head.probability,
            tail_probability: proposal.tail.probability,
        }));
    }

    Ok(output)
}
