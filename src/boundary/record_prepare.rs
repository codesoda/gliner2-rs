//! Source-faithful ragged preparation for the boundary record head.
//!
//! The ONNX graph has a padded, positive-dimension ABI. This module keeps the
//! query routing, candidate compaction, instance metadata, and output trimming
//! outside that graph so decoding still observes `RecordHead.forward_group`'s
//! ragged field order.

use anyhow::{Context, Result, ensure};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, s};

use super::{
    pool::CandidatePool,
    record_decode::{FieldSpec, RecordGroup, RecordMode},
    record_schema::CompiledRecordSpec,
    records::{RecordInput, RecordOutput},
    scorer::ScorerOutput,
};

const ANCHORLESS_INSTANCES: usize = 32;

/// A positive-dimension graph input plus the ragged metadata needed after ONNX.
#[derive(Debug)]
pub(crate) struct PreparedRecordGroup {
    pub(crate) input: RecordInput,
    pub(crate) mode: RecordMode,
    pub(crate) field_specs: Vec<FieldSpec>,
    pub(crate) field_query_ids: Vec<usize>,
    pub(crate) natural_anchor_query_id: Option<usize>,
    pub(crate) field_spans: Vec<Vec<[usize; 2]>>,
    pub(crate) instance_seed: Vec<Option<(usize, usize)>>,
    pub(crate) instance_spans: Vec<Option<[usize; 2]>>,
}

impl PreparedRecordGroup {
    /// Validate graph output and restore the source ragged assignment widths.
    pub(crate) fn into_decode_group(self, output: RecordOutput) -> Result<RecordGroup> {
        self.input
            .validate()
            .context("prepared record input became invalid")?;

        let fields = self.field_specs.len();
        let hidden = self.input.field_query_states.ncols();
        let padded_candidates = self.input.field_candidate_states.shape()[1];
        let instances = match self.mode {
            RecordMode::Anchorless => ANCHORLESS_INSTANCES,
            RecordMode::Natural | RecordMode::Latent => self.instance_seed.len(),
        };
        let assignment_width = padded_candidates
            .checked_add(1)
            .context("record assignment width overflow")?;

        ensure!(
            output.instance_states.dim() == (instances, hidden),
            "record instance_states shape {:?}, expected [{instances}, {hidden}]",
            output.instance_states.dim()
        );
        ensure!(
            output.object_logits.len() == instances,
            "record object_logits shape {:?}, expected [{instances}]",
            output.object_logits.shape()
        );
        ensure!(
            output.assignment_logits.dim() == (fields, instances, assignment_width),
            "record assignment_logits shape {:?}, expected [{fields}, {instances}, {assignment_width}]",
            output.assignment_logits.dim()
        );
        ensure!(
            output.instance_states.iter().all(|value| value.is_finite()),
            "record instance_states contains non-finite values"
        );
        ensure!(
            output.object_logits.iter().all(|value| value.is_finite()),
            "record object_logits contains non-finite values"
        );
        ensure!(
            output
                .assignment_logits
                .iter()
                .all(|value| value.is_finite()),
            "record assignment_logits contains non-finite values"
        );
        ensure!(
            self.field_query_ids.len() == fields
                && self.field_spans.len() == fields
                && self.instance_spans.len() == instances
                && self.instance_seed.len() == instances,
            "prepared record metadata dimensions are inconsistent"
        );

        let mut assign_logits = Vec::with_capacity(fields);
        for (field, spans) in self.field_spans.iter().enumerate() {
            let ragged_width = spans
                .len()
                .checked_add(1)
                .context("ragged record assignment width overflow")?;
            ensure!(
                ragged_width <= assignment_width,
                "field {field} has {} candidates but padded graph width is {padded_candidates}",
                spans.len()
            );
            // Column zero is ABSENT. Padded candidate columns are deliberately
            // excluded so they can never enter a decoder softmax.
            assign_logits.push(
                output
                    .assignment_logits
                    .slice(s![field, .., 0..ragged_width])
                    .to_owned(),
            );
        }

        Ok(RecordGroup {
            mode: self.mode,
            field_specs: self.field_specs,
            field_query_ids: self.field_query_ids,
            natural_anchor_query_id: self.natural_anchor_query_id,
            object_logits: output.object_logits.iter().copied().collect(),
            assign_logits,
            field_spans: self.field_spans,
            instance_seed: self.instance_seed,
            instance_spans: self.instance_spans,
        })
    }
}

/// Prepare one compiled record group from the shared document candidate pool.
///
/// Candidate membership is exactly `pool.mask & query_mask[qid]`. Candidates
/// are never probability-filtered and remain in original pool order. The
/// natural and latent empty-instance cases return `None` instead of presenting
/// a zero seed dimension to ORT.
pub(crate) fn prepare_record_group(
    spec: &CompiledRecordSpec,
    query_states: ArrayView2<'_, f32>,
    query_mask: ArrayView1<'_, bool>,
    pool: &CandidatePool,
    scores: &ScorerOutput,
) -> Result<Option<PreparedRecordGroup>> {
    let query_count = query_states.nrows();
    let hidden = query_states.ncols();
    let candidate_count = pool.indices.nrows();

    validate_inputs(
        query_states,
        query_mask,
        pool,
        scores,
        query_count,
        hidden,
        candidate_count,
    )?;

    if spec.fields.is_empty() {
        return Ok(None);
    }
    ensure!(hidden > 0, "record hidden width H must be positive");

    for (field, compiled) in spec.fields.iter().enumerate() {
        ensure!(
            compiled.query_id < query_count,
            "record field {field} ({:?}) query id {} is outside Q={query_count}",
            compiled.name,
            compiled.query_id
        );
    }
    let natural_anchor_field = match spec.mode {
        RecordMode::Natural => {
            let anchor_query_id = spec
                .anchor_query_id
                .context("natural record requires an anchor query id")?;
            ensure!(
                anchor_query_id < query_count,
                "natural anchor query id {anchor_query_id} is outside Q={query_count}"
            );
            Some(
                spec.fields
                    .iter()
                    .position(|field| field.query_id == anchor_query_id)
                    .context("natural record anchor query id is not a field")?,
            )
        }
        RecordMode::Latent | RecordMode::Anchorless => {
            ensure!(
                spec.anchor_query_id.is_none(),
                "non-natural record mode must not declare an anchor query id"
            );
            None
        }
    };

    let live_pool_indices = validate_live_spans(pool)?;
    let mut field_pool_indices = Vec::with_capacity(spec.fields.len());
    let mut field_spans = Vec::with_capacity(spec.fields.len());
    let mut max_field_candidates = 0_usize;
    let mut context_count = 0_usize;

    for compiled in &spec.fields {
        let indices = if query_mask[compiled.query_id] {
            live_pool_indices.clone()
        } else {
            Vec::new()
        };
        let mut spans = Vec::with_capacity(indices.len());
        for &candidate in &indices {
            spans.push(checked_span(pool, candidate)?);
        }
        max_field_candidates = max_field_candidates.max(indices.len());
        context_count = context_count
            .checked_add(indices.len())
            .context("record context candidate count overflow")?;
        field_pool_indices.push(indices);
        field_spans.push(spans);
    }

    let field_count = spec.fields.len();
    let padded_candidates = max_field_candidates.max(1);
    checked_tensor_size(
        &[field_count, padded_candidates, hidden],
        "field candidates",
    )?;
    checked_tensor_size(&[context_count.max(1), hidden], "record context")?;

    let mut field_query_states = Array2::<f32>::zeros((field_count, hidden));
    let mut field_candidate_states = Array3::<f32>::zeros((field_count, padded_candidates, hidden));
    let mut field_candidate_mask =
        Array2::<bool>::from_elem((field_count, padded_candidates), false);

    for (field, compiled) in spec.fields.iter().enumerate() {
        field_query_states
            .row_mut(field)
            .assign(&query_states.row(compiled.query_id));
        for (local_candidate, &pool_candidate) in field_pool_indices[field].iter().enumerate() {
            field_candidate_states
                .slice_mut(s![field, local_candidate, ..])
                .assign(&scores.candidate_states.slice(s![0, pool_candidate, ..]));
            field_candidate_mask[(field, local_candidate)] = true;
        }
    }

    let (context_states, context_mask) = if context_count == 0 {
        (
            Array2::<f32>::zeros((1, hidden)),
            Array1::<bool>::from_elem(1, false),
        )
    } else {
        let mut states = Array2::<f32>::zeros((context_count, hidden));
        let mut context = 0;
        for indices in &field_pool_indices {
            for &pool_candidate in indices {
                states
                    .row_mut(context)
                    .assign(&scores.candidate_states.slice(s![0, pool_candidate, ..]));
                context += 1;
            }
        }
        (states, Array1::<bool>::from_elem(context_count, true))
    };

    let mut instance_seed = Vec::new();
    let mut instance_spans = Vec::new();
    let (seed_states, seed_object_logits, mode) = match spec.mode {
        RecordMode::Natural => {
            let anchor_query_id = spec
                .anchor_query_id
                .context("natural record anchor query id disappeared after route validation")?;
            let anchor_field = natural_anchor_field
                .context("natural record anchor field disappeared after route validation")?;
            let anchors = &field_pool_indices[anchor_field];
            if anchors.is_empty() {
                return Ok(None);
            }
            checked_tensor_size(&[anchors.len(), hidden], "natural record seeds")?;
            let mut states = Array2::<f32>::zeros((anchors.len(), hidden));
            let mut logits = Array1::<f32>::zeros(anchors.len());
            for (local_candidate, &pool_candidate) in anchors.iter().enumerate() {
                states
                    .row_mut(local_candidate)
                    .assign(&scores.candidate_states.slice(s![0, pool_candidate, ..]));
                logits[local_candidate] = scores.pair_logits[(0, anchor_query_id, pool_candidate)];
                instance_seed.push(Some((anchor_field, local_candidate)));
                instance_spans.push(Some(field_spans[anchor_field][local_candidate]));
            }
            (states, logits, 0)
        }
        RecordMode::Latent => {
            if context_count == 0 {
                return Ok(None);
            }
            // The anchorless context and latent seeds have the same field-major
            // sequence. Repeated pool candidates in different fields stay
            // repeated, matching the ragged source implementation.
            let states = context_states.clone();
            let logits = Array1::<f32>::zeros(context_count);
            for (field, spans) in field_spans.iter().enumerate() {
                for (candidate, &span) in spans.iter().enumerate() {
                    instance_seed.push(Some((field, candidate)));
                    instance_spans.push(Some(span));
                }
            }
            (states, logits, 1)
        }
        RecordMode::Anchorless => {
            instance_seed.resize(ANCHORLESS_INSTANCES, None);
            instance_spans.resize(ANCHORLESS_INSTANCES, None);
            (
                Array2::<f32>::zeros((1, hidden)),
                Array1::<f32>::zeros(1),
                2,
            )
        }
    };

    let field_specs = spec
        .fields
        .iter()
        .map(|field| FieldSpec {
            query_id: field.query_id,
            cardinality: field.cardinality,
            exclusive: field.exclusive,
        })
        .collect();
    let field_query_ids = spec.fields.iter().map(|field| field.query_id).collect();
    let natural_anchor_query_id = match spec.mode {
        RecordMode::Natural => spec.anchor_query_id,
        RecordMode::Latent | RecordMode::Anchorless => None,
    };

    let input = RecordInput {
        field_query_states,
        field_candidate_states,
        field_candidate_mask,
        context_states,
        context_mask,
        seed_states,
        seed_object_logits,
        mode,
    };
    input.validate()?;

    Ok(Some(PreparedRecordGroup {
        input,
        mode: spec.mode,
        field_specs,
        field_query_ids,
        natural_anchor_query_id,
        field_spans,
        instance_seed,
        instance_spans,
    }))
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    query_states: ArrayView2<'_, f32>,
    query_mask: ArrayView1<'_, bool>,
    pool: &CandidatePool,
    scores: &ScorerOutput,
    query_count: usize,
    hidden: usize,
    candidate_count: usize,
) -> Result<()> {
    ensure!(
        query_mask.len() == query_count,
        "query_mask shape {:?}, expected [{query_count}]",
        query_mask.shape()
    );
    ensure!(
        pool.indices.ncols() == 2,
        "pool indices shape {:?}, expected [C, 2]",
        pool.indices.dim()
    );
    ensure!(
        pool.mask.len() == candidate_count,
        "pool mask shape {:?}, expected [{candidate_count}]",
        pool.mask.shape()
    );
    ensure!(
        pool.compat_logits.len() == candidate_count,
        "pool compat_logits shape {:?}, expected [{candidate_count}]",
        pool.compat_logits.shape()
    );
    ensure!(
        pool.proposal_logits.len() == candidate_count,
        "pool proposal_logits shape {:?}, expected [{candidate_count}]",
        pool.proposal_logits.shape()
    );
    ensure!(
        scores.pair_logits.dim() == (1, query_count, candidate_count),
        "record pair_logits shape {:?}, expected [1, {query_count}, {candidate_count}]",
        scores.pair_logits.dim()
    );
    ensure!(
        scores.candidate_states.dim() == (1, candidate_count, hidden),
        "record candidate_states shape {:?}, expected [1, {candidate_count}, {hidden}]",
        scores.candidate_states.dim()
    );
    ensure!(
        scores.null_logits.dim() == (1, query_count),
        "record null_logits shape {:?}, expected [1, {query_count}]",
        scores.null_logits.dim()
    );
    ensure!(
        scores.count_log_rates.dim() == (1, query_count),
        "record count_log_rates shape {:?}, expected [1, {query_count}]",
        scores.count_log_rates.dim()
    );

    for (name, finite) in [
        (
            "query_states",
            query_states.iter().all(|value| value.is_finite()),
        ),
        (
            "pool compat_logits",
            pool.compat_logits.iter().all(|value| value.is_finite()),
        ),
        (
            "pool proposal_logits",
            pool.proposal_logits.iter().all(|value| value.is_finite()),
        ),
        (
            "record pair_logits",
            scores.pair_logits.iter().all(|value| value.is_finite()),
        ),
        (
            "record candidate_states",
            scores
                .candidate_states
                .iter()
                .all(|value| value.is_finite()),
        ),
        (
            "record null_logits",
            scores.null_logits.iter().all(|value| value.is_finite()),
        ),
        (
            "record count_log_rates",
            scores.count_log_rates.iter().all(|value| value.is_finite()),
        ),
    ] {
        ensure!(finite, "{name} contains non-finite values");
    }
    Ok(())
}

fn validate_live_spans(pool: &CandidatePool) -> Result<Vec<usize>> {
    let mut live = Vec::new();
    for candidate in 0..pool.mask.len() {
        if !pool.mask[candidate] {
            continue;
        }
        checked_span(pool, candidate)?;
        live.push(candidate);
    }
    Ok(live)
}

fn checked_span(pool: &CandidatePool, candidate: usize) -> Result<[usize; 2]> {
    let start = pool.indices[(candidate, 0)];
    let end = pool.indices[(candidate, 1)];
    ensure!(
        start >= 0 && start < end,
        "live pool candidate {candidate} has invalid half-open span [{start}, {end})"
    );
    Ok([
        usize::try_from(start).context("live pool span start exceeds usize")?,
        usize::try_from(end).context("live pool span end exceeds usize")?,
    ])
}

fn checked_tensor_size(dimensions: &[usize], name: &str) -> Result<()> {
    dimensions.iter().try_fold(1_usize, |size, &dimension| {
        size.checked_mul(dimension)
            .with_context(|| format!("{name} tensor dimensions overflow usize"))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3, arr1, arr2};

    use super::*;
    use crate::boundary::{
        pool::CandidatePool,
        record_decode::Cardinality,
        record_schema::{CompiledRecordField, CompiledRecordSpec},
        scorer::ScorerOutput,
    };

    fn field(query_id: usize, role_index: usize) -> CompiledRecordField {
        CompiledRecordField {
            name: format!("field-{role_index}"),
            role_index,
            query_id,
            cardinality: Cardinality::OptionalOne,
            exclusive: false,
        }
    }

    fn spec(mode: RecordMode, query_ids: &[usize], anchor: Option<usize>) -> CompiledRecordSpec {
        CompiledRecordSpec {
            structure_index: 0,
            name: "record".into(),
            mode,
            fields: query_ids
                .iter()
                .enumerate()
                .map(|(role, &query)| field(query, role))
                .collect(),
            anchor_query_id: anchor,
        }
    }

    fn inputs(
        queries: usize,
        hidden: usize,
        spans: &[[i64; 2]],
        pool_mask: &[bool],
        query_mask: &[bool],
    ) -> (Array2<f32>, Array1<bool>, CandidatePool, ScorerOutput) {
        let candidates = spans.len();
        let query_states =
            Array2::from_shape_fn((queries, hidden), |(q, h)| (100 * q + h) as f32 + 0.25);
        let query_mask = Array1::from_vec(query_mask.to_vec());
        let indices = Array2::from_shape_fn((candidates, 2), |(candidate, endpoint)| {
            spans[candidate][endpoint]
        });
        let pool = CandidatePool {
            indices,
            mask: Array1::from_vec(pool_mask.to_vec()),
            compat_logits: Array1::zeros(candidates),
            proposal_logits: Array1::zeros(candidates),
        };
        let pair_logits = Array3::from_shape_fn((1, queries, candidates), |(_, q, c)| {
            (1_000 * q + c) as f32 + 0.5
        });
        let candidate_states = Array3::from_shape_fn((1, candidates, hidden), |(_, c, h)| {
            (100 * c + h) as f32 + 0.75
        });
        let scores = ScorerOutput {
            pair_logits,
            candidate_states,
            null_logits: Array2::zeros((1, queries)),
            count_log_rates: Array2::zeros((1, queries)),
        };
        (query_states, query_mask, pool, scores)
    }

    #[test]
    fn noncontiguous_natural_routing_keeps_every_anchor_and_exact_logits() -> Result<()> {
        let spans: Vec<_> = (0..40).map(|index| [index, index + 1]).collect();
        let (queries, query_mask, pool, scores) =
            inputs(5, 3, &spans, &[true; 40], &[true, true, false, true, true]);
        let prepared = prepare_record_group(
            &spec(RecordMode::Natural, &[3, 1], Some(1)),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();

        assert_eq!(prepared.field_query_ids, vec![3, 1]);
        assert_eq!(prepared.input.field_query_states.row(0), queries.row(3));
        assert_eq!(prepared.input.field_query_states.row(1), queries.row(1));
        assert_eq!(
            prepared.instance_seed.len(),
            40,
            "natural seeds are uncapped"
        );
        assert_eq!(prepared.instance_seed[39], Some((1, 39)));
        assert_eq!(prepared.instance_spans[39], Some([39, 40]));
        assert_eq!(
            prepared.input.seed_object_logits,
            scores.pair_logits.slice(s![0, 1, ..])
        );
        Ok(())
    }

    #[test]
    fn latent_seeds_are_field_major_and_preserve_duplicates() -> Result<()> {
        let (queries, query_mask, pool, scores) = inputs(
            4,
            2,
            &[[2, 4], [7, 8], [90, -4]],
            &[true, true, false],
            &[true, false, true, true],
        );
        let prepared = prepare_record_group(
            &spec(RecordMode::Latent, &[2, 0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();

        assert_eq!(prepared.field_spans, vec![vec![[2, 4], [7, 8]]; 2]);
        assert_eq!(
            prepared.instance_seed,
            vec![Some((0, 0)), Some((0, 1)), Some((1, 0)), Some((1, 1))]
        );
        assert_eq!(
            prepared.instance_spans,
            vec![Some([2, 4]), Some([7, 8]), Some([2, 4]), Some([7, 8])]
        );
        assert_eq!(
            prepared.input.seed_states.row(0),
            scores.candidate_states.slice(s![0, 0, ..])
        );
        assert_eq!(
            prepared.input.seed_states.row(2),
            scores.candidate_states.slice(s![0, 0, ..])
        );
        assert!(
            prepared
                .input
                .seed_object_logits
                .iter()
                .all(|&value| value == 0.0)
        );
        Ok(())
    }

    #[test]
    fn empty_natural_latent_and_fieldless_groups_bypass() -> Result<()> {
        let (queries, query_mask, pool, scores) = inputs(2, 2, &[[0, 1]], &[true], &[false, true]);
        for (mode, anchor) in [(RecordMode::Natural, Some(0)), (RecordMode::Latent, None)] {
            assert!(
                prepare_record_group(
                    &spec(mode, &[0], anchor),
                    queries.view(),
                    query_mask.view(),
                    &pool,
                    &scores,
                )?
                .is_none()
            );
        }
        assert!(
            prepare_record_group(
                &spec(RecordMode::Anchorless, &[], None),
                queries.view(),
                query_mask.view(),
                &pool,
                &scores,
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn anchorless_context_is_field_major_with_duplicates() -> Result<()> {
        let (queries, query_mask, pool, scores) =
            inputs(3, 2, &[[0, 1], [4, 6]], &[true, true], &[true, false, true]);
        let prepared = prepare_record_group(
            &spec(RecordMode::Anchorless, &[2, 0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();

        assert_eq!(prepared.input.context_states.dim(), (4, 2));
        assert_eq!(
            prepared.input.context_states.row(0),
            prepared.input.context_states.row(2)
        );
        assert_eq!(
            prepared.input.context_states.row(1),
            prepared.input.context_states.row(3)
        );
        assert!(prepared.input.context_mask.iter().all(|&value| value));
        assert_eq!(prepared.instance_seed, vec![None; ANCHORLESS_INSTANCES]);
        assert_eq!(prepared.instance_spans, vec![None; ANCHORLESS_INSTANCES]);
        assert_eq!(prepared.input.seed_states.dim(), (1, 2));
        Ok(())
    }

    #[test]
    fn anchorless_empty_context_uses_masked_zero_rows() -> Result<()> {
        let (queries, query_mask, pool, scores) = inputs(1, 3, &[[0, 1]], &[true], &[false]);
        let prepared = prepare_record_group(
            &spec(RecordMode::Anchorless, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();

        assert_eq!(prepared.input.field_candidate_states.dim(), (1, 1, 3));
        assert_eq!(prepared.input.field_candidate_mask, arr2(&[[false]]));
        assert_eq!(prepared.input.context_states, Array2::<f32>::zeros((1, 3)));
        assert_eq!(prepared.input.context_mask, arr1(&[false]));
        assert_eq!(prepared.input.seed_states, Array2::<f32>::zeros((1, 3)));
        assert_eq!(prepared.input.seed_object_logits, arr1(&[0.0]));
        prepared.input.validate()?;
        Ok(())
    }

    #[test]
    fn output_is_trimmed_to_each_ragged_width_including_absent() -> Result<()> {
        let (queries, query_mask, pool, scores) =
            inputs(2, 2, &[[0, 1], [3, 5]], &[true, true], &[true, false]);
        let prepared = prepare_record_group(
            &spec(RecordMode::Anchorless, &[0, 1], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();
        let assignments =
            Array3::from_shape_fn((2, ANCHORLESS_INSTANCES, 3), |(field, instance, column)| {
                (field * 10_000 + instance * 10 + column) as f32
            });
        let output = RecordOutput {
            instance_states: Array2::zeros((ANCHORLESS_INSTANCES, 2)),
            object_logits: Array1::zeros(ANCHORLESS_INSTANCES),
            assignment_logits: assignments.clone(),
        };
        let group = prepared.into_decode_group(output)?;

        assert_eq!(group.assign_logits[0].dim(), (ANCHORLESS_INSTANCES, 3));
        assert_eq!(group.assign_logits[1].dim(), (ANCHORLESS_INSTANCES, 1));
        assert_eq!(group.assign_logits[0], assignments.slice(s![0, .., 0..3]));
        assert_eq!(group.assign_logits[1], assignments.slice(s![1, .., 0..1]));
        Ok(())
    }

    #[test]
    fn malformed_shapes_routes_spans_and_floats_fail_closed() {
        let (queries, query_mask, mut pool, mut scores) =
            inputs(2, 2, &[[0, 1], [2, 3]], &[true, false], &[true, true]);

        let bad_route = prepare_record_group(
            &spec(RecordMode::Latent, &[2], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )
        .unwrap_err()
        .to_string();
        assert!(bad_route.contains("outside Q=2"), "{bad_route}");

        pool.indices[(0, 0)] = -1;
        let negative = prepare_record_group(
            &spec(RecordMode::Latent, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )
        .unwrap_err()
        .to_string();
        assert!(negative.contains("invalid half-open span"), "{negative}");

        pool.indices[(0, 0)] = 4;
        pool.indices[(0, 1)] = 4;
        let reversed = prepare_record_group(
            &spec(RecordMode::Latent, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )
        .unwrap_err()
        .to_string();
        assert!(reversed.contains("invalid half-open span"), "{reversed}");

        pool.indices[(0, 0)] = 0;
        pool.indices[(0, 1)] = 1;
        scores.pair_logits = Array3::zeros((1, 1, 2));
        let shape = prepare_record_group(
            &spec(RecordMode::Latent, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )
        .unwrap_err()
        .to_string();
        assert!(shape.contains("pair_logits shape"), "{shape}");

        scores.pair_logits = Array3::zeros((1, 2, 2));
        scores.candidate_states[(0, 1, 0)] = f32::NAN;
        let finite = prepare_record_group(
            &spec(RecordMode::Latent, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )
        .unwrap_err()
        .to_string();
        assert!(finite.contains("non-finite"), "{finite}");
    }

    #[test]
    fn masked_span_garbage_is_never_read() -> Result<()> {
        let (queries, query_mask, pool, scores) =
            inputs(1, 1, &[[0, 1], [-99, -100]], &[true, false], &[true]);
        let prepared = prepare_record_group(
            &spec(RecordMode::Latent, &[0], None),
            queries.view(),
            query_mask.view(),
            &pool,
            &scores,
        )?
        .unwrap();
        assert_eq!(prepared.field_spans, vec![vec![[0, 1]]]);
        Ok(())
    }

    #[test]
    fn malformed_output_shapes_and_nonfinite_values_are_rejected() -> Result<()> {
        let make_prepared = || -> Result<PreparedRecordGroup> {
            let (queries, query_mask, pool, scores) = inputs(1, 2, &[[0, 1]], &[true], &[true]);
            Ok(prepare_record_group(
                &spec(RecordMode::Natural, &[0], Some(0)),
                queries.view(),
                query_mask.view(),
                &pool,
                &scores,
            )?
            .unwrap())
        };

        let shape = make_prepared()?
            .into_decode_group(RecordOutput {
                instance_states: Array2::zeros((1, 2)),
                object_logits: Array1::zeros(1),
                assignment_logits: Array3::zeros((1, 1, 1)),
            })
            .unwrap_err()
            .to_string();
        assert!(shape.contains("assignment_logits shape"), "{shape}");

        let nonfinite = make_prepared()?
            .into_decode_group(RecordOutput {
                instance_states: Array2::zeros((1, 2)),
                object_logits: arr1(&[f32::INFINITY]),
                assignment_logits: Array3::zeros((1, 1, 2)),
            })
            .unwrap_err()
            .to_string();
        assert!(nonfinite.contains("non-finite"), "{nonfinite}");
        Ok(())
    }
}
