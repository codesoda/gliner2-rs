//! Pure record-group decoding for the boundary architecture.
//!
//! This module consumes one ragged record-head group and mirrors the pinned
//! GLiNER2 `decode_group` implementation. Formatting, validators, choice
//! association, and candidate/assignment score combination happen later.

use std::{cmp::Ordering, collections::BTreeMap};

use anyhow::{Context, Result, ensure};
use ndarray::Array2;

use super::assignment::linear_sum_assignment;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordMode {
    Natural,
    Latent,
    Anchorless,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cardinality {
    OptionalOne,
    RequiredOne,
    ZeroOrMore,
    OneOrMore,
}

impl Cardinality {
    fn is_scalar(self) -> bool {
        matches!(self, Self::OptionalOne | Self::RequiredOne)
    }

    fn allows_absent(self) -> bool {
        matches!(self, Self::OptionalOne | Self::ZeroOrMore)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldSpec {
    pub query_id: usize,
    pub cardinality: Cardinality,
    pub exclusive: bool,
}

#[derive(Clone, Debug)]
pub struct RecordGroup {
    pub mode: RecordMode,
    pub field_specs: Vec<FieldSpec>,
    pub field_query_ids: Vec<usize>,
    pub natural_anchor_query_id: Option<usize>,
    pub object_logits: Vec<f32>,
    /// Per field, `[Ni, Cf + 1]`; column zero is ABSENT.
    pub assign_logits: Vec<Array2<f32>>,
    /// Per field, `Cf` half-open token spans.
    pub field_spans: Vec<Vec<[usize; 2]>>,
    pub instance_seed: Vec<Option<(usize, usize)>>,
    pub instance_spans: Vec<Option<[usize; 2]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedRecord {
    pub fields: BTreeMap<usize, Vec<[usize; 2]>>,
    pub field_scores: BTreeMap<usize, Vec<f32>>,
    pub anchor_span: Option<[usize; 2]>,
    pub score: f32,
}

type DedupKey = Vec<(usize, Vec<[usize; 2]>)>;

/// Decode one record group using the pinned upstream record semantics.
pub fn decode_group(
    group: &RecordGroup,
    anchor_threshold: f32,
    field_threshold: f32,
    object_threshold: f32,
    temperature: f32,
) -> Result<Vec<DecodedRecord>> {
    validate(
        group,
        anchor_threshold,
        field_threshold,
        object_threshold,
        temperature,
    )?;
    let instance_count = group.object_logits.len();
    if instance_count == 0 {
        return Ok(Vec::new());
    }

    let object_probabilities: Vec<_> = group
        .object_logits
        .iter()
        .map(|&logit| sigmoid(logit / temperature))
        .collect();
    let selection_threshold = if group.mode == RecordMode::Anchorless {
        object_threshold
    } else {
        anchor_threshold
    };
    let mut selected_instances: Vec<_> = (0..instance_count)
        .filter(|&instance| object_probabilities[instance] >= selection_threshold)
        .collect();
    selected_instances.sort_by(|&left, &right| {
        object_probabilities[right]
            .total_cmp(&object_probabilities[left])
            .then(left.cmp(&right))
    });

    // Indexed as [field][instance]. Upstream keys these by (instance, field).
    let mut scalar_choices = vec![vec![None; instance_count]; group.field_specs.len()];
    // Indexed as [field][candidate]. Values are (owning instance, probability).
    let mut list_owners: Vec<Vec<Option<(usize, f32)>>> = group
        .field_spans
        .iter()
        .map(|spans| vec![None; spans.len()])
        .collect();

    for (field_index, spec) in group.field_specs.iter().enumerate() {
        if !spec.exclusive || selected_instances.is_empty() {
            continue;
        }
        let candidate_count = group.field_spans[field_index].len();
        if spec.cardinality.is_scalar() {
            if candidate_count == 0 {
                continue;
            }
            let row_count = selected_instances.len();
            let probabilities: Vec<Vec<f32>> = selected_instances
                .iter()
                .map(|&instance| {
                    softmax(
                        group.assign_logits[field_index]
                            .row(instance)
                            .iter()
                            .map(|&value| value / temperature),
                    )
                })
                .collect();
            let epsilon = f32::EPSILON;
            let mut candidate_cost = vec![vec![0.0_f32; candidate_count]; row_count];
            let mut maximum_candidate_cost = f32::NEG_INFINITY;
            for row in 0..row_count {
                for candidate in 0..candidate_count {
                    let cost = -probabilities[row][candidate + 1].max(epsilon).ln();
                    candidate_cost[row][candidate] = cost;
                    maximum_candidate_cost = maximum_candidate_cost.max(cost);
                }
            }
            let mut diagonal: Vec<f32> = probabilities
                .iter()
                .map(|row| -row[0].max(epsilon).ln())
                .collect();
            if !spec.cardinality.allows_absent() {
                // This addition is intentionally f32 before cost promotion.
                diagonal.fill(maximum_candidate_cost + 50.0_f32);
            }
            let maximum_diagonal = diagonal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let invalid_cost = maximum_candidate_cost.max(maximum_diagonal) + 1_000.0_f32;
            let columns = candidate_count + row_count;
            let mut costs = Array2::from_elem((row_count, columns), f64::from(invalid_cost));
            for row in 0..row_count {
                for candidate in 0..candidate_count {
                    costs[(row, candidate)] = f64::from(candidate_cost[row][candidate]);
                }
                costs[(row, candidate_count + row)] = f64::from(diagonal[row]);
            }
            let (rows, columns) = linear_sum_assignment(costs.view())
                .context("exclusive scalar assignment failed")?;
            let mut assignments = vec![None; row_count];
            for (row, column) in rows.into_iter().zip(columns) {
                assignments[row] = Some(column);
            }
            for (row, &instance) in selected_instances.iter().enumerate() {
                let column = assignments[row].unwrap_or(candidate_count + row);
                if column >= candidate_count {
                    continue;
                }
                let probability = probabilities[row][column + 1];
                if probability < field_threshold && spec.cardinality.allows_absent() {
                    continue;
                }
                scalar_choices[field_index][instance] = Some((column, probability));
            }
        } else {
            if candidate_count == 0 {
                continue;
            }
            for (candidate, owner_slot) in list_owners[field_index].iter_mut().enumerate() {
                let mut strongest: Option<(usize, f32)> = None;
                for &instance in &selected_instances {
                    let probability = sigmoid(
                        group.assign_logits[field_index][(instance, candidate + 1)] / temperature,
                    );
                    // torch.max returns the first selected row on an exact tie.
                    if strongest.is_none_or(|(_, current)| probability > current) {
                        strongest = Some((instance, probability));
                    }
                }
                if let Some(owner) =
                    strongest.filter(|(_, probability)| *probability >= field_threshold)
                {
                    *owner_slot = Some(owner);
                }
            }
        }
    }

    let anchor_field_index = if group.mode == RecordMode::Natural {
        let anchor_query_id = group
            .natural_anchor_query_id
            .context("natural mode requires an anchor query id")?;
        Some(
            group
                .field_query_ids
                .iter()
                .position(|&query_id| query_id == anchor_query_id)
                .context("natural anchor query id is not a field")?,
        )
    } else {
        None
    };

    let mut records = Vec::new();
    for &instance in &selected_instances {
        let mut record = DecodedRecord {
            fields: BTreeMap::new(),
            field_scores: BTreeMap::new(),
            anchor_span: None,
            score: object_probabilities[instance],
        };
        if group.mode == RecordMode::Natural && group.instance_seed[instance].is_some() {
            record.anchor_span = group.instance_spans[instance];
        }

        for (field_index, spec) in group.field_specs.iter().enumerate() {
            let query_id = spec.query_id;
            if anchor_field_index == Some(field_index) {
                if let Some(span) = record.anchor_span {
                    record.fields.entry(query_id).or_default().push(span);
                    record
                        .field_scores
                        .entry(query_id)
                        .or_default()
                        .push(record.score);
                }
                continue;
            }

            if spec.cardinality.is_scalar() {
                let choice = if spec.exclusive {
                    scalar_choices[field_index][instance]
                } else {
                    let probabilities = softmax(
                        group.assign_logits[field_index]
                            .row(instance)
                            .iter()
                            .map(|&value| value / temperature),
                    );
                    let mut chosen = None;
                    for column in torch_unstable_argsort_desc(&probabilities) {
                        if column == 0 {
                            if spec.cardinality.allows_absent() {
                                chosen = Some(0);
                                break;
                            }
                            continue;
                        }
                        chosen = Some(column);
                        break;
                    }
                    match chosen {
                        None | Some(0) => None,
                        Some(column)
                            if probabilities[column] < field_threshold
                                && spec.cardinality.allows_absent() =>
                        {
                            None
                        }
                        Some(column) => Some((column - 1, probabilities[column])),
                    }
                };
                if let Some((candidate, probability)) = choice {
                    record
                        .fields
                        .entry(query_id)
                        .or_default()
                        .push(group.field_spans[field_index][candidate]);
                    record
                        .field_scores
                        .entry(query_id)
                        .or_default()
                        .push(probability);
                }
            } else {
                for (candidate, &span) in group.field_spans[field_index].iter().enumerate() {
                    let probability = if spec.exclusive {
                        match list_owners[field_index][candidate] {
                            Some((owner, probability)) if owner == instance => probability,
                            _ => continue,
                        }
                    } else {
                        let probability = sigmoid(
                            group.assign_logits[field_index][(instance, candidate + 1)]
                                / temperature,
                        );
                        if probability < field_threshold {
                            continue;
                        }
                        probability
                    };
                    record.fields.entry(query_id).or_default().push(span);
                    record
                        .field_scores
                        .entry(query_id)
                        .or_default()
                        .push(probability);
                }
            }
        }
        if !record.fields.is_empty() {
            records.push(record);
        }
    }

    match group.mode {
        RecordMode::Latent | RecordMode::Anchorless => {
            let mut unique: Vec<(DedupKey, DecodedRecord)> = Vec::new();
            for record in records {
                let key = dedup_key(&record);
                if let Some((_, best)) = unique.iter_mut().find(|(existing, _)| *existing == key) {
                    if record.score > best.score {
                        *best = record;
                    }
                } else {
                    unique.push((key, record));
                }
            }
            Ok(unique.into_iter().map(|(_, record)| record).collect())
        }
        RecordMode::Natural => {
            records.sort_by(|left, right| match (left.anchor_span, right.anchor_span) {
                (Some(left), Some(right)) => left.cmp(&right),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            });
            Ok(records)
        }
    }
}

pub fn derive_count(records: &[DecodedRecord]) -> usize {
    records.len()
}

fn validate(
    group: &RecordGroup,
    anchor_threshold: f32,
    field_threshold: f32,
    object_threshold: f32,
    temperature: f32,
) -> Result<()> {
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be finite and > 0"
    );
    for (name, threshold) in [
        ("anchor_threshold", anchor_threshold),
        ("field_threshold", field_threshold),
        ("object_threshold", object_threshold),
    ] {
        ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "{name} must be finite and in [0, 1]"
        );
    }
    ensure!(
        group.object_logits.iter().all(|value| value.is_finite()),
        "object_logits contains non-finite values"
    );
    for (instance, &logit) in group.object_logits.iter().enumerate() {
        let scaled = logit / temperature;
        ensure!(
            scaled.is_finite(),
            "object_logits[{instance}] / temperature is non-finite"
        );
        ensure!(
            sigmoid(scaled).is_finite(),
            "object probability for instance {instance} is non-finite"
        );
    }
    let field_count = group.field_specs.len();
    ensure!(
        group.field_query_ids.len() == field_count,
        "field_query_ids length does not match field_specs"
    );
    ensure!(
        group.assign_logits.len() == field_count,
        "assign_logits length does not match field_specs"
    );
    ensure!(
        group.field_spans.len() == field_count,
        "field_spans length does not match field_specs"
    );
    for field in 0..field_count {
        ensure!(
            group.field_query_ids[field] == group.field_specs[field].query_id,
            "field {field} query id does not match field_specs"
        );
        let expected = (
            group.object_logits.len(),
            group.field_spans[field].len() + 1,
        );
        ensure!(
            group.assign_logits[field].dim() == expected,
            "assign_logits[{field}] shape {:?}, expected {:?}",
            group.assign_logits[field].dim(),
            expected
        );
        ensure!(
            group.assign_logits[field]
                .iter()
                .all(|value| value.is_finite()),
            "assign_logits[{field}] contains non-finite values"
        );
        for ((instance, column), &logit) in group.assign_logits[field].indexed_iter() {
            let scaled = logit / temperature;
            ensure!(
                scaled.is_finite(),
                "assign_logits[{field}][{instance}, {column}] / temperature is non-finite"
            );
            if !group.field_specs[field].cardinality.is_scalar() {
                ensure!(
                    sigmoid(scaled).is_finite(),
                    "sigmoid activation for assign_logits[{field}][{instance}, {column}] is non-finite"
                );
            }
        }
        if group.field_specs[field].cardinality.is_scalar() {
            for instance in 0..group.object_logits.len() {
                let probabilities = softmax(
                    group.assign_logits[field]
                        .row(instance)
                        .iter()
                        .map(|&value| value / temperature),
                );
                ensure!(
                    probabilities.iter().all(|value| value.is_finite()),
                    "softmax activation for assign_logits[{field}] row {instance} contains non-finite values"
                );
            }
        }
    }
    ensure!(
        group.instance_seed.len() == group.object_logits.len(),
        "instance_seed length does not match object_logits"
    );
    ensure!(
        group.instance_spans.len() == group.object_logits.len(),
        "instance_spans length does not match object_logits"
    );
    if group.mode == RecordMode::Natural {
        let anchor = group
            .natural_anchor_query_id
            .context("natural mode requires an anchor query id")?;
        ensure!(
            group.field_query_ids.contains(&anchor),
            "natural anchor query id is not a field"
        );
    } else {
        ensure!(
            group.natural_anchor_query_id.is_none(),
            "non-natural mode must not declare an anchor query id"
        );
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax(values: impl IntoIterator<Item = f32>) -> Vec<f32> {
    let values: Vec<_> = values.into_iter().collect();
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut output: Vec<_> = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect();
    let sum: f32 = output.iter().sum();
    for value in &mut output {
        *value /= sum;
    }
    output
}

fn dedup_key(record: &DecodedRecord) -> DedupKey {
    record
        .fields
        .iter()
        .map(|(&query_id, spans)| {
            let mut spans = spans.clone();
            spans.sort_unstable();
            (query_id, spans)
        })
        .collect()
}

// `torch.argsort(..., descending=True)` defaults to unstable sorting. On the
// pinned CPU wheel it reaches libc++ `std::sort` over value/index proxy pairs.
// Modified Rust adaptation of LLVM libc++ algorithm code under Apache-2.0
// WITH LLVM-exception; see THIRD-PARTY-NOTICES.md and docs/licenses/LLVM-libcxx.txt.
// This focused port preserves that observable tie behavior rather than claiming
// stable first-index equivalence (notably, equal widths above 128 move the
// middle element to the front). Values are finite by validation.
pub(crate) fn torch_unstable_argsort_desc(values: &[f32]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    let depth = if indices.is_empty() {
        0
    } else {
        2 * (usize::BITS - 1 - indices.len().leading_zeros()) as usize
    };
    libcxx_introsort(&mut indices, values, 0, values.len(), depth, true);
    indices
}

fn before(left: usize, right: usize, values: &[f32]) -> bool {
    values[left] > values[right]
}

fn sort3(indices: &mut [usize], values: &[f32], x: usize, y: usize, z: usize) {
    if !before(indices[y], indices[x], values) {
        if !before(indices[z], indices[y], values) {
            return;
        }
        indices.swap(y, z);
        if before(indices[y], indices[x], values) {
            indices.swap(x, y);
        }
        return;
    }
    if before(indices[z], indices[y], values) {
        indices.swap(x, z);
        return;
    }
    indices.swap(x, y);
    if before(indices[z], indices[y], values) {
        indices.swap(y, z);
    }
}

fn sort4(indices: &mut [usize], values: &[f32], first: usize) {
    sort3(indices, values, first, first + 1, first + 2);
    if before(indices[first + 3], indices[first + 2], values) {
        indices.swap(first + 2, first + 3);
        if before(indices[first + 2], indices[first + 1], values) {
            indices.swap(first + 1, first + 2);
            if before(indices[first + 1], indices[first], values) {
                indices.swap(first, first + 1);
            }
        }
    }
}

fn sort5(indices: &mut [usize], values: &[f32], first: usize) {
    sort4(indices, values, first);
    if before(indices[first + 4], indices[first + 3], values) {
        indices.swap(first + 3, first + 4);
        if before(indices[first + 3], indices[first + 2], values) {
            indices.swap(first + 2, first + 3);
            if before(indices[first + 2], indices[first + 1], values) {
                indices.swap(first + 1, first + 2);
                if before(indices[first + 1], indices[first], values) {
                    indices.swap(first, first + 1);
                }
            }
        }
    }
}

fn insertion_sort(indices: &mut [usize], values: &[f32], first: usize, last: usize) {
    for index in first + 1..last {
        if before(indices[index], indices[index - 1], values) {
            let item = indices[index];
            let mut position = index;
            while position > first && before(item, indices[position - 1], values) {
                indices[position] = indices[position - 1];
                position -= 1;
            }
            indices[position] = item;
        }
    }
}

fn insertion_sort_incomplete(
    indices: &mut [usize],
    values: &[f32],
    first: usize,
    last: usize,
) -> bool {
    match last - first {
        0 | 1 => return true,
        2 => {
            if before(indices[first + 1], indices[first], values) {
                indices.swap(first, first + 1);
            }
            return true;
        }
        3 => {
            sort3(indices, values, first, first + 1, first + 2);
            return true;
        }
        4 => {
            sort4(indices, values, first);
            return true;
        }
        5 => {
            sort5(indices, values, first);
            return true;
        }
        _ => {}
    }
    sort3(indices, values, first, first + 1, first + 2);
    let mut changes = 0;
    let mut index = first + 3;
    while index < last {
        if before(indices[index], indices[index - 1], values) {
            let item = indices[index];
            let mut position = index;
            while position > first && before(item, indices[position - 1], values) {
                indices[position] = indices[position - 1];
                position -= 1;
            }
            indices[position] = item;
            changes += 1;
            if changes == 8 {
                return index + 1 == last;
            }
        }
        index += 1;
    }
    true
}

fn partition_equals_right(
    indices: &mut [usize],
    values: &[f32],
    begin: usize,
    end: usize,
) -> (usize, bool) {
    let pivot = indices[begin];
    let mut first = begin + 1;
    while before(indices[first], pivot, values) {
        first += 1;
    }
    let mut last = end;
    if begin == first - 1 {
        while first < last {
            last -= 1;
            if before(indices[last], pivot, values) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if before(indices[last], pivot, values) {
                break;
            }
        }
    }
    let already_partitioned = first >= last;
    while first < last {
        indices.swap(first, last);
        loop {
            first += 1;
            if !before(indices[first], pivot, values) {
                break;
            }
        }
        loop {
            last -= 1;
            if before(indices[last], pivot, values) {
                break;
            }
        }
    }
    let pivot_position = first - 1;
    if begin != pivot_position {
        indices[begin] = indices[pivot_position];
    }
    indices[pivot_position] = pivot;
    (pivot_position, already_partitioned)
}

fn partition_equals_left(indices: &mut [usize], values: &[f32], begin: usize, end: usize) -> usize {
    let pivot = indices[begin];
    let mut first = begin + 1;
    while first < end && !before(pivot, indices[first], values) {
        first += 1;
    }
    let mut last = end;
    if first < last {
        loop {
            last -= 1;
            if !before(pivot, indices[last], values) {
                break;
            }
        }
    }
    while first < last {
        indices.swap(first, last);
        loop {
            first += 1;
            if before(pivot, indices[first], values) {
                break;
            }
        }
        loop {
            last -= 1;
            if !before(pivot, indices[last], values) {
                break;
            }
        }
    }
    let pivot_position = first - 1;
    if begin != pivot_position {
        indices[begin] = indices[pivot_position];
    }
    indices[pivot_position] = pivot;
    first
}

// Introsort's depth-limit fallback is libc++'s
// `__partial_sort(first, last, last)`: make a heap, then sort that heap. The
// movement details below matter for equivalent elements and intentionally track
// libc++ rather than using Rust's unstable sort as an approximation.
//
// Derived from LLVM libc++ `partial_sort.h`, `make_heap.h`, `sort_heap.h`,
// `pop_heap.h`, `sift_down.h`, and `push_heap.h`, licensed under
// Apache-2.0 WITH LLVM-exception. Full license: docs/licenses/LLVM-libcxx.txt;
// source attribution and modification notice: THIRD-PARTY-NOTICES.md.
fn libcxx_partial_sort_all(indices: &mut [usize], values: &[f32], first: usize, last: usize) {
    let length = last - first;
    if length < 2 {
        return;
    }

    // libc++ __make_heap starts at the final parent and sifts toward the root.
    for root in (0..=((length - 2) / 2)).rev() {
        libcxx_sift_down(indices, values, first, root, length);
    }

    // libc++ __sort_heap repeatedly invokes its Floyd-style __pop_heap.
    for heap_length in (2..=length).rev() {
        libcxx_pop_heap(indices, values, first, heap_length);
    }
}

fn libcxx_sift_down(
    indices: &mut [usize],
    values: &[f32],
    first: usize,
    mut root: usize,
    length: usize,
) {
    if length < 2 || (length - 2) / 2 < root {
        return;
    }

    let mut child = 2 * root + 1;
    if child + 1 < length && before(indices[first + child], indices[first + child + 1], values) {
        child += 1;
    }
    if before(indices[first + child], indices[first + root], values) {
        return;
    }

    let top = indices[first + root];
    loop {
        indices[first + root] = indices[first + child];
        root = child;
        if (length - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        if child + 1 < length && before(indices[first + child], indices[first + child + 1], values)
        {
            child += 1;
        }
        if before(indices[first + child], top, values) {
            break;
        }
    }
    indices[first + root] = top;
}

fn libcxx_floyd_sift_down(
    indices: &mut [usize],
    values: &[f32],
    first: usize,
    length: usize,
) -> usize {
    debug_assert!(length >= 2);
    let mut hole = 0;
    let mut child = 0;
    loop {
        child = 2 * child + 1;
        if child + 1 < length && before(indices[first + child], indices[first + child + 1], values)
        {
            child += 1;
        }
        indices[first + hole] = indices[first + child];
        hole = child;
        if child > (length - 2) / 2 {
            return hole;
        }
    }
}

fn libcxx_sift_up(indices: &mut [usize], values: &[f32], first: usize, mut length: usize) {
    if length <= 1 {
        return;
    }
    let mut item = length - 1;
    length = (length - 2) / 2;
    let mut parent = length;
    if !before(indices[first + parent], indices[first + item], values) {
        return;
    }

    let value = indices[first + item];
    loop {
        indices[first + item] = indices[first + parent];
        item = parent;
        if length == 0 {
            break;
        }
        length = (length - 1) / 2;
        parent = length;
        if !before(indices[first + parent], value, values) {
            break;
        }
    }
    indices[first + item] = value;
}

fn libcxx_pop_heap(indices: &mut [usize], values: &[f32], first: usize, length: usize) {
    debug_assert!(length > 1);
    let top = indices[first];
    let hole = libcxx_floyd_sift_down(indices, values, first, length);
    let last = length - 1;
    if hole == last {
        indices[first + hole] = top;
    } else {
        indices[first + hole] = indices[first + last];
        indices[first + last] = top;
        libcxx_sift_up(indices, values, first, hole + 1);
    }
}

fn libcxx_introsort(
    indices: &mut [usize],
    values: &[f32],
    mut first: usize,
    mut last: usize,
    mut depth: usize,
    mut leftmost: bool,
) {
    loop {
        let length = last - first;
        match length {
            0 | 1 => return,
            2 => {
                if before(indices[first + 1], indices[first], values) {
                    indices.swap(first, first + 1);
                }
                return;
            }
            3 => {
                sort3(indices, values, first, first + 1, first + 2);
                return;
            }
            4 => {
                sort4(indices, values, first);
                return;
            }
            5 => {
                sort5(indices, values, first);
                return;
            }
            _ => {}
        }
        if length < 24 {
            insertion_sort(indices, values, first, last);
            return;
        }
        if depth == 0 {
            libcxx_partial_sort_all(indices, values, first, last);
            return;
        }
        depth -= 1;
        let half = length / 2;
        if length > 128 {
            sort3(indices, values, first, first + half, last - 1);
            sort3(indices, values, first + 1, first + half - 1, last - 2);
            sort3(indices, values, first + 2, first + half + 1, last - 3);
            sort3(
                indices,
                values,
                first + half - 1,
                first + half,
                first + half + 1,
            );
            indices.swap(first, first + half);
        } else {
            sort3(indices, values, first + half, first, last - 1);
        }
        if !leftmost && !before(indices[first - 1], indices[first], values) {
            first = partition_equals_left(indices, values, first, last);
            continue;
        }
        let (pivot, already_partitioned) = partition_equals_right(indices, values, first, last);
        if already_partitioned {
            let left_sorted = insertion_sort_incomplete(indices, values, first, pivot);
            if insertion_sort_incomplete(indices, values, pivot + 1, last) {
                if left_sorted {
                    return;
                }
                last = pivot;
                continue;
            } else if left_sorted {
                first = pivot + 1;
                continue;
            }
        }
        libcxx_introsort(indices, values, first, pivot, depth, leftmost);
        leftmost = false;
        first = pivot + 1;
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, fs, path::Path};

    use serde::Deserialize;

    use super::{libcxx_introsort, torch_unstable_argsort_desc};

    #[derive(Debug, Deserialize)]
    struct ArgsortVectorFile {
        format_version: u32,
        upstream_commit: String,
        torch_version: String,
        oracle_platform: String,
        argsort_contract: String,
        argsort_cases: Vec<ArgsortCase>,
    }

    #[derive(Debug, Deserialize)]
    struct ArgsortCase {
        name: String,
        values: Vec<f32>,
        expected: Vec<usize>,
    }

    fn argsort_vectors() -> Result<ArgsortVectorFile, Box<dyn Error>> {
        let path = std::env::var_os("RECORD_DECODE_VECTOR_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/record-decode-vectors.json")
            });
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    #[test]
    fn pinned_torch_unstable_argsort_permutations_match_exactly() -> Result<(), Box<dyn Error>> {
        let vectors = argsort_vectors()?;
        assert_eq!(vectors.format_version, 2);
        assert_eq!(
            vectors.upstream_commit,
            "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
        );
        assert_eq!(vectors.torch_version, "2.8.0");
        assert_eq!(vectors.oracle_platform, "darwin-arm64");
        assert_eq!(
            vectors.argsort_contract,
            "torch.argsort(descending=True, stable=False) exact permutation"
        );
        assert_eq!(vectors.argsort_cases.len(), 12);
        for width in [17, 33, 193] {
            assert!(
                vectors
                    .argsort_cases
                    .iter()
                    .any(|case| case.name == format!("all_equal_width_{width}")
                        && case.values.len() == width),
                "missing all-equal width {width} probe"
            );
            assert!(
                vectors
                    .argsort_cases
                    .iter()
                    .filter(|case| case.values.len() == width)
                    .count()
                    >= 4,
                "width {width} needs all-equal and diverse seeded probes"
            );
        }
        for case in vectors.argsort_cases {
            assert!(
                case.values.iter().all(|value| value.is_finite()),
                "{} values must be finite",
                case.name
            );
            assert_eq!(
                torch_unstable_argsort_desc(&case.values),
                case.expected,
                "{} exact permutation",
                case.name
            );
        }
        Ok(())
    }

    #[test]
    fn forced_depth_zero_fallback_sorts_distinct_values_descending() {
        let values: Vec<_> = (0..33)
            .map(|index| ((index * 17 + 11) % 33) as f32)
            .collect();
        let mut indices: Vec<_> = (0..values.len()).collect();
        let length = indices.len();
        libcxx_introsort(&mut indices, &values, 0, length, 0, true);

        assert!(
            indices
                .windows(2)
                .all(|pair| values[pair[0]] > values[pair[1]])
        );
        assert_eq!(
            indices,
            vec![
                9, 7, 5, 3, 1, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 31,
                29, 27, 25, 23, 21, 19, 17, 15, 13, 11,
            ]
        );
    }

    #[test]
    fn forced_depth_zero_fallback_matches_libcxx_tied_permutation() {
        let values: Vec<_> = (0..33).map(|index| ((index * 7 + 3) % 5) as f32).collect();
        let mut indices: Vec<_> = (0..values.len()).collect();
        let length = indices.len();
        libcxx_introsort(&mut indices, &values, 0, length, 0, true);

        // Captured from libc++ partial_sort(first, last, last) with the same
        // comparator. This covers tie-sensitive Floyd sift-down/sift-up moves.
        assert_eq!(
            indices,
            vec![
                28, 8, 18, 3, 23, 13, 25, 10, 5, 0, 20, 30, 15, 27, 12, 2, 22, 17, 7, 32, 29, 14,
                24, 4, 9, 19, 6, 26, 11, 21, 1, 16, 31,
            ]
        );
    }
}
