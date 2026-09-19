//! Pure-Rust shared candidate-pool selection for boundary models.
//!
//! This reproduces the batch-one inference path of upstream
//! `DocumentCandidatePool`. Inputs are already projected by the shared pool's
//! `start_projection` and `end_projection`; query-head projections are not
//! interchangeable with these tensors.

use anyhow::{Result, bail, ensure};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;

/// Finite masking sentinel used by the pinned upstream implementation.
pub const MASK_LOGIT: f32 = -10_000.0;

/// Shared candidate-pool settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolConfig {
    pub boundary_top_k: usize,
    pub capacity: usize,
    pub min_per_query: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            boundary_top_k: 32,
            capacity: 192,
            min_per_query: 8,
        }
    }
}

/// A padded, query-agnostic candidate pool.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidatePool {
    /// Half-open `[start, end)` boundary indices, shape `[capacity, 2]`.
    pub indices: Array2<i64>,
    /// True for retained candidates, shape `[capacity]`.
    pub mask: Array1<bool>,
    /// Marginal-free projected endpoint compatibility, shape `[capacity]`.
    pub compat_logits: Array1<f32>,
    /// Compatibility plus union start/end marginals, shape `[capacity]`.
    pub proposal_logits: Array1<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Endpoint {
    index: usize,
    valid: bool,
}

#[derive(Clone, Copy, Debug)]
struct Pair {
    start: usize,
    end: usize,
    valid: bool,
    compat: f32,
    global_score: f32,
}

#[derive(Clone, Copy, Debug)]
struct RankedKey {
    key: i64,
    score: f32,
    valid: bool,
    keep: bool,
}

fn descending_f32(left: f32, right: f32) -> Ordering {
    // Inputs are validated as finite. `partial_cmp` intentionally considers
    // -0.0 and +0.0 equal, preserving upstream's stable input-order tie break.
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn validate_inputs(
    boundary_mask: ArrayView1<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
    start_logits: ArrayView2<'_, f32>,
    end_logits: ArrayView2<'_, f32>,
    start_projection: ArrayView2<'_, f32>,
    end_projection: ArrayView2<'_, f32>,
    config: PoolConfig,
) -> Result<(usize, usize, usize)> {
    let n = boundary_mask.len();
    let q = query_mask.len();

    ensure!(n >= 1, "boundary pool requires at least one boundary");
    ensure!(
        q >= 1,
        "boundary pool requires at least one query; upstream amax over Q is undefined for Q=0"
    );
    ensure!(
        start_logits.dim() == (q, n),
        "start_logits must have shape [Q,N]=[{q},{n}], got {:?}",
        start_logits.dim()
    );
    ensure!(
        end_logits.dim() == (q, n),
        "end_logits must have shape [Q,N]=[{q},{n}], got {:?}",
        end_logits.dim()
    );
    ensure!(
        start_projection.nrows() == n,
        "start_projection must have N={n} rows, got {}",
        start_projection.nrows()
    );
    ensure!(
        end_projection.nrows() == n,
        "end_projection must have N={n} rows, got {}",
        end_projection.nrows()
    );
    let d = start_projection.ncols();
    ensure!(d >= 1, "pool projection width D must be at least one");
    ensure!(
        end_projection.ncols() == d,
        "projection widths differ: start D={d}, end D={}",
        end_projection.ncols()
    );
    ensure!(
        start_logits.iter().all(|value| value.is_finite()),
        "start_logits contains a non-finite value"
    );
    ensure!(
        end_logits.iter().all(|value| value.is_finite()),
        "end_logits contains a non-finite value"
    );
    ensure!(
        start_projection.iter().all(|value| value.is_finite()),
        "start_projection contains a non-finite value"
    );
    ensure!(
        end_projection.iter().all(|value| value.is_finite()),
        "end_projection contains a non-finite value"
    );

    let k = config.boundary_top_k.min(n);
    k.checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("boundary Cartesian product overflows usize"))?;
    let n_i64 = i64::try_from(n)
        .map_err(|_| anyhow::anyhow!("boundary count {n} cannot be encoded as i64"))?;
    n_i64
        .checked_mul(n_i64)
        .ok_or_else(|| anyhow::anyhow!("encoded span-key sentinel overflows i64"))?;
    q.checked_mul(config.min_per_query.min(k.saturating_mul(k)))
        .ok_or_else(|| anyhow::anyhow!("query quota allocation overflows usize"))?;

    Ok((n, q, d))
}

fn union_logits(
    logits: ArrayView2<'_, f32>,
    boundary_mask: ArrayView1<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
) -> Vec<f32> {
    (0..boundary_mask.len())
        .map(|boundary| {
            let mut best = f32::NEG_INFINITY;
            for query in 0..query_mask.len() {
                let value = if boundary_mask[boundary] && query_mask[query] {
                    logits[(query, boundary)]
                } else {
                    MASK_LOGIT
                };
                best = best.max(value);
            }
            best
        })
        .collect()
}

fn select_top_boundaries(scores: &[f32], valid: &[bool], k: usize) -> Vec<Endpoint> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    // Stable descending sort leaves original boundary index as the tie-break.
    order.sort_by(|&left, &right| {
        let left_score = if valid[left] {
            scores[left]
        } else {
            MASK_LOGIT
        };
        let right_score = if valid[right] {
            scores[right]
        } else {
            MASK_LOGIT
        };
        descending_f32(left_score, right_score)
    });
    order
        .into_iter()
        .take(k)
        .map(|index| {
            if valid[index] {
                Endpoint { index, valid: true }
            } else {
                // `select_top_boundaries` zeroes invalid selected indices.
                Endpoint {
                    index: 0,
                    valid: false,
                }
            }
        })
        .collect()
}

/// Reproduce the pinned oracle's PyTorch CPU inner-sum reduction.
///
/// PyTorch 2.8's `SumKernel.cpp` uses a four-way ILP cascade over SIMD vectors,
/// then reduces SIMD lanes in order. The committed oracle was generated on
/// AArch64, where an fp32 vector has four lanes. Expressing that grouping with
/// scalar f32 operations makes the committed result portable and bit-exact; it
/// intentionally does not vary with the Rust host's native SIMD width. A newly
/// generated x86 PyTorch oracle can differ because PyTorch's native vector width
/// is architecture dependent.
#[allow(clippy::needless_range_loop)] // Indexed loops mirror PyTorch's reduction grouping.
fn compatibility(
    start_projection: ArrayView2<'_, f32>,
    end_projection: ArrayView2<'_, f32>,
    start: usize,
    end: usize,
    scale: f32,
) -> f32 {
    const LANES: usize = 4;
    const ILP: usize = 4;
    const LEVELS: usize = 4;

    let start_row = start_projection.row(start);
    let end_row = end_projection.row(end);
    let d = start_row.len();
    let vector_count = d / LANES;
    let cascade_size = vector_count / ILP;
    let ceil_log2 = if cascade_size <= 1 {
        0
    } else {
        usize::BITS as usize - (cascade_size - 1).leading_zeros() as usize
    };
    let level_power = 4_usize.max(ceil_log2 / LEVELS);
    let level_step = 1_usize << level_power;
    let level_mask = level_step - 1;
    let mut accumulators = [[[0.0_f32; LANES]; ILP]; LEVELS];
    let mut index = 0;

    fn add_vector(
        accumulator: &mut [f32; LANES],
        start_row: &ArrayView1<'_, f32>,
        end_row: &ArrayView1<'_, f32>,
        vector: usize,
    ) {
        for lane in 0..LANES {
            let element = vector * LANES + lane;
            accumulator[lane] += start_row[element] * end_row[element];
        }
    }

    while index + level_step <= cascade_size {
        for _ in 0..level_step {
            for partial in 0..ILP {
                add_vector(
                    &mut accumulators[0][partial],
                    &start_row,
                    &end_row,
                    index * ILP + partial,
                );
            }
            index += 1;
        }
        for level in 1..LEVELS {
            for partial in 0..ILP {
                for lane in 0..LANES {
                    accumulators[level][partial][lane] += accumulators[level - 1][partial][lane];
                    accumulators[level - 1][partial][lane] = 0.0;
                }
            }
            let mask = level_mask << (level * level_power);
            if index & mask != 0 {
                break;
            }
        }
    }
    while index < cascade_size {
        for partial in 0..ILP {
            add_vector(
                &mut accumulators[0][partial],
                &start_row,
                &end_row,
                index * ILP + partial,
            );
        }
        index += 1;
    }

    for level in 1..LEVELS {
        for partial in 0..ILP {
            for lane in 0..LANES {
                accumulators[0][partial][lane] += accumulators[level][partial][lane];
            }
        }
    }
    for vector in (cascade_size * ILP)..vector_count {
        for lane in 0..LANES {
            let element = vector * LANES + lane;
            accumulators[0][0][lane] += start_row[element] * end_row[element];
        }
    }
    for partial in 1..ILP {
        for lane in 0..LANES {
            accumulators[0][0][lane] += accumulators[0][partial][lane];
        }
    }

    // PyTorch accumulates a scalar tail before horizontally adding vector lanes.
    let mut sum = 0.0_f32;
    for element in (vector_count * LANES)..d {
        sum += start_row[element] * end_row[element];
    }
    for lane in 0..LANES {
        sum += accumulators[0][0][lane];
    }
    sum / scale
}

fn deduplicate_pool(
    mut rows: Vec<RankedKey>,
    capacity: usize,
    invalid_key: i64,
) -> Vec<(i64, bool)> {
    for row in &mut rows {
        if !row.valid {
            row.key = invalid_key;
            row.score = MASK_LOGIT;
        }
    }

    // This sequence is literal upstream `_deduplicate_pool`: stable score-desc,
    // then stable key-asc, keep the first valid key occurrence, then stable
    // effective-priority-desc. The last sort therefore breaks equal priorities
    // by encoded span key.
    rows.sort_by(|left, right| descending_f32(left.score, right.score));
    rows.sort_by_key(|row| row.key);

    let mut previous_key = None;
    for row in &mut rows {
        let first = previous_key != Some(row.key);
        previous_key = Some(row.key);
        row.keep = row.valid && first;
    }
    rows.sort_by(|left, right| {
        let left_score = if left.keep { left.score } else { MASK_LOGIT };
        let right_score = if right.keep { right.score } else { MASK_LOGIT };
        descending_f32(left_score, right_score)
    });

    let take = capacity.min(rows.len());
    let mut selected: Vec<_> = rows
        .into_iter()
        .take(take)
        .map(|row| (row.key, row.keep))
        .collect();
    selected.resize(capacity, (0, false));
    selected
}

/// Build the shared, batch-one document candidate pool.
///
/// `start_projection` and `end_projection` must be outputs of the shared pool
/// builder's endpoint projections, both `[N,D]`. The returned arrays are padded
/// to `config.capacity`.
#[allow(clippy::too_many_arguments)]
pub fn build_shared_candidate_pool(
    boundary_mask: ArrayView1<'_, bool>,
    query_mask: ArrayView1<'_, bool>,
    start_logits: ArrayView2<'_, f32>,
    end_logits: ArrayView2<'_, f32>,
    start_projection: ArrayView2<'_, f32>,
    end_projection: ArrayView2<'_, f32>,
    config: PoolConfig,
) -> Result<CandidatePool> {
    let (n, q, d) = validate_inputs(
        boundary_mask,
        query_mask,
        start_logits,
        end_logits,
        start_projection,
        end_projection,
        config,
    )?;
    let n_i64 = i64::try_from(n)?;
    let invalid_key = n_i64
        .checked_mul(n_i64)
        .ok_or_else(|| anyhow::anyhow!("encoded span-key sentinel overflows i64"))?;

    let union_start = union_logits(start_logits, boundary_mask, query_mask);
    let union_end = union_logits(end_logits, boundary_mask, query_mask);
    let any_query = query_mask.iter().copied().any(|valid| valid);
    let union_valid: Vec<bool> = boundary_mask
        .iter()
        .copied()
        .map(|valid| valid && any_query)
        .collect();
    let k = config.boundary_top_k.min(n);
    let starts = select_top_boundaries(&union_start, &union_valid, k);
    let ends = select_top_boundaries(&union_end, &union_valid, k);
    let pair_count = k
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("boundary Cartesian product overflows usize"))?;
    let scale = (d as f32).sqrt();
    let mut pairs = Vec::with_capacity(pair_count);

    // Start-major Cartesian order matches expand(...).reshape(...) upstream.
    for start in &starts {
        for end in &ends {
            let valid = start.valid && end.valid && end.index > start.index;
            let compat = compatibility(
                start_projection,
                end_projection,
                start.index,
                end.index,
                scale,
            );
            // Association is significant: `(compat + union_start) + union_end`.
            let global_score = (compat + union_start[start.index]) + union_end[end.index];
            ensure!(
                compat.is_finite() && global_score.is_finite(),
                "pool compatibility or proposal score overflowed for [{},{})",
                start.index,
                end.index
            );
            pairs.push(Pair {
                start: start.index,
                end: end.index,
                valid,
                compat,
                global_score,
            });
        }
    }

    let quota = config.min_per_query.min(pair_count);
    let quota_len = q
        .checked_mul(quota)
        .ok_or_else(|| anyhow::anyhow!("query quota allocation overflows usize"))?;
    let total_rows = quota_len
        .checked_add(pair_count)
        .ok_or_else(|| anyhow::anyhow!("pool candidate allocation overflows usize"))?;
    let mut rows = Vec::with_capacity(total_rows);

    for query in 0..q {
        let scores: Vec<f32> = pairs
            .iter()
            .map(|pair| {
                if pair.valid && query_mask[query] {
                    // Association is significant: `(start + end) + compat`.
                    (start_logits[(query, pair.start)] + end_logits[(query, pair.end)])
                        + pair.compat
                } else {
                    MASK_LOGIT
                }
            })
            .collect();
        ensure!(
            scores.iter().all(|score| score.is_finite()),
            "pool quota score overflowed for query {query}"
        );
        let mut ranked: Vec<usize> = (0..pair_count).collect();
        ranked.sort_by(|&left, &right| descending_f32(scores[left], scores[right]));
        for (rank, pair_index) in ranked.into_iter().take(quota).enumerate() {
            let pair = pairs[pair_index];
            let key = i64::try_from(pair.start)? * n_i64 + i64::try_from(pair.end)?;
            // Upstream reserves 5000+quota through 5001, irrespective of score.
            let rank_bonus = (quota - rank) as f32;
            rows.push(RankedKey {
                key,
                score: 5_000.0_f32 + rank_bonus,
                valid: pair.valid && query_mask[query],
                keep: false,
            });
        }
    }

    for pair in &pairs {
        let key = i64::try_from(pair.start)? * n_i64 + i64::try_from(pair.end)?;
        rows.push(RankedKey {
            key,
            score: pair.global_score,
            valid: pair.valid,
            keep: false,
        });
    }

    let selected = deduplicate_pool(rows, config.capacity, invalid_key);
    let mut indices = Array2::<i64>::zeros((config.capacity, 2));
    let mut mask = Array1::<bool>::from_elem(config.capacity, false);
    let mut compat_logits = Array1::<f32>::zeros(config.capacity);
    let mut proposal_logits = Array1::<f32>::from_elem(config.capacity, MASK_LOGIT);

    for (slot, (key, valid)) in selected.into_iter().enumerate() {
        if !valid {
            continue;
        }
        let start_i64 = key / n_i64;
        let end_i64 = key - start_i64 * n_i64;
        let start = usize::try_from(start_i64)
            .map_err(|_| anyhow::anyhow!("selected negative start boundary {start_i64}"))?;
        let end = usize::try_from(end_i64)
            .map_err(|_| anyhow::anyhow!("selected negative end boundary {end_i64}"))?;
        if start >= n || end >= n {
            bail!("selected span [{start},{end}) is outside N={n}");
        }
        let compat = compatibility(start_projection, end_projection, start, end, scale);
        indices[(slot, 0)] = start_i64;
        indices[(slot, 1)] = end_i64;
        mask[slot] = true;
        compat_logits[slot] = compat;
        proposal_logits[slot] = (compat + union_start[start]) + union_end[end];
    }

    Ok(CandidatePool {
        indices,
        mask,
        compat_logits,
        proposal_logits,
    })
}
