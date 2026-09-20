//! Deterministic minimum-cost rectangular assignment.
//!
//! This is a direct Rust port of GLiNER2's dependency-free shortest
//! augmenting-path branch. In particular, candidate columns are visited in
//! ascending order and all comparisons are strict so upstream tie behavior is
//! preserved.

use std::{error::Error, fmt};

use ndarray::ArrayView2;

/// Failure returned when an assignment cannot be evaluated safely in `f64`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssignmentError {
    /// The source matrix contains a NaN.
    NaN,
    /// The finite replacement derived for a source infinity overflowed.
    SentinelOverflow,
    /// Finite source costs produced non-finite Hungarian working arithmetic.
    ArithmeticOverflow,
}

impl fmt::Display for AssignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaN => formatter.write_str("cost matrix contains NaN"),
            Self::SentinelOverflow => {
                formatter.write_str("infinity replacement sentinel overflows f64")
            }
            Self::ArithmeticOverflow => {
                formatter.write_str("cost range overflows Hungarian f64 arithmetic")
            }
        }
    }
}

impl Error for AssignmentError {}

/// Find a deterministic minimum-cost assignment for a rectangular matrix.
///
/// The returned vectors contain `min(rows, columns)` pairs and row indices are
/// ascending. Source infinities follow upstream behavior: they are replaced by
/// `+/- 1e6 * (max_abs_finite + 1)`, using a scale of one when every value is
/// infinite. NaNs and non-finite derived arithmetic are rejected.
pub fn linear_sum_assignment(
    cost_matrix: ArrayView2<'_, f64>,
) -> Result<(Vec<usize>, Vec<usize>), AssignmentError> {
    let mut has_infinity = false;
    let mut has_finite = false;
    let mut max_abs_finite = 0.0_f64;
    for &value in cost_matrix {
        if value.is_nan() {
            return Err(AssignmentError::NaN);
        }
        if value.is_infinite() {
            has_infinity = true;
        } else {
            has_finite = true;
            max_abs_finite = max_abs_finite.max(value.abs());
        }
    }

    let sentinel = if has_infinity {
        let scale = if has_finite { max_abs_finite } else { 1.0 };
        let big = 1.0e6 * (scale + 1.0);
        if !big.is_finite() {
            return Err(AssignmentError::SentinelOverflow);
        }
        Some(big)
    } else {
        None
    };

    let (rows, columns) = cost_matrix.dim();
    if rows == 0 || columns == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let transposed = columns < rows;
    let (n, m) = if transposed {
        (columns, rows)
    } else {
        (rows, columns)
    };

    // Match upstream's transpose-then-list materialization. This also makes
    // arbitrary/non-contiguous ndarray views behave identically.
    let mut cost = vec![vec![0.0; m]; n];
    for (i, output_row) in cost.iter_mut().enumerate() {
        for (j, output) in output_row.iter_mut().enumerate() {
            let value = if transposed {
                cost_matrix[(j, i)]
            } else {
                cost_matrix[(i, j)]
            };
            *output = match (sentinel, value) {
                (Some(big), value) if value == f64::INFINITY => big,
                (Some(big), value) if value == f64::NEG_INFINITY => -big,
                _ => value,
            };
        }
    }

    // Jonker-Volgenant-style shortest augmenting path, ported statement for
    // statement from the pinned Python implementation. Arrays are 1-indexed.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0_usize; m + 1];
    let mut way = vec![0_usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = None;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if !cur.is_finite() {
                        return Err(AssignmentError::ArithmeticOverflow);
                    }
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = Some(j);
                    }
                }
            }
            let Some(next_column) = j1 else {
                return Err(AssignmentError::ArithmeticOverflow);
            };
            if !delta.is_finite() {
                return Err(AssignmentError::ArithmeticOverflow);
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                    if !u[p[j]].is_finite() || !v[j].is_finite() {
                        return Err(AssignmentError::ArithmeticOverflow);
                    }
                } else {
                    minv[j] -= delta;
                    if !minv[j].is_finite() {
                        return Err(AssignmentError::ArithmeticOverflow);
                    }
                }
            }
            j0 = next_column;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(n);
    for (j, &assigned_row) in p.iter().enumerate().skip(1) {
        if assigned_row != 0 {
            pairs.push((assigned_row - 1, j - 1));
        }
    }
    pairs.sort_unstable();

    if transposed {
        pairs = pairs
            .into_iter()
            .map(|(row, column)| (column, row))
            .collect();
        pairs.sort_unstable();
    }

    let (row_indices, column_indices) = pairs.into_iter().unzip();
    Ok((row_indices, column_indices))
}
