use std::{collections::HashSet, error::Error, fs, path::Path};

use gliner2_rs::boundary::assignment::{AssignmentError, linear_sum_assignment};
use ndarray::{Array2, array};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    backend: String,
    seed: u64,
    cases: Vec<VectorCase>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum CostValue {
    Number(f64),
    Symbol(String),
}

impl CostValue {
    fn value(&self) -> Result<f64, String> {
        match self {
            Self::Number(value) => Ok(*value),
            Self::Symbol(symbol) if symbol == "+inf" => Ok(f64::INFINITY),
            Self::Symbol(symbol) if symbol == "-inf" => Ok(f64::NEG_INFINITY),
            Self::Symbol(symbol) => Err(format!("unsupported cost symbol {symbol:?}")),
        }
    }
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    rows: usize,
    columns: usize,
    cost: Vec<Vec<CostValue>>,
    expected_rows: Vec<usize>,
    expected_columns: Vec<usize>,
}

fn vectors() -> Result<VectorFile, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/assignment-vectors.json");
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn matrix(case: &VectorCase) -> Result<Array2<f64>, Box<dyn Error>> {
    if case.cost.len() != case.rows {
        return Err(format!("{}: fixture row count mismatch", case.name).into());
    }
    let mut values = Vec::with_capacity(case.rows * case.columns);
    for row in &case.cost {
        if row.len() != case.columns {
            return Err(format!("{}: fixture column count mismatch", case.name).into());
        }
        for value in row {
            values.push(value.value()?);
        }
    }
    Ok(Array2::from_shape_vec((case.rows, case.columns), values)?)
}

fn normalized_costs(cost: &Array2<f64>) -> Vec<Vec<f64>> {
    let has_infinity = cost.iter().any(|value| value.is_infinite());
    let finite: Vec<_> = cost
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let scale = finite
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let big = 1.0e6 * (if finite.is_empty() { 1.0 } else { scale } + 1.0);
    (0..cost.nrows())
        .map(|row| {
            (0..cost.ncols())
                .map(|column| match cost[(row, column)] {
                    value if has_infinity && value == f64::INFINITY => big,
                    value if has_infinity && value == f64::NEG_INFINITY => -big,
                    value => value,
                })
                .collect()
        })
        .collect()
}

fn enumerate_minimum(cost: &[Vec<f64>], rows: usize, columns: usize) -> f64 {
    fn visit_rows(
        row: usize,
        rows: usize,
        columns: usize,
        cost: &[Vec<f64>],
        used: &mut [bool],
        total: f64,
        best: &mut f64,
    ) {
        if row == rows {
            *best = best.min(total);
            return;
        }
        for column in 0..columns {
            if !used[column] {
                used[column] = true;
                visit_rows(
                    row + 1,
                    rows,
                    columns,
                    cost,
                    used,
                    total + cost[row][column],
                    best,
                );
                used[column] = false;
            }
        }
    }

    fn visit_columns(
        column: usize,
        rows: usize,
        columns: usize,
        cost: &[Vec<f64>],
        used: &mut [bool],
        total: f64,
        best: &mut f64,
    ) {
        if column == columns {
            *best = best.min(total);
            return;
        }
        for row in 0..rows {
            if !used[row] {
                used[row] = true;
                visit_columns(
                    column + 1,
                    rows,
                    columns,
                    cost,
                    used,
                    total + cost[row][column],
                    best,
                );
                used[row] = false;
            }
        }
    }

    if rows == 0 || columns == 0 {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    if rows <= columns {
        visit_rows(
            0,
            rows,
            columns,
            cost,
            &mut vec![false; columns],
            0.0,
            &mut best,
        );
    } else {
        visit_columns(
            0,
            rows,
            columns,
            cost,
            &mut vec![false; rows],
            0.0,
            &mut best,
        );
    }
    best
}

#[test]
fn pinned_internal_backend_vectors_match_every_index() -> Result<(), Box<dyn Error>> {
    let vectors = vectors()?;
    assert_eq!(vectors.format_version, 1);
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert_eq!(
        vectors.backend,
        "internal_shortest_augmenting_path_scipy_absent"
    );
    assert_eq!(vectors.seed, 1729);
    assert!(vectors.cases.len() >= 28);

    for case in vectors.cases {
        let cost = matrix(&case)?;
        let (rows, columns) = linear_sum_assignment(cost.view())?;
        assert_eq!(rows, case.expected_rows, "{} rows", case.name);
        assert_eq!(columns, case.expected_columns, "{} columns", case.name);

        let pair_count = case.rows.min(case.columns);
        assert_eq!(rows.len(), pair_count, "{} pair count", case.name);
        assert_eq!(columns.len(), pair_count, "{} pair count", case.name);
        assert!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "{} rows are not strictly ascending: {rows:?}",
            case.name
        );
        assert!(
            rows.iter().all(|&row| row < case.rows),
            "{} row out of range",
            case.name
        );
        assert!(
            columns.iter().all(|&column| column < case.columns),
            "{} column out of range",
            case.name
        );
        assert_eq!(
            columns.iter().copied().collect::<HashSet<_>>().len(),
            pair_count,
            "{} repeats a column",
            case.name
        );

        let normalized = normalized_costs(&cost);
        let actual_total: f64 = rows
            .iter()
            .zip(&columns)
            .map(|(&row, &column)| normalized[row][column])
            .sum();
        let minimum = enumerate_minimum(&normalized, case.rows, case.columns);
        assert_eq!(actual_total, minimum, "{} is not minimum-cost", case.name);
    }
    Ok(())
}

#[test]
fn validates_non_finite_and_extreme_costs() {
    let nan = array![[0.0, f64::NAN]];
    assert_eq!(linear_sum_assignment(nan.view()), Err(AssignmentError::NaN));

    let overflowing_sentinel = array![[f64::MAX, f64::INFINITY]];
    assert_eq!(
        linear_sum_assignment(overflowing_sentinel.view()),
        Err(AssignmentError::SentinelOverflow)
    );

    let overflowing_work = array![[f64::MAX, -f64::MAX], [-f64::MAX, f64::MAX]];
    assert_eq!(
        linear_sum_assignment(overflowing_work.view()),
        Err(AssignmentError::ArithmeticOverflow)
    );
}

#[test]
fn accepts_f32_derived_costs_promoted_to_f64() {
    let promoted = array![[0.1_f32, 0.2], [0.3, -0.4]].mapv(f64::from);
    assert_eq!(
        linear_sum_assignment(promoted.view()).unwrap(),
        (vec![0, 1], vec![0, 1])
    );
}
