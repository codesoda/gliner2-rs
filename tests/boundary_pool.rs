use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use gliner2_rs::boundary::{
    marginals::{MarginalInput, MarginalModel},
    pool::{CandidatePool, PoolConfig, build_shared_candidate_pool},
};
use ndarray::{Array1, Array2, Array3, Axis, Ix1, Ix2};
use ndarray_npy::NpzReader;

#[allow(dead_code)]
mod common;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    config: VectorConfig,
    boundary_mask: Vec<bool>,
    query_mask: Vec<bool>,
    start_logits: Vec<Vec<f32>>,
    end_logits: Vec<Vec<f32>>,
    start_projection: Vec<Vec<f32>>,
    end_projection: Vec<Vec<f32>>,
    expected: VectorExpected,
}

#[derive(Debug, Deserialize)]
struct VectorConfig {
    boundary_top_k: usize,
    capacity: usize,
    min_per_query: usize,
}

#[derive(Debug, Deserialize)]
struct VectorExpected {
    indices: Vec<Vec<i64>>,
    mask: Vec<bool>,
    compat_logits: Vec<f32>,
    proposal_logits: Vec<f32>,
}

fn array2_f32(rows: Vec<Vec<f32>>, name: &str) -> Result<Array2<f32>> {
    let row_count = rows.len();
    let columns = rows.first().map_or(0, Vec::len);
    ensure!(
        rows.iter().all(|row| row.len() == columns),
        "{name} is ragged"
    );
    Array2::from_shape_vec((row_count, columns), rows.into_iter().flatten().collect())
        .with_context(|| format!("invalid {name} shape"))
}

fn array2_i64(rows: Vec<Vec<i64>>, name: &str) -> Result<Array2<i64>> {
    let row_count = rows.len();
    let columns = rows.first().map_or(0, Vec::len);
    ensure!(
        rows.iter().all(|row| row.len() == columns),
        "{name} is ragged"
    );
    Array2::from_shape_vec((row_count, columns), rows.into_iter().flatten().collect())
        .with_context(|| format!("invalid {name} shape"))
}

fn first_array_mismatch<T: PartialEq + std::fmt::Debug>(
    actual: impl IntoIterator<Item = T>,
    expected: impl IntoIterator<Item = T>,
) -> Option<(usize, T, T)> {
    actual
        .into_iter()
        .zip(expected)
        .enumerate()
        .find_map(|(index, (actual, expected))| {
            (actual != expected).then_some((index, actual, expected))
        })
}

fn assert_discrete_exact(
    case: &str,
    actual: &CandidatePool,
    indices: &Array2<i64>,
    mask: &Array1<bool>,
) {
    assert_eq!(
        actual.indices.dim(),
        indices.dim(),
        "{case}: index shape differs"
    );
    if let Some((offset, actual_value, expected_value)) =
        first_array_mismatch(actual.indices.iter().copied(), indices.iter().copied())
    {
        panic!(
            "{case}: exact index/order mismatch at flat offset {offset}: Rust {actual_value}, oracle {expected_value}"
        );
    }
    assert_eq!(actual.mask, *mask, "{case}: exact candidate mask differs");
}

fn assert_float_exact(case: &str, tensor: &str, actual: &Array1<f32>, expected: &Array1<f32>) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case}: {tensor} length differs"
    );
    let mut unequal = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut first = None;
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        if actual_value.to_bits() != expected_value.to_bits() {
            unequal += 1;
            max_abs = max_abs.max((actual_value - expected_value).abs());
            first.get_or_insert((index, actual_value, expected_value));
        }
    }
    assert!(
        unequal == 0,
        "{case}: {tensor} is not bit-exact: {unequal}/{} unequal, max_abs={max_abs:e}, first={first:?}. No tolerance is approved; Rust deliberately reproduces the pinned AArch64 PyTorch four-lane reduction grouping",
        actual.len()
    );
}

fn run_vector_case(case: VectorCase) -> Result<()> {
    let expected_indices = array2_i64(case.expected.indices, "expected indices")?;
    let expected_mask = Array1::from(case.expected.mask);
    let expected_compat = Array1::from(case.expected.compat_logits);
    let expected_proposal = Array1::from(case.expected.proposal_logits);
    let actual = build_shared_candidate_pool(
        Array1::from(case.boundary_mask).view(),
        Array1::from(case.query_mask).view(),
        array2_f32(case.start_logits, "start_logits")?.view(),
        array2_f32(case.end_logits, "end_logits")?.view(),
        array2_f32(case.start_projection, "start_projection")?.view(),
        array2_f32(case.end_projection, "end_projection")?.view(),
        PoolConfig {
            boundary_top_k: case.config.boundary_top_k,
            capacity: case.config.capacity,
            min_per_query: case.config.min_per_query,
        },
    )?;

    assert_discrete_exact(&case.name, &actual, &expected_indices, &expected_mask);
    assert_float_exact(
        &case.name,
        "compat_logits",
        &actual.compat_logits,
        &expected_compat,
    );
    assert_float_exact(
        &case.name,
        "proposal_logits",
        &actual.proposal_logits,
        &expected_proposal,
    );
    Ok(())
}

#[test]
fn synthetic_upstream_vectors_are_exact() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/pool-test-vectors.json");
    let vectors: VectorFile =
        serde_json::from_slice(&fs::read(&path).with_context(|| path.display().to_string())?)?;
    assert_eq!(vectors.format_version, 1);
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert_eq!(vectors.cases.len(), 9);
    for case in vectors.cases {
        run_vector_case(case)?;
    }
    Ok(())
}

#[test]
fn malformed_inputs_return_errors_without_panics() {
    let empty_mask = Array1::<bool>::from(vec![]);
    let one_mask = Array1::from(vec![true]);
    let empty_logits = Array2::<f32>::zeros((1, 0));
    let empty_projection = Array2::<f32>::zeros((0, 1));
    assert!(
        build_shared_candidate_pool(
            empty_mask.view(),
            one_mask.view(),
            empty_logits.view(),
            empty_logits.view(),
            empty_projection.view(),
            empty_projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );

    let no_queries = Array1::<bool>::from(vec![]);
    let no_query_logits = Array2::<f32>::zeros((0, 1));
    let projection = Array2::<f32>::zeros((1, 1));
    assert!(
        build_shared_candidate_pool(
            one_mask.view(),
            no_queries.view(),
            no_query_logits.view(),
            no_query_logits.view(),
            projection.view(),
            projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );

    let nan_logits = Array2::from_elem((1, 1), f32::NAN);
    let finite_logits = Array2::<f32>::zeros((1, 1));
    assert!(
        build_shared_candidate_pool(
            one_mask.view(),
            one_mask.view(),
            nan_logits.view(),
            finite_logits.view(),
            projection.view(),
            projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );

    let wrong_logits = Array2::<f32>::zeros((1, 2));
    assert!(
        build_shared_candidate_pool(
            one_mask.view(),
            one_mask.view(),
            wrong_logits.view(),
            finite_logits.view(),
            projection.view(),
            projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );
}

#[test]
fn finite_inputs_that_overflow_are_rejected() {
    let boundaries = Array1::from(vec![true, true]);
    let queries = Array1::from(vec![true]);
    let zeros = Array2::<f32>::zeros((1, 2));
    let huge_projection = Array2::from_elem((2, 4), f32::MAX);
    assert!(
        build_shared_candidate_pool(
            boundaries.view(),
            queries.view(),
            zeros.view(),
            zeros.view(),
            huge_projection.view(),
            huge_projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );
    let huge_logits = Array2::from_elem((1, 2), f32::MAX);
    let zero_projection = Array2::<f32>::zeros((2, 4));
    assert!(
        build_shared_candidate_pool(
            boundaries.view(),
            queries.view(),
            huge_logits.view(),
            huge_logits.view(),
            zero_projection.view(),
            zero_projection.view(),
            PoolConfig::default(),
        )
        .is_err()
    );
}

struct GoldenPool {
    boundary_mask: Array1<bool>,
    query_mask: Array1<bool>,
    start_logits: Array2<f32>,
    end_logits: Array2<f32>,
    start_projection: Array2<f32>,
    end_projection: Array2<f32>,
    indices: Array2<i64>,
    mask: Array1<bool>,
    compat_logits: Array1<f32>,
    proposal_logits: Array1<f32>,
}

fn load_golden(path: &Path) -> Result<Option<GoldenPool>> {
    let file = File::open(path).with_context(|| path.display().to_string())?;
    let mut npz = NpzReader::new(file)?;
    if !npz
        .names()?
        .iter()
        .any(|name| name.trim_end_matches(".npy") == "pool_0_indices")
    {
        return Ok(None);
    }

    let boundary_mask: Array2<bool> = npz.by_name("pool_0_boundary_mask")?;
    let query_mask: Array2<bool> = npz.by_name("pool_0_query_mask")?;
    let start_logits: Array3<f32> = npz.by_name("pool_0_start_logits")?;
    let end_logits: Array3<f32> = npz.by_name("pool_0_end_logits")?;
    let start_projection: Array3<f32> = npz.by_name("pool_start_projection_0_output")?;
    let end_projection: Array3<f32> = npz.by_name("pool_end_projection_0_output")?;
    let indices: Array3<i64> = npz.by_name("pool_0_indices")?;
    let mask: Array2<bool> = npz.by_name("pool_0_mask")?;
    let compat_logits: Array2<f32> = npz.by_name("pool_0_compat_logits")?;
    let proposal_logits: Array2<f32> = npz.by_name("pool_0_proposal_logits")?;

    ensure!(
        boundary_mask.nrows() == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        query_mask.nrows() == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        start_logits.len_of(Axis(0)) == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        end_logits.len_of(Axis(0)) == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        start_projection.len_of(Axis(0)) == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        end_projection.len_of(Axis(0)) == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        indices.len_of(Axis(0)) == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(mask.nrows() == 1, "{}: batch must be one", path.display());
    ensure!(
        compat_logits.nrows() == 1,
        "{}: batch must be one",
        path.display()
    );
    ensure!(
        proposal_logits.nrows() == 1,
        "{}: batch must be one",
        path.display()
    );

    Ok(Some(GoldenPool {
        boundary_mask: boundary_mask
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix1>()?,
        query_mask: query_mask
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix1>()?,
        start_logits: start_logits
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix2>()?,
        end_logits: end_logits
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix2>()?,
        start_projection: start_projection
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix2>()?,
        end_projection: end_projection
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix2>()?,
        indices: indices
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix2>()?,
        mask: mask
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix1>()?,
        compat_logits: compat_logits
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix1>()?,
        proposal_logits: proposal_logits
            .index_axis_move(Axis(0), 0)
            .into_dimensionality::<Ix1>()?,
    }))
}

/// Unlike the frozen-input exact tests, this exercises real ORT projections.
/// Only floating values get tolerance; selected spans and ordering remain exact.
#[test]
fn onnx_marginals_produce_exact_candidate_selection() -> Result<()> {
    let graph = common::model_root().join("onnx/gliner2.5-base-v1/boundary_marginals.onnx");
    if !graph.is_file() {
        ensure!(
            std::env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() != Ok("1"),
            "missing boundary graph: {}",
            graph.display()
        );
        eprintln!("SKIP: missing boundary graph {}", graph.display());
        return Ok(());
    }
    let full = std::env::var("GLINER2_POOL_FULL_FIXTURES").as_deref() == Ok("1");
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join(if full {
        "fixtures/gliner2.5-base-v1"
    } else {
        "fixtures/gliner2.5-base-v1-subset"
    });
    let model = MarginalModel::new(&graph)?;
    let mut cases = 0;
    for path in npz_paths(&directory)? {
        let Some(expected) = load_golden(&path)? else {
            continue;
        };
        let mut npz = NpzReader::new(File::open(&path)?)?;
        let query_mask: Array2<bool> = npz.by_name("query_mask")?;
        let actual = model.infer(MarginalInput {
            text_states: npz.by_name("text_states")?,
            text_mask: npz.by_name("text_mask")?,
            query_states: npz.by_name("query_states")?,
            query_mask: query_mask.clone(),
        })?;
        let pool = build_shared_candidate_pool(
            actual.boundary_mask.index_axis(Axis(0), 0),
            query_mask.index_axis(Axis(0), 0),
            actual.start_logits.index_axis(Axis(0), 0),
            actual.end_logits.index_axis(Axis(0), 0),
            actual.start_all.index_axis(Axis(0), 0),
            actual.end_all.index_axis(Axis(0), 0),
            PoolConfig::default(),
        )?;
        let case = path.file_stem().unwrap().to_str().unwrap();
        assert_discrete_exact(case, &pool, &expected.indices, &expected.mask);
        for (name, actual, expected) in [
            ("compat", &pool.compat_logits, &expected.compat_logits),
            ("proposal", &pool.proposal_logits, &expected.proposal_logits),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 1e-4 + 1e-3 * expected.abs(),
                    "{case}/{name}[{index}]: {actual} != {expected}"
                );
            }
        }
        eprintln!("{case}: ORT -> Rust pool selection exact");
        cases += 1;
    }
    assert_eq!(cases, if full { 24 } else { 2 });
    Ok(())
}

fn npz_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory).with_context(|| directory.display().to_string())? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "npz") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn run_real_fixture(path: &Path) -> Result<bool> {
    let Some(golden) = load_golden(path)? else {
        return Ok(false);
    };
    let case = path
        .file_stem()
        .and_then(|name| name.to_str())
        .context("fixture filename is not UTF-8")?;
    let actual = build_shared_candidate_pool(
        golden.boundary_mask.view(),
        golden.query_mask.view(),
        golden.start_logits.view(),
        golden.end_logits.view(),
        golden.start_projection.view(),
        golden.end_projection.view(),
        PoolConfig::default(),
    )?;
    assert_discrete_exact(case, &actual, &golden.indices, &golden.mask);
    assert_float_exact(
        case,
        "compat_logits",
        &actual.compat_logits,
        &golden.compat_logits,
    );
    assert_float_exact(
        case,
        "proposal_logits",
        &actual.proposal_logits,
        &golden.proposal_logits,
    );
    Ok(true)
}

#[test]
fn committed_real_subset_matches_exactly() -> Result<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1-subset");
    ensure!(
        directory.is_dir(),
        "required model-free pool fixture subset is missing: {}",
        directory.display()
    );
    let mut applicable = 0;
    for path in npz_paths(&directory)? {
        applicable += usize::from(run_real_fixture(&path)?);
    }
    assert_eq!(
        applicable, 2,
        "committed subset must contain exactly two extraction pool fixtures"
    );
    Ok(())
}

#[test]
fn optional_full_real_directory_matches_all_24_exactly() -> Result<()> {
    if std::env::var("GLINER2_POOL_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!(
            "SKIP: set GLINER2_POOL_FULL_FIXTURES=1 for strict 24-case full pool oracle validation"
        );
        return Ok(());
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1");
    ensure!(
        directory.is_dir(),
        "strict full pool fixture directory is missing: {}",
        directory.display()
    );
    let mut applicable = 0;
    for path in npz_paths(&directory)? {
        applicable += usize::from(run_real_fixture(&path)?);
    }
    assert_eq!(
        applicable, 24,
        "full oracle must contain 24 applicable pool cases (30 corpus cases minus five classification cases and one empty-schema case)"
    );
    Ok(())
}
