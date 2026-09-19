use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::boundary::{
    marginals::{MarginalInput, MarginalModel},
    pool::{PoolConfig, build_shared_candidate_pool},
    scorer::{ScorerInput, ScorerModel, ScorerOutput},
};
use ndarray::{Array2, Array3, Array4, Axis};
use ndarray_npy::NpzReader;

#[allow(dead_code)]
mod common;
use common::model_root;

const MASK_LOGIT: f32 = -10_000.0;

struct GoldenCase {
    input: ScorerInput,
    expected: ScorerOutput,
}

fn fixture_directory() -> PathBuf {
    let full = env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1");
    Path::new(env!("CARGO_MANIFEST_DIR")).join(if full {
        "fixtures/gliner2.5-base-v1"
    } else {
        "fixtures/gliner2.5-base-v1-subset"
    })
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

fn has_name(names: &[String], expected: &str) -> bool {
    names
        .iter()
        .any(|name| name.trim_end_matches(".npy") == expected)
}

fn load_golden(path: &Path) -> Result<Option<GoldenCase>> {
    let mut npz = NpzReader::new(File::open(path).with_context(|| path.display().to_string())?)?;
    let names = npz.names()?;
    if !has_name(&names, "shared_scorer_0_pair_logits_candidate_major") {
        return Ok(None);
    }
    for name in [
        "shared_scorer_0_boundary_states",
        "marginals_0_text_states",
        "marginals_0_text_mask",
        "shared_scorer_0_query_states",
        "shared_scorer_0_query_mask",
        "marginals_0_start_logits",
        "marginals_0_end_logits",
        "marginals_0_inside_prefix",
        "marginals_0_inside_prefix_mean",
        "shared_scorer_0_indices",
        "shared_scorer_0_mask",
        "shared_scorer_0_compat",
        "boundary_head_0_pair_logits",
        "boundary_head_0_candidate_states",
        "boundary_head_0_null_logits",
        "boundary_head_0_count_log_rates",
    ] {
        ensure!(
            has_name(&names, name),
            "{} is missing {name}",
            path.display()
        );
    }

    let expanded_states: Array4<f32> = npz.by_name("boundary_head_0_candidate_states")?;
    ensure!(
        expanded_states.shape()[1] > 0,
        "{} has no extraction queries",
        path.display()
    );
    let candidate_states = expanded_states.index_axis(Axis(1), 0).to_owned();
    for query in 1..expanded_states.shape()[1] {
        ensure!(
            expanded_states.index_axis(Axis(1), query) == candidate_states.view(),
            "{} has query-dependent shared candidate states",
            path.display()
        );
    }

    let candidate_major: Array3<f32> =
        npz.by_name("shared_scorer_0_pair_logits_candidate_major")?;
    let pair_logits: Array3<f32> = npz.by_name("boundary_head_0_pair_logits")?;
    ensure!(
        candidate_major.view().permuted_axes([0, 2, 1]) == pair_logits.view(),
        "{} has inconsistent saved pair-logit order",
        path.display()
    );

    Ok(Some(GoldenCase {
        input: ScorerInput {
            boundary_states: npz.by_name("shared_scorer_0_boundary_states")?,
            text_states: npz.by_name("marginals_0_text_states")?,
            text_mask: npz.by_name("marginals_0_text_mask")?,
            query_states: npz.by_name("shared_scorer_0_query_states")?,
            query_mask: npz.by_name("shared_scorer_0_query_mask")?,
            start_logits: npz.by_name("marginals_0_start_logits")?,
            end_logits: npz.by_name("marginals_0_end_logits")?,
            inside_prefix: npz.by_name("marginals_0_inside_prefix")?,
            inside_prefix_mean: npz.by_name("marginals_0_inside_prefix_mean")?,
            candidate_indices: npz.by_name("shared_scorer_0_indices")?,
            candidate_mask: npz.by_name("shared_scorer_0_mask")?,
            candidate_compat: npz.by_name("shared_scorer_0_compat")?,
        },
        expected: ScorerOutput {
            pair_logits,
            candidate_states,
            null_logits: npz.by_name("boundary_head_0_null_logits")?,
            count_log_rates: npz.by_name("boundary_head_0_count_log_rates")?,
        },
    }))
}

fn scorer_graph() -> Result<Option<PathBuf>> {
    let path = model_root().join("onnx/gliner2.5-base-v1/boundary_scorer.onnx");
    if path.is_file() {
        return Ok(Some(path));
    }
    let message = format!("missing boundary scorer graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn marginal_graph() -> Result<Option<PathBuf>> {
    let path = model_root().join("onnx/gliner2.5-base-v1/boundary_marginals.onnx");
    if path.is_file() {
        return Ok(Some(path));
    }
    let message = format!("missing boundary marginal graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn assert_close_3(name: &str, actual: &Array3<f32>, expected: &Array3<f32>) {
    assert_eq!(actual.shape(), expected.shape(), "{name}: shape mismatch");
    assert_close(name, actual.iter().copied(), expected.iter().copied());
}

fn assert_close_2(name: &str, actual: &Array2<f32>, expected: &Array2<f32>) {
    assert_eq!(actual.shape(), expected.shape(), "{name}: shape mismatch");
    assert_close(name, actual.iter().copied(), expected.iter().copied());
}

fn assert_close(
    name: &str,
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
) {
    let mut max_abs = 0.0_f32;
    for (index, (observed, reference)) in actual.zip(expected).enumerate() {
        assert!(observed.is_finite(), "{name}[{index}] is non-finite");
        assert!(
            reference.is_finite(),
            "golden {name}[{index}] is non-finite"
        );
        let difference = (observed - reference).abs();
        max_abs = max_abs.max(difference);
        let tolerance = 1e-4 + 1e-3 * reference.abs();
        assert!(
            difference <= tolerance,
            "{name}[{index}]: actual={observed}, golden={reference}, abs={difference}, tolerance={tolerance}"
        );
    }
    eprintln!("{name}: max_abs={max_abs:.9}");
}

fn assert_masked_candidate_states_are_exact_zero(
    output: &ScorerOutput,
    candidate_mask: &Array2<bool>,
) {
    for ((batch, candidate), &valid) in candidate_mask.indexed_iter() {
        if !valid {
            assert!(
                output
                    .candidate_states
                    .slice(ndarray::s![batch, candidate, ..])
                    .iter()
                    .all(|value| value.to_bits() == 0),
                "masked candidate state [{batch},{candidate}] is not exact +0"
            );
        }
    }
}

fn assert_masked_pair_logits(
    output: &ScorerOutput,
    query_mask: &Array2<bool>,
    candidate_mask: &Array2<bool>,
) {
    for batch in 0..query_mask.nrows() {
        for query in 0..query_mask.ncols() {
            for candidate in 0..candidate_mask.ncols() {
                if !(query_mask[(batch, query)] && candidate_mask[(batch, candidate)]) {
                    assert_eq!(
                        output.pair_logits[(batch, query, candidate)].to_bits(),
                        MASK_LOGIT.to_bits(),
                        "invalid pair [{batch},{query},{candidate}] is not exact mask logit"
                    );
                }
            }
        }
    }
}

#[test]
fn scorer_matches_saved_shared_head_fixtures() -> Result<()> {
    let Some(graph) = scorer_graph()? else {
        return Ok(());
    };
    let directory = fixture_directory();
    ensure!(
        directory.is_dir(),
        "missing fixtures: {}",
        directory.display()
    );
    let full = env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1");
    let model = ScorerModel::new(graph)?;
    let mut cases = 0;
    for path in npz_paths(&directory)? {
        let Some(case) = load_golden(&path)? else {
            continue;
        };
        let mask = case.input.candidate_mask.clone();
        let query_mask = case.input.query_mask.clone();
        let actual = model.infer(case.input)?;
        assert_close_3(
            "pair_logits",
            &actual.pair_logits,
            &case.expected.pair_logits,
        );
        assert_close_3(
            "candidate_states",
            &actual.candidate_states,
            &case.expected.candidate_states,
        );
        assert_close_2(
            "null_logits",
            &actual.null_logits,
            &case.expected.null_logits,
        );
        assert_close_2(
            "count_log_rates",
            &actual.count_log_rates,
            &case.expected.count_log_rates,
        );
        assert_masked_candidate_states_are_exact_zero(&actual, &mask);
        assert_masked_pair_logits(&actual, &query_mask, &mask);
        cases += 1;
    }
    assert_eq!(cases, if full { 24 } else { 2 });
    Ok(())
}

fn synthetic_input(length: usize, queries: usize, candidates: usize, hidden: usize) -> ScorerInput {
    let boundaries = length.saturating_add(1);
    let mut indices = Array3::zeros((1, candidates, 2));
    if length > 0 {
        for candidate in 0..candidates {
            indices[(0, candidate, 1)] = 1;
        }
    }
    ScorerInput {
        boundary_states: Array3::zeros((1, boundaries, 128)),
        text_states: Array3::zeros((1, length, hidden)),
        text_mask: Array2::from_elem((1, length), true),
        query_states: Array3::zeros((1, queries, hidden)),
        query_mask: Array2::from_elem((1, queries), true),
        start_logits: Array3::zeros((1, queries, boundaries)),
        end_logits: Array3::zeros((1, queries, boundaries)),
        inside_prefix: Array3::zeros((1, queries, boundaries)),
        inside_prefix_mean: Array3::zeros((1, queries, 1)),
        candidate_indices: indices,
        candidate_mask: Array2::from_elem((1, candidates), length > 0),
        candidate_compat: Array2::zeros((1, candidates)),
    }
}

fn validation_error(input: &ScorerInput) -> String {
    input.validate().unwrap_err().to_string()
}

#[test]
fn malformed_inputs_are_rejected_before_ort() {
    synthetic_input(2, 1, 1, 384).validate().unwrap();
    synthetic_input(2, 1, 1, 768).validate().unwrap();
    synthetic_input(4097, 1, 1, 1)
        .validate()
        .expect("choice prefixes may take scorer L beyond 4096");

    let empty_text = synthetic_input(0, 1, 1, 384);
    assert!(validation_error(&empty_text).contains("L=0 inputs must bypass"));
    let empty_queries = synthetic_input(2, 0, 1, 384);
    assert!(validation_error(&empty_queries).contains("Q=0 inputs must bypass"));
    let empty_candidates = synthetic_input(2, 1, 0, 384);
    assert!(validation_error(&empty_candidates).contains("C=0 inputs must bypass"));

    let mut wrong_boundaries = synthetic_input(2, 1, 1, 384);
    wrong_boundaries.boundary_states = Array3::zeros((1, 2, 128));
    assert!(validation_error(&wrong_boundaries).contains("must equal L+1"));

    let mut wrong_width = synthetic_input(2, 1, 1, 384);
    wrong_width.query_states = Array3::zeros((1, 1, 768));
    assert!(validation_error(&wrong_width).contains("hidden-width mismatch"));

    let mut wrong_batch = synthetic_input(2, 1, 1, 384);
    wrong_batch.text_states = Array3::zeros((2, 2, 384));
    assert!(validation_error(&wrong_batch).contains("batch mismatch"));

    let mut wrong_shape = synthetic_input(2, 1, 1, 384);
    wrong_shape.inside_prefix_mean = Array3::zeros((1, 1, 2));
    assert!(validation_error(&wrong_shape).contains("inside_prefix_mean shape"));

    let mut non_finite = synthetic_input(2, 1, 1, 384);
    non_finite.candidate_compat[(0, 0)] = f32::NAN;
    assert!(validation_error(&non_finite).contains("non-finite"));

    let mut padded_out_of_range = synthetic_input(2, 1, 1, 384);
    padded_out_of_range.candidate_mask[(0, 0)] = false;
    padded_out_of_range.candidate_indices[(0, 0, 1)] = 3;
    assert!(validation_error(&padded_out_of_range).contains("outside 0..N"));

    let mut reversed_live_span = synthetic_input(2, 1, 1, 384);
    reversed_live_span.candidate_indices[(0, 0, 0)] = 1;
    reversed_live_span.candidate_indices[(0, 0, 1)] = 1;
    assert!(validation_error(&reversed_live_span).contains("start < end <= L"));

    let mut all_padded = synthetic_input(2, 1, 7, 384);
    all_padded.candidate_mask.fill(false);
    all_padded.candidate_indices.fill(0);
    all_padded
        .validate()
        .expect("an all-false padded pool is valid");
}

#[test]
fn all_false_candidate_pool_runs_without_native_failure() -> Result<()> {
    let Some(graph) = scorer_graph()? else {
        return Ok(());
    };
    let path = fixture_directory().join("unicode_combining_emoji.npz");
    let Some(mut case) = load_golden(&path)? else {
        return Err(anyhow!("{} is not a scorer fixture", path.display()));
    };
    case.input.candidate_mask.fill(false);
    case.input.candidate_indices.fill(0);
    case.input.candidate_compat.fill(0.0);
    let mask = case.input.candidate_mask.clone();
    let query_mask = case.input.query_mask.clone();
    let actual = ScorerModel::new(graph)?.infer(case.input)?;
    assert_masked_candidate_states_are_exact_zero(&actual, &mask);
    assert_masked_pair_logits(&actual, &query_mask, &mask);
    assert!(actual.pair_logits.iter().all(|value| *value == MASK_LOGIT));
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// Frozen encoder states exercise the complete low-level heads path:
/// ORT marginals -> exact Rust candidate pool -> ORT scorer.
#[test]
fn marginal_pool_scorer_heads_match_frozen_end_to_end_outputs() -> Result<()> {
    let (Some(marginal_path), Some(scorer_path)) = (marginal_graph()?, scorer_graph()?) else {
        return Ok(());
    };
    let marginals = MarginalModel::new(marginal_path)?;
    let scorer = ScorerModel::new(scorer_path)?;
    let directory = fixture_directory();
    let full = env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1");
    let mut cases = 0;

    for path in npz_paths(&directory)? {
        let Some(golden) = load_golden(&path)? else {
            continue;
        };
        let GoldenCase {
            input: frozen,
            expected,
        } = golden;
        let text_states = frozen.text_states;
        let text_mask = frozen.text_mask;
        let query_states = frozen.query_states;
        let query_mask = frozen.query_mask;
        let marginal = marginals.infer(MarginalInput {
            text_states: text_states.clone(),
            text_mask: text_mask.clone(),
            query_states: query_states.clone(),
            query_mask: query_mask.clone(),
        })?;
        let pool = build_shared_candidate_pool(
            marginal.boundary_mask.index_axis(Axis(0), 0),
            query_mask.index_axis(Axis(0), 0),
            marginal.start_logits.index_axis(Axis(0), 0),
            marginal.end_logits.index_axis(Axis(0), 0),
            marginal.start_all.index_axis(Axis(0), 0),
            marginal.end_all.index_axis(Axis(0), 0),
            PoolConfig::default(),
        )?;

        let capacity = pool.mask.len();
        let candidate_indices =
            Array3::from_shape_vec((1, capacity, 2), pool.indices.iter().copied().collect())?;
        let candidate_mask = Array2::from_shape_vec((1, capacity), pool.mask.to_vec())?;
        let candidate_compat = Array2::from_shape_vec((1, capacity), pool.compat_logits.to_vec())?;

        // Candidate selection and ordering remain a discrete exact contract.
        assert_eq!(
            candidate_indices,
            frozen.candidate_indices,
            "{} indices",
            path.display()
        );
        assert_eq!(
            candidate_mask,
            frozen.candidate_mask,
            "{} mask",
            path.display()
        );

        let actual = scorer.infer(ScorerInput {
            boundary_states: marginal.boundary_states,
            text_states,
            text_mask,
            query_states,
            query_mask,
            start_logits: marginal.start_logits,
            end_logits: marginal.end_logits,
            inside_prefix: marginal.inside_prefix,
            inside_prefix_mean: marginal.inside_prefix_mean,
            candidate_indices,
            candidate_mask: candidate_mask.clone(),
            candidate_compat,
        })?;
        assert_close_3(
            "e2e pair_logits",
            &actual.pair_logits,
            &expected.pair_logits,
        );
        assert_close_3(
            "e2e candidate_states",
            &actual.candidate_states,
            &expected.candidate_states,
        );
        assert_close_2(
            "e2e null_logits",
            &actual.null_logits,
            &expected.null_logits,
        );
        assert_close_2(
            "e2e count_log_rates",
            &actual.count_log_rates,
            &expected.count_log_rates,
        );
        for (index, (&observed, &reference)) in actual
            .pair_logits
            .iter()
            .zip(expected.pair_logits.iter())
            .enumerate()
        {
            let confidence_difference = (sigmoid(observed) - sigmoid(reference)).abs();
            assert!(
                confidence_difference <= 1e-3,
                "{} confidence[{index}] drift {confidence_difference} exceeds 1e-3",
                path.display()
            );
        }
        assert_masked_candidate_states_are_exact_zero(&actual, &candidate_mask);
        cases += 1;
    }
    assert_eq!(cases, if full { 24 } else { 2 });
    Ok(())
}
