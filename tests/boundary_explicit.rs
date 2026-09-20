use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array2, Array3, Array4};
use ndarray_npy::NpzReader;

use gliner2_rs::boundary::explicit::{ExplicitInput, ExplicitModel, ExplicitOutput};

#[allow(dead_code)]
mod common;
use common::model_root;

const MASK_LOGIT: f32 = -10_000.0;
const BASE_HIDDEN: usize = 768;

fn synthetic_input(
    batch: usize,
    length: usize,
    queries: usize,
    candidates: usize,
    hidden: usize,
) -> ExplicitInput {
    let boundaries = length.saturating_add(1);
    let mut indices = Array4::zeros((batch, queries, candidates, 2));
    if length > 0 {
        for batch_index in 0..batch {
            for query in 0..queries {
                for candidate in 0..candidates {
                    let start = candidate % length;
                    indices[(batch_index, query, candidate, 0)] = start as i64;
                    indices[(batch_index, query, candidate, 1)] = (start + 1) as i64;
                }
            }
        }
    }
    ExplicitInput {
        boundary_states: Array3::zeros((batch, boundaries, 128)),
        text_states: Array3::zeros((batch, length, hidden)),
        text_mask: Array2::from_elem((batch, length), true),
        query_states: Array3::zeros((batch, queries, hidden)),
        query_mask: Array2::from_elem((batch, queries), true),
        start_logits: Array3::zeros((batch, queries, boundaries)),
        end_logits: Array3::zeros((batch, queries, boundaries)),
        inside_prefix: Array3::zeros((batch, queries, boundaries)),
        inside_prefix_mean: Array3::zeros((batch, queries, 1)),
        candidate_indices: indices,
        candidate_mask: Array3::from_elem((batch, queries, candidates), true),
    }
}

fn validation_error(input: &ExplicitInput) -> String {
    input.validate().unwrap_err().to_string()
}

#[test]
fn model_free_validation_accepts_dynamic_shapes_and_illegal_coordinates() {
    synthetic_input(1, 2, 1, 1, 384).validate().unwrap();
    synthetic_input(2, 17, 3, 7, 768).validate().unwrap();
    synthetic_input(1, 4097, 2, 3, 1)
        .validate()
        .expect("choice prefixes may take explicit scorer L beyond 4096");

    let mut invalid = synthetic_input(2, 4, 2, 8, 384);
    invalid.candidate_indices[(0, 0, 0, 0)] = -1;
    invalid.candidate_indices[(0, 0, 1, 0)] = 3;
    invalid.candidate_indices[(0, 0, 1, 1)] = 2;
    invalid.candidate_indices[(0, 0, 2, 1)] = 99;
    invalid.candidate_indices[(0, 0, 3, 0)] = i64::MIN;
    invalid.candidate_indices[(0, 0, 3, 1)] = i64::MAX;
    invalid.candidate_indices[(1, 1, 4, 0)] = i64::MAX;
    invalid.candidate_indices[(1, 1, 4, 1)] = i64::MIN;
    invalid.candidate_mask[(0, 0, 5)] = false;
    invalid.query_mask[(1, 1)] = false;
    invalid
        .validate()
        .expect("illegal candidate coordinates are graph inputs, not validation errors");
}

#[test]
fn malformed_and_zero_dimension_inputs_are_rejected_before_ort() {
    assert!(validation_error(&synthetic_input(0, 2, 1, 1, 384)).contains("B=0"));
    assert!(validation_error(&synthetic_input(1, 0, 1, 1, 384)).contains("L=0"));
    assert!(validation_error(&synthetic_input(1, 2, 0, 1, 384)).contains("Q=0"));
    assert!(validation_error(&synthetic_input(1, 2, 1, 0, 384)).contains("C=0"));
    assert!(validation_error(&synthetic_input(1, 2, 1, 1, 0)).contains("H=0"));

    let mut wrong_boundaries = synthetic_input(1, 2, 1, 1, 384);
    wrong_boundaries.boundary_states = Array3::zeros((1, 2, 128));
    assert!(validation_error(&wrong_boundaries).contains("must equal L+1"));

    let mut wrong_boundary_width = synthetic_input(1, 2, 1, 1, 384);
    wrong_boundary_width.boundary_states = Array3::zeros((1, 3, 127));
    assert!(validation_error(&wrong_boundary_width).contains("width must be 128"));

    let mut wrong_hidden = synthetic_input(1, 2, 1, 1, 384);
    wrong_hidden.query_states = Array3::zeros((1, 1, 768));
    assert!(validation_error(&wrong_hidden).contains("hidden-width mismatch"));

    let mut wrong_batch = synthetic_input(1, 2, 1, 1, 384);
    wrong_batch.text_states = Array3::zeros((2, 2, 384));
    assert!(validation_error(&wrong_batch).contains("batch mismatch"));

    let mut wrong_candidate_queries = synthetic_input(1, 2, 1, 1, 384);
    wrong_candidate_queries.candidate_indices = Array4::zeros((1, 2, 1, 2));
    assert!(validation_error(&wrong_candidate_queries).contains("candidate/query count mismatch"));

    let mut wrong_index_width = synthetic_input(1, 2, 1, 1, 384);
    wrong_index_width.candidate_indices = Array4::zeros((1, 1, 1, 3));
    assert!(validation_error(&wrong_index_width).contains("[B,Q,C,2]"));

    let mut wrong_mask = synthetic_input(1, 2, 1, 2, 384);
    wrong_mask.candidate_mask = Array3::from_elem((1, 1, 1), true);
    assert!(validation_error(&wrong_mask).contains("candidate_mask shape"));

    let mut wrong_logits = synthetic_input(1, 2, 1, 1, 384);
    wrong_logits.inside_prefix_mean = Array3::zeros((1, 1, 2));
    assert!(validation_error(&wrong_logits).contains("inside_prefix_mean shape"));

    for field in 0..7 {
        let mut non_finite = synthetic_input(1, 2, 1, 1, 384);
        match field {
            0 => non_finite.boundary_states[(0, 0, 0)] = f32::NAN,
            1 => non_finite.text_states[(0, 0, 0)] = f32::INFINITY,
            2 => non_finite.query_states[(0, 0, 0)] = f32::NEG_INFINITY,
            3 => non_finite.start_logits[(0, 0, 0)] = f32::NAN,
            4 => non_finite.end_logits[(0, 0, 0)] = f32::INFINITY,
            5 => non_finite.inside_prefix[(0, 0, 0)] = f32::NEG_INFINITY,
            6 => non_finite.inside_prefix_mean[(0, 0, 0)] = f32::NAN,
            _ => unreachable!(),
        }
        assert!(validation_error(&non_finite).contains("non-finite"));
    }
}

fn explicit_graph() -> Result<Option<PathBuf>> {
    let path = model_root().join("onnx/gliner2.5-base-v1/boundary_explicit_scorer.onnx");
    if path.is_file() {
        return Ok(Some(path));
    }
    let message = format!("missing boundary explicit scorer graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn invalid_input(candidates: usize) -> ExplicitInput {
    let mut input = synthetic_input(2, 6, 3, candidates, BASE_HIDDEN);
    input.text_mask.slice_mut(ndarray::s![1, 4..]).fill(false);
    input.query_mask[(0, 2)] = false;
    input.query_mask[(1, 1)] = false;

    let patterns = [
        [0, 1],
        [-1, 1],
        [3, 2],
        [2, 2],
        [0, 99],
        [i64::MIN, i64::MAX],
        [i64::MAX, i64::MIN],
        [1, 4],
        [0, 2],
    ];
    for batch in 0..2 {
        for query in 0..3 {
            for candidate in 0..candidates {
                let [start, end] = patterns[candidate % patterns.len()];
                input.candidate_indices[(batch, query, candidate, 0)] = start;
                input.candidate_indices[(batch, query, candidate, 1)] = end;
            }
        }
    }
    if candidates > 1 {
        input
            .candidate_mask
            .slice_mut(ndarray::s![.., .., 1])
            .fill(false);
    }
    input
}

fn assert_invalid_contract(output: &ExplicitOutput) {
    assert!(output.pair_logits.iter().all(|value| value.is_finite()));
    assert!(output.compatibility.iter().all(|value| value.is_finite()));
    for (index, &legal) in output.legal_mask.indexed_iter() {
        if !legal {
            assert_eq!(output.compatibility[index].to_bits(), 0.0_f32.to_bits());
            assert_eq!(output.pair_logits[index].to_bits(), MASK_LOGIT.to_bits());
        }
    }
}

#[test]
fn actual_graph_supports_batch_two_variable_candidate_counts_and_i64_extremes() -> Result<()> {
    let Some(graph) = explicit_graph()? else {
        return Ok(());
    };
    let model = ExplicitModel::new(graph)?;

    let single = model.infer(invalid_input(1))?;
    assert_eq!(single.pair_logits.shape(), [2, 3, 1]);
    assert_invalid_contract(&single);

    let varied = model.infer(invalid_input(9))?;
    assert_eq!(varied.pair_logits.shape(), [2, 3, 9]);
    assert_eq!(varied.compatibility.shape(), [2, 3, 9]);
    assert_eq!(varied.legal_mask.shape(), [2, 3, 9]);
    assert_invalid_contract(&varied);
    assert!(varied.legal_mask.iter().any(|value| *value));
    assert!(varied.legal_mask.iter().any(|value| !*value));
    Ok(())
}

struct GoldenCase {
    input: ExplicitInput,
    expected: ExplicitOutput,
}

fn load_golden(path: &Path) -> Result<GoldenCase> {
    let mut archive =
        NpzReader::new(File::open(path).with_context(|| path.display().to_string())?)?;
    Ok(GoldenCase {
        input: ExplicitInput {
            boundary_states: archive.by_name("boundary_states")?,
            text_states: archive.by_name("text_states")?,
            text_mask: archive.by_name("text_mask")?,
            query_states: archive.by_name("query_states")?,
            query_mask: archive.by_name("query_mask")?,
            start_logits: archive.by_name("start_logits")?,
            end_logits: archive.by_name("end_logits")?,
            inside_prefix: archive.by_name("inside_prefix")?,
            inside_prefix_mean: archive.by_name("inside_prefix_mean")?,
            candidate_indices: archive.by_name("candidate_indices")?,
            candidate_mask: archive.by_name("candidate_mask")?,
        },
        expected: ExplicitOutput {
            pair_logits: archive.by_name("pair_logits")?,
            compatibility: archive.by_name("compatibility")?,
            legal_mask: archive.by_name("legal_mask")?,
        },
    })
}

fn assert_close(name: &str, actual: &Array3<f32>, expected: &Array3<f32>) {
    assert_eq!(actual.shape(), expected.shape(), "{name}: shape mismatch");
    let mut count = 0;
    for (index, (&observed, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(observed.is_finite(), "{name}[{index}] is non-finite");
        assert!(
            reference.is_finite(),
            "golden {name}[{index}] is non-finite"
        );
        let difference = (observed - reference).abs();
        let tolerance = 1e-4 + 1e-3 * reference.abs();
        assert!(
            difference <= tolerance,
            "{name}[{index}]: actual={observed}, golden={reference}, abs={difference}, tolerance={tolerance}"
        );
        count += 1;
    }
    assert!(count > 0, "{name} comparison was empty");
}

#[test]
fn actual_graph_matches_untouched_upstream_explicit_vectors() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!("SKIP: set GLINER2_BOUNDARY_FULL_FIXTURES=1 for explicit parity vectors");
        return Ok(());
    }
    let Some(graph) = explicit_graph()? else {
        return Err(anyhow!(
            "GLINER2_BOUNDARY_FULL_FIXTURES=1 requires boundary_explicit_scorer.onnx"
        ));
    };
    let directory = model_root().join("fixtures/gliner2.5-explicit");
    ensure!(
        directory.is_dir(),
        "missing fixtures: {}",
        directory.display()
    );
    ensure!(
        directory.join("manifest.json").is_file(),
        "missing explicit fixture manifest"
    );

    let mut paths = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "npz") {
            paths.push(path);
        }
    }
    paths.sort();
    ensure!(
        paths.len() == 4,
        "expected 4 explicit vectors, got {}",
        paths.len()
    );

    let model = ExplicitModel::new(graph)?;
    for path in paths {
        let case = load_golden(&path)?;
        let actual = model.infer(case.input)?;
        assert_close(
            "pair_logits",
            &actual.pair_logits,
            &case.expected.pair_logits,
        );
        assert_close(
            "compatibility",
            &actual.compatibility,
            &case.expected.compatibility,
        );
        assert_eq!(actual.legal_mask, case.expected.legal_mask);
        assert_invalid_contract(&actual);
    }
    Ok(())
}
