use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array1, Array3};
use ndarray_npy::NpzReader;

use gliner2_rs::boundary::relations::{RelationInput, RelationModel, RelationOutput};

#[allow(dead_code)]
mod common;
use common::model_root;

const BASE_HIDDEN: usize = 768;

fn synthetic_input(
    batch: usize,
    length: usize,
    relations: usize,
    pairs: usize,
    hidden: usize,
) -> RelationInput {
    let mut batch_index = Array1::zeros(pairs);
    let mut relation_index = Array1::zeros(pairs);
    let mut head_start = Array1::zeros(pairs);
    let mut head_end = Array1::ones(pairs);
    let mut tail_start = Array1::zeros(pairs);
    let mut tail_end = Array1::ones(pairs);
    for pair in 0..pairs {
        batch_index[pair] = if batch == 0 { 0 } else { (pair % batch) as i64 };
        relation_index[pair] = if relations == 0 {
            0
        } else {
            (pair % relations) as i64
        };
        if length > 0 {
            let hs = pair % length;
            let ts = (length - 1).saturating_sub(pair % length);
            head_start[pair] = hs as i64;
            head_end[pair] = (hs + 1) as i64;
            tail_start[pair] = ts as i64;
            tail_end[pair] = (ts + 1) as i64;
        }
    }
    RelationInput {
        text_states: Array3::zeros((batch, length, hidden)),
        relation_query_states: Array3::zeros((batch, relations, hidden.saturating_mul(2))),
        batch_index,
        relation_index,
        head_start,
        head_end,
        tail_start,
        tail_end,
        pair_mask: Array1::from_elem(pairs, true),
    }
}

fn validation_error(input: &RelationInput) -> String {
    input.validate().unwrap_err().to_string()
}

#[test]
fn model_free_validation_accepts_supported_dynamic_shapes() {
    synthetic_input(1, 1, 1, 1, 384).validate().unwrap();
    synthetic_input(2, 7, 3, 65, 768).validate().unwrap();
    synthetic_input(2, 129, 3, 97, 768).validate().unwrap();

    let mut masked = synthetic_input(1, 7, 1, 2, 384);
    masked.pair_mask.fill(false);
    masked
        .validate()
        .expect("false pair masks with valid coordinates remain supported");
}

#[test]
fn zero_mismatch_nonfinite_routing_and_coordinate_inputs_are_rejected() {
    assert!(validation_error(&synthetic_input(0, 1, 1, 1, 384)).contains("B=0"));
    assert!(validation_error(&synthetic_input(1, 0, 1, 1, 384)).contains("L=0"));
    assert!(validation_error(&synthetic_input(1, 1, 0, 1, 384)).contains("R=0"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 0, 384)).contains("P=0"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 1, 0)).contains("H=0"));

    let mut wrong_batch = synthetic_input(2, 3, 2, 2, 384);
    wrong_batch.relation_query_states = Array3::zeros((1, 2, 768));
    assert!(validation_error(&wrong_batch).contains("batch mismatch"));

    let mut wrong_width = synthetic_input(1, 3, 2, 2, 384);
    wrong_width.relation_query_states = Array3::zeros((1, 2, 767));
    assert!(validation_error(&wrong_width).contains("must equal 2H"));

    let mut wrong_length = synthetic_input(1, 3, 2, 2, 384);
    wrong_length.tail_end = Array1::ones(1);
    assert!(validation_error(&wrong_length).contains("tail_end length"));

    let mut non_finite_text = synthetic_input(1, 3, 1, 1, 384);
    non_finite_text.text_states[(0, 0, 0)] = f32::NAN;
    assert!(validation_error(&non_finite_text).contains("non-finite"));
    let mut non_finite_relation = synthetic_input(1, 3, 1, 1, 384);
    non_finite_relation.relation_query_states[(0, 0, 0)] = f32::INFINITY;
    assert!(validation_error(&non_finite_relation).contains("non-finite"));

    let mut bad_batch = synthetic_input(1, 3, 1, 1, 384);
    bad_batch.batch_index[0] = 1;
    assert!(validation_error(&bad_batch).contains("batch_index"));
    let mut negative_batch = synthetic_input(1, 3, 1, 1, 384);
    negative_batch.batch_index[0] = -1;
    assert!(validation_error(&negative_batch).contains("batch_index"));
    let mut bad_relation = synthetic_input(1, 3, 1, 1, 384);
    bad_relation.relation_index[0] = i64::MAX;
    assert!(validation_error(&bad_relation).contains("relation_index"));

    let invalid_spans = [(-1, 1), (1, 1), (2, 1), (0, 4), (i64::MIN, i64::MAX)];
    for (start, end) in invalid_spans {
        let mut bad_head = synthetic_input(1, 3, 1, 1, 384);
        bad_head.head_start[0] = start;
        bad_head.head_end[0] = end;
        bad_head.pair_mask[0] = false;
        assert!(
            validation_error(&bad_head).contains("head coordinates"),
            "masked invalid coordinates must fail closed"
        );

        let mut bad_tail = synthetic_input(1, 3, 1, 1, 384);
        bad_tail.tail_start[0] = start;
        bad_tail.tail_end[0] = end;
        assert!(validation_error(&bad_tail).contains("tail coordinates"));
    }
}

fn relation_graph() -> Result<Option<PathBuf>> {
    let path = model_root().join("onnx/gliner2.5-base-v1/boundary_relations.onnx");
    if path.is_file() {
        return Ok(Some(path));
    }
    let message = format!("missing boundary relation graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn assert_output(output: &RelationOutput, mask: &Array1<bool>) {
    assert_eq!(output.relation_logits.shape(), mask.shape());
    assert!(output.relation_logits.iter().all(|value| value.is_finite()));
    for (pair, (&enabled, &logit)) in mask.iter().zip(output.relation_logits.iter()).enumerate() {
        if !enabled {
            assert_eq!(
                logit.to_bits(),
                0.0_f32.to_bits(),
                "masked relation logit {pair} is not exact positive zero"
            );
        }
    }
}

#[test]
fn actual_graph_supports_dynamic_b_l_r_and_p_with_masked_pairs() -> Result<()> {
    let Some(graph) = relation_graph()? else {
        return Ok(());
    };
    let model = RelationModel::new(graph)?;
    for (batch, length, relations, pairs) in
        [(1, 1, 1, 1), (1, 7, 3, 9), (2, 31, 1, 67), (2, 129, 3, 97)]
    {
        let mut input = synthetic_input(batch, length, relations, pairs, BASE_HIDDEN);
        if pairs > 1 {
            input.pair_mask[pairs - 1] = false;
        }
        let mask = input.pair_mask.clone();
        let output = model.infer(&input)?;
        assert_output(&output, &mask);
    }
    Ok(())
}

struct GoldenCase {
    input: RelationInput,
    expected: RelationOutput,
}

fn load_golden(path: &Path) -> Result<GoldenCase> {
    let mut archive =
        NpzReader::new(File::open(path).with_context(|| path.display().to_string())?)?;
    Ok(GoldenCase {
        input: RelationInput {
            text_states: archive.by_name("text_states")?,
            relation_query_states: archive.by_name("relation_query_states")?,
            batch_index: archive.by_name("batch_index")?,
            relation_index: archive.by_name("relation_index")?,
            head_start: archive.by_name("head_start")?,
            head_end: archive.by_name("head_end")?,
            tail_start: archive.by_name("tail_start")?,
            tail_end: archive.by_name("tail_end")?,
            pair_mask: archive.by_name("pair_mask")?,
        },
        expected: RelationOutput {
            relation_logits: archive.by_name("relation_logits")?,
        },
    })
}

fn assert_close(actual: &Array1<f32>, expected: &Array1<f32>) {
    assert_eq!(actual.shape(), expected.shape());
    let mut compared = 0;
    let mut max_confidence_error = 0.0_f32;
    for (pair, (&observed, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            observed.is_finite(),
            "relation_logits[{pair}] is non-finite"
        );
        assert!(
            reference.is_finite(),
            "golden relation_logits[{pair}] is non-finite"
        );
        let difference = (observed - reference).abs();
        let tolerance = 1e-4 + 1e-3 * reference.abs();
        assert!(
            difference <= tolerance,
            "relation_logits[{pair}]: actual={observed}, golden={reference}, abs={difference}, tolerance={tolerance}"
        );
        let observed_confidence = 1.0 / (1.0 + (-observed).exp());
        let reference_confidence = 1.0 / (1.0 + (-reference).exp());
        max_confidence_error =
            max_confidence_error.max((observed_confidence - reference_confidence).abs());
        compared += 1;
    }
    assert!(compared > 0, "relation comparison was empty");
    assert!(
        max_confidence_error <= 1e-3,
        "relation confidence max error {max_confidence_error} exceeds 1e-3"
    );
}

fn relation_fixture_dir() -> PathBuf {
    env::var_os("GLINER2_BOUNDARY_RELATION_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_root().join("fixtures/gliner2.5-base-v1/relation-runtime-vectors"))
}

#[test]
fn actual_public_wrapper_matches_four_untouched_relation_vectors() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!("SKIP: set GLINER2_BOUNDARY_FULL_FIXTURES=1 for relation parity vectors");
        return Ok(());
    }
    let Some(graph) = relation_graph()? else {
        return Err(anyhow!(
            "GLINER2_BOUNDARY_FULL_FIXTURES=1 requires boundary_relations.onnx"
        ));
    };
    let directory = relation_fixture_dir();
    ensure!(
        directory.is_dir(),
        "missing relation fixtures: {}",
        directory.display()
    );
    ensure!(
        directory.join("manifest.json").is_file(),
        "missing relation fixture manifest"
    );

    let mut paths = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "npz") {
            paths.push(path);
        }
    }
    paths.sort();
    let expected_names = [
        "relation_employment.npz",
        "relation_founded.npz",
        "relation_location.npz",
        "relation_multiple_types.npz",
    ];
    let actual_names: Vec<_> = paths
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
        })
        .collect();
    ensure!(
        actual_names == expected_names,
        "relation fixture IDs differ: {actual_names:?} != {expected_names:?}"
    );

    let model = RelationModel::new(graph)?;
    for path in paths {
        let case = load_golden(&path)?;
        let mask = case.input.pair_mask.clone();
        let actual = model.infer(&case.input)?;
        assert_close(&actual.relation_logits, &case.expected.relation_logits);
        assert_output(&actual, &mask);
    }
    Ok(())
}
