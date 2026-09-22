use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzReader;

use gliner2_rs::boundary::records::{RecordInput, RecordModel, RecordOutput};

#[allow(dead_code)]
mod common;
use common::model_root;

const MASK_LOGIT: f32 = -10_000.0;
const HIDDEN: usize = 768;

fn synthetic_input(
    fields: usize,
    candidates: usize,
    context: usize,
    seeds: usize,
    hidden: usize,
    mode: i64,
) -> RecordInput {
    RecordInput {
        field_query_states: Array2::zeros((fields, hidden)),
        field_candidate_states: Array3::zeros((fields, candidates, hidden)),
        field_candidate_mask: Array2::from_elem((fields, candidates), true),
        context_states: Array2::zeros((context, hidden)),
        context_mask: Array1::from_elem(context, true),
        seed_states: Array2::zeros((seeds, hidden)),
        seed_object_logits: Array1::from_iter((0..seeds).map(|index| index as f32 / 10.0)),
        mode,
    }
}

fn validation_error(input: &RecordInput) -> String {
    input.validate().unwrap_err().to_string()
}

#[test]
fn model_free_validation_accepts_supported_dynamic_shapes() {
    synthetic_input(3, 7, 11, 153, 384, 0).validate().unwrap();
    synthetic_input(2, 1, 1, 408, 768, 1).validate().unwrap();

    let mut empty_masks = synthetic_input(4, 5, 1, 1, 768, 2);
    empty_masks.field_candidate_mask.fill(false);
    empty_masks.context_mask.fill(false);
    empty_masks
        .validate()
        .expect("all-false masks with positive padded dimensions are valid");
}

#[test]
fn malformed_inputs_and_zero_dimension_calls_are_rejected_before_ort() {
    assert!(validation_error(&synthetic_input(0, 1, 1, 1, 768, 0)).contains("F=0"));
    assert!(validation_error(&synthetic_input(1, 0, 1, 1, 768, 0)).contains("C=0"));
    assert!(validation_error(&synthetic_input(1, 1, 0, 1, 768, 2)).contains("M=0"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 0, 768, 0)).contains("Ni=0"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 0, 768, 1)).contains("Ni=0"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 0, 768, 2)).contains("dummy seed"));
    assert!(validation_error(&synthetic_input(1, 1, 1, 1, 0, 0)).contains("H=0"));

    for mode in [-1, 3, i64::MAX] {
        assert!(validation_error(&synthetic_input(1, 1, 1, 1, 384, mode)).contains("record mode"));
    }

    let mut wrong_fields = synthetic_input(2, 3, 4, 5, 384, 0);
    wrong_fields.field_candidate_states = Array3::zeros((3, 3, 384));
    assert!(validation_error(&wrong_fields).contains("field count mismatch"));

    let mut wrong_hidden = synthetic_input(2, 3, 4, 5, 384, 0);
    wrong_hidden.context_states = Array2::zeros((4, 768));
    assert!(validation_error(&wrong_hidden).contains("hidden-width mismatch"));

    let mut wrong_field_mask = synthetic_input(2, 3, 4, 5, 384, 0);
    wrong_field_mask.field_candidate_mask = Array2::from_elem((2, 2), true);
    assert!(validation_error(&wrong_field_mask).contains("field_candidate_mask shape"));

    let mut wrong_context_mask = synthetic_input(2, 3, 4, 5, 384, 0);
    wrong_context_mask.context_mask = Array1::from_elem(3, true);
    assert!(validation_error(&wrong_context_mask).contains("context_mask shape"));

    let mut wrong_seed_logits = synthetic_input(2, 3, 4, 5, 384, 0);
    wrong_seed_logits.seed_object_logits = Array1::zeros(4);
    assert!(validation_error(&wrong_seed_logits).contains("seed_object_logits shape"));

    for field in 0..5 {
        let mut non_finite = synthetic_input(1, 1, 1, 1, 384, 0);
        match field {
            0 => non_finite.field_query_states[(0, 0)] = f32::NAN,
            1 => non_finite.field_candidate_states[(0, 0, 0)] = f32::INFINITY,
            2 => non_finite.context_states[(0, 0)] = f32::NEG_INFINITY,
            3 => non_finite.seed_states[(0, 0)] = f32::NAN,
            4 => non_finite.seed_object_logits[0] = f32::NAN,
            _ => unreachable!(),
        }
        assert!(validation_error(&non_finite).contains("non-finite"));
    }
}

fn record_graph() -> Result<Option<PathBuf>> {
    let path = model_root().join("onnx/gliner2.5-base-v1/boundary_records.onnx");
    if path.is_file() {
        return Ok(Some(path));
    }
    let message = format!("missing boundary record graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn assert_shape_and_finite(
    output: &RecordOutput,
    fields: usize,
    instances: usize,
    candidates: usize,
    hidden: usize,
) {
    assert_eq!(output.instance_states.shape(), [instances, hidden]);
    assert_eq!(output.object_logits.shape(), [instances]);
    assert_eq!(
        output.assignment_logits.shape(),
        [fields, instances, candidates + 1]
    );
    assert!(output.instance_states.iter().all(|value| value.is_finite()));
    assert!(output.object_logits.iter().all(|value| value.is_finite()));
    assert!(
        output
            .assignment_logits
            .iter()
            .all(|value| value.is_finite())
    );
}

fn assert_masked_columns(output: &RecordOutput, mask: &Array2<bool>) {
    for ((field, candidate), &valid) in mask.indexed_iter() {
        if !valid {
            for instance in 0..output.assignment_logits.shape()[1] {
                assert_eq!(
                    output.assignment_logits[(field, instance, candidate + 1)].to_bits(),
                    MASK_LOGIT.to_bits(),
                    "masked assignment [{field},{instance},{}] is not exact -1e4",
                    candidate + 1
                );
            }
        }
    }
}

#[test]
fn one_actual_graph_runs_all_modes_with_dynamic_instance_counts() -> Result<()> {
    let Some(graph) = record_graph()? else {
        return Ok(());
    };
    let model = RecordModel::new(graph)?;

    let mut natural = synthetic_input(2, 3, 2, 153, HIDDEN, 0);
    natural.field_candidate_mask[(0, 1)] = false;
    natural.field_candidate_mask[(1, 2)] = false;
    let natural_mask = natural.field_candidate_mask.clone();
    let natural_seed_logits = natural.seed_object_logits.clone();
    let natural_output = model.infer(&natural)?;
    assert_shape_and_finite(&natural_output, 2, 153, 3, HIDDEN);
    assert_eq!(natural_output.object_logits, natural_seed_logits);
    assert_masked_columns(&natural_output, &natural_mask);

    let mut latent = synthetic_input(3, 2, 3, 408, HIDDEN, 1);
    latent.field_candidate_mask.fill(false);
    let latent_mask = latent.field_candidate_mask.clone();
    let latent_output = model.infer(&latent)?;
    assert_shape_and_finite(&latent_output, 3, 408, 2, HIDDEN);
    assert_masked_columns(&latent_output, &latent_mask);

    // Empty anchorless context uses one zero row with mask=false; the seed row
    // is a positive-sized dummy because both dimensions are graph ABI guards.
    let mut anchorless = synthetic_input(4, 5, 1, 1, HIDDEN, 2);
    anchorless.context_mask.fill(false);
    anchorless.field_candidate_mask.fill(false);
    let anchorless_mask = anchorless.field_candidate_mask.clone();
    let anchorless_output = model.infer(&anchorless)?;
    assert_shape_and_finite(&anchorless_output, 4, 32, 5, HIDDEN);
    assert_masked_columns(&anchorless_output, &anchorless_mask);
    Ok(())
}

struct GoldenCase {
    input: RecordInput,
    expected: RecordOutput,
}

fn load_golden(path: &Path) -> Result<GoldenCase> {
    let mut archive =
        NpzReader::new(File::open(path).with_context(|| path.display().to_string())?)?;
    // The deterministic NPZ writer stores scalar ndarrays as one-element
    // contiguous arrays; the runtime ABI itself remains a scalar tensor.
    let mode: Array1<i64> = archive.by_name("mode")?;
    ensure!(mode.len() == 1, "{} has non-scalar mode", path.display());
    Ok(GoldenCase {
        input: RecordInput {
            field_query_states: archive.by_name("field_query_states")?,
            field_candidate_states: archive.by_name("field_candidate_states")?,
            field_candidate_mask: archive.by_name("field_candidate_mask")?,
            context_states: archive.by_name("context_states")?,
            context_mask: archive.by_name("context_mask")?,
            seed_states: archive.by_name("seed_states")?,
            seed_object_logits: archive.by_name("seed_object_logits")?,
            mode: mode[0],
        },
        expected: RecordOutput {
            instance_states: archive.by_name("instance_states")?,
            object_logits: archive.by_name("object_logits")?,
            assignment_logits: archive.by_name("assignment_logits")?,
        },
    })
}

fn assert_close(
    name: &str,
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
) {
    let mut count = 0;
    for (index, (observed, reference)) in actual.zip(expected).enumerate() {
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

fn compare_output(actual: &RecordOutput, expected: &RecordOutput) {
    assert_eq!(
        actual.instance_states.shape(),
        expected.instance_states.shape()
    );
    assert_eq!(actual.object_logits.shape(), expected.object_logits.shape());
    assert_eq!(
        actual.assignment_logits.shape(),
        expected.assignment_logits.shape()
    );
    assert_close(
        "instance_states",
        actual.instance_states.iter().copied(),
        expected.instance_states.iter().copied(),
    );
    assert_close(
        "object_logits",
        actual.object_logits.iter().copied(),
        expected.object_logits.iter().copied(),
    );
    assert_close(
        "assignment_logits",
        actual.assignment_logits.iter().copied(),
        expected.assignment_logits.iter().copied(),
    );
}

#[test]
fn actual_graph_matches_four_untouched_forward_group_vectors() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!("SKIP: set GLINER2_BOUNDARY_FULL_FIXTURES=1 for record parity vectors");
        return Ok(());
    }
    let Some(graph) = record_graph()? else {
        return Err(anyhow!(
            "GLINER2_BOUNDARY_FULL_FIXTURES=1 requires boundary_records.onnx"
        ));
    };
    let directory = model_root().join("fixtures/gliner2.5-records");
    ensure!(
        directory.is_dir(),
        "missing record fixtures: {}",
        directory.display()
    );
    ensure!(
        directory.join("manifest.json").is_file(),
        "missing record fixture manifest"
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
        "json_anchorless_list.npz",
        "json_latent_products.npz",
        "json_natural_choice.npz",
        "json_natural_people.npz",
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
        "record fixture IDs differ: {actual_names:?} != {expected_names:?}"
    );

    let model = RecordModel::new(graph)?;
    for path in paths {
        let case = load_golden(&path)?;
        let mask = case.input.field_candidate_mask.clone();
        let actual = model.infer(&case.input)?;
        compare_output(&actual, &case.expected);
        assert_masked_columns(&actual, &mask);
    }
    Ok(())
}
