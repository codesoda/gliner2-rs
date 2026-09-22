use std::{env, fs::File, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::boundary::marginals::{MarginalInput, MarginalModel, MarginalOutput};
use ndarray::{Array2, Array3};
use ndarray_npy::NpzReader;

#[allow(dead_code)]
mod common;
use common::model_root;

const FIXTURE_IDS: [&str; 3] = [
    "unicode_combining_emoji",
    "classification_multi_task",
    "relation_employment",
];

struct Fixture {
    input: MarginalInput,
    expected: Option<MarginalOutput>,
}

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1-subset")
}

fn full_fixture_dir() -> std::path::PathBuf {
    model_root().join("fixtures/gliner2.5-base-v1")
}

fn load_fixture(path: &Path) -> Result<Fixture> {
    let file = File::open(path).with_context(|| path.display().to_string())?;
    let mut npz = NpzReader::new(file)?;
    let names = npz.names()?;
    let has = |name: &str| {
        names
            .iter()
            .any(|candidate| candidate.trim_end_matches(".npy") == name)
    };

    for name in ["text_states", "text_mask", "query_states", "query_mask"] {
        ensure!(has(name), "{} is missing {name}", path.display());
    }
    let text_states: Array3<f32> = npz.by_name("text_states")?;
    let text_mask: Array2<bool> = npz.by_name("text_mask")?;
    let query_states: Array3<f32> = npz.by_name("query_states")?;
    let query_mask: Array2<bool> = npz.by_name("query_mask")?;
    ensure!(text_states.shape()[0] == 1, "fixture batch must be one");
    ensure!(
        text_states.shape()[2] == 768,
        "fixture hidden width must be 768"
    );
    ensure!(
        text_mask.shape() == &text_states.shape()[0..2],
        "text mask/state shape mismatch"
    );
    ensure!(
        query_states.shape()[0] == 1 && query_states.shape()[2] == 768,
        "query state shape mismatch"
    );
    ensure!(
        query_mask.shape() == &query_states.shape()[0..2],
        "query mask/state shape mismatch"
    );

    let output_names = [
        "boundary_encoder_0_states",
        "boundary_encoder_0_mask",
        "marginals_0_start_logits",
        "marginals_0_end_logits",
        "marginals_0_inside_logits",
        "marginals_0_inside_prefix",
        "marginals_0_inside_prefix_mean",
        "pool_start_projection_0_output",
        "pool_end_projection_0_output",
    ];
    let output_count = output_names.iter().filter(|name| has(name)).count();
    let expected = if query_states.shape()[1] == 0 {
        ensure!(
            output_count == 0,
            "Q=0 public bypass fixture unexpectedly has marginal outputs"
        );
        None
    } else {
        ensure!(
            output_count == output_names.len(),
            "fixture has only {output_count}/{} marginal outputs",
            output_names.len()
        );
        Some(MarginalOutput {
            boundary_states: npz.by_name("boundary_encoder_0_states")?,
            boundary_mask: npz.by_name("boundary_encoder_0_mask")?,
            start_logits: npz.by_name("marginals_0_start_logits")?,
            end_logits: npz.by_name("marginals_0_end_logits")?,
            inside_logits: npz.by_name("marginals_0_inside_logits")?,
            inside_prefix: npz.by_name("marginals_0_inside_prefix")?,
            inside_prefix_mean: npz.by_name("marginals_0_inside_prefix_mean")?,
            start_all: npz.by_name("pool_start_projection_0_output")?,
            end_all: npz.by_name("pool_end_projection_0_output")?,
        })
    };

    Ok(Fixture {
        input: MarginalInput {
            text_states,
            text_mask,
            query_states,
            query_mask,
        },
        expected,
    })
}

#[test]
fn committed_subset_has_expected_marginal_structure() -> Result<()> {
    let directory = fixture_dir();
    ensure!(
        directory.join("manifest.json").is_file(),
        "missing committed subset manifest at {}",
        directory.display()
    );
    let mut head_cases = 0;
    let mut q0_cases = 0;
    for case_id in FIXTURE_IDS {
        let path = directory.join(format!("{case_id}.npz"));
        let fixture = load_fixture(&path)?;
        if let Some(expected) = fixture.expected {
            head_cases += 1;
            let batch = fixture.input.text_states.shape()[0];
            let length = fixture.input.text_states.shape()[1];
            let queries = fixture.input.query_states.shape()[1];
            assert_eq!(expected.boundary_states.shape(), &[batch, length + 1, 128]);
            assert_eq!(expected.boundary_mask.shape(), &[batch, length + 1]);
            assert_eq!(expected.start_logits.shape(), &[batch, queries, length + 1]);
            assert_eq!(expected.end_logits.shape(), &[batch, queries, length + 1]);
            assert_eq!(expected.inside_logits.shape(), &[batch, queries, length]);
            assert_eq!(
                expected.inside_prefix.shape(),
                &[batch, queries, length + 1]
            );
            assert_eq!(expected.inside_prefix_mean.shape(), &[batch, queries, 1]);
            assert_eq!(expected.start_all.shape(), &[batch, length + 1, 128]);
            assert_eq!(expected.end_all.shape(), &[batch, length + 1, 128]);
        } else {
            q0_cases += 1;
        }
    }
    assert_eq!(head_cases, 2);
    assert_eq!(q0_cases, 1);
    Ok(())
}

fn assert_close(
    name: &str,
    actual: &Array3<f32>,
    expected: &Array3<f32>,
    prefix_coordinate_bound: bool,
) {
    assert_eq!(actual.shape(), expected.shape(), "{name}: shape mismatch");
    let coordinate_count = expected.shape()[2];
    let mut maximum = 0.0_f32;
    let mut raw_original_failures = 0_usize;
    let mut mismatch = None;
    for (index, (&observed, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(observed.is_finite(), "{name}: non-finite actual at {index}");
        assert!(
            reference.is_finite(),
            "{name}: non-finite expected at {index}"
        );
        let difference = (observed - reference).abs();
        maximum = maximum.max(difference);
        let original_tolerance = 1e-4 + 1e-3 * reference.abs();
        raw_original_failures += usize::from(difference > original_tolerance);
        let coordinate_allowance = if prefix_coordinate_bound {
            1.1e-6 * (index % coordinate_count) as f32
        } else {
            0.0
        };
        let tolerance = original_tolerance + coordinate_allowance;
        if mismatch.is_none() && difference > tolerance {
            mismatch = Some((index, observed, reference, difference, tolerance));
        }
    }
    if let Some((index, observed, reference, difference, tolerance)) = mismatch {
        panic!(
            "{name}: mismatch at flat index {index}: actual={observed}, expected={reference}, abs={difference}, tolerance={tolerance}; max_abs={maximum}"
        );
    }
    eprintln!("{name}: max_abs={maximum:.9}, raw_original_failures={raw_original_failures}");
}

fn assert_outputs(actual: &MarginalOutput, expected: &MarginalOutput) {
    assert_eq!(actual.boundary_mask, expected.boundary_mask);
    assert_close(
        "boundary_states",
        &actual.boundary_states,
        &expected.boundary_states,
        false,
    );
    assert_close(
        "start_logits",
        &actual.start_logits,
        &expected.start_logits,
        false,
    );
    assert_close(
        "end_logits",
        &actual.end_logits,
        &expected.end_logits,
        false,
    );
    assert_close(
        "inside_logits",
        &actual.inside_logits,
        &expected.inside_logits,
        false,
    );
    assert_close(
        "inside_prefix",
        &actual.inside_prefix,
        &expected.inside_prefix,
        true,
    );
    assert_close(
        "inside_prefix_mean",
        &actual.inside_prefix_mean,
        &expected.inside_prefix_mean,
        false,
    );
    assert_close("start_all", &actual.start_all, &expected.start_all, false);
    assert_close("end_all", &actual.end_all, &expected.end_all, false);
}

fn boundary_graph_available(path: &Path) -> Result<bool> {
    if path.is_file() {
        return Ok(true);
    }
    let message = format!("missing boundary marginal graph: {}", path.display());
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(false)
}

#[test]
fn onnx_marginals_match_committed_oracle_subset() -> Result<()> {
    let graph = model_root().join("onnx/gliner2.5-base-v1/boundary_marginals.onnx");
    if !boundary_graph_available(&graph)? {
        return Ok(());
    }
    let model = MarginalModel::new(&graph)?;
    let l0_error = model
        .infer(MarginalInput {
            text_states: Array3::zeros((1, 0, 768)),
            text_mask: Array2::from_elem((1, 0), true),
            query_states: Array3::zeros((1, 1, 768)),
            query_mask: Array2::from_elem((1, 1), true),
        })
        .expect_err("L=0 must be rejected before calling ONNX Runtime");
    assert!(
        l0_error
            .to_string()
            .contains("text length must be non-zero")
    );

    let mut head_cases = 0;
    let mut q0_cases = 0;
    for case_id in FIXTURE_IDS {
        let fixture = load_fixture(&fixture_dir().join(format!("{case_id}.npz")))?;
        if fixture.expected.is_none() {
            q0_cases += 1;
            let error = model
                .infer(fixture.input)
                .expect_err("Q=0 must use the classification bypass");
            assert!(error.to_string().contains("must bypass"));
            continue;
        }
        let actual = model.infer(fixture.input)?;
        if let Some(expected) = fixture.expected {
            head_cases += 1;
            assert_outputs(&actual, &expected);
        }
    }
    assert_eq!(head_cases, 2);
    assert_eq!(q0_cases, 1);
    Ok(())
}

#[test]
fn onnx_marginals_match_full_oracle_fixtures_when_enabled() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        eprintln!("SKIP: set GLINER2_BOUNDARY_FULL_FIXTURES=1 for all 30 fixtures");
        return Ok(());
    }

    let graph = model_root().join("onnx/gliner2.5-base-v1/boundary_marginals.onnx");
    ensure!(
        graph.is_file(),
        "full marginal fixture test requires graph at {}",
        graph.display()
    );
    let directory = full_fixture_dir();
    ensure!(
        directory.join("manifest.json").is_file(),
        "full marginal fixture test requires manifest at {}",
        directory.display()
    );
    let mut paths = std::fs::read_dir(&directory)?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("npz"))
        .collect::<Vec<_>>();
    paths.sort();
    ensure!(
        paths.len() == 30,
        "expected 30 full fixtures, found {}",
        paths.len()
    );

    let model = MarginalModel::new(&graph)?;
    let mut head_cases = 0;
    let mut bypass_cases = 0;
    for path in paths {
        let fixture = load_fixture(&path)?;
        let case_id = path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("fixture path has no UTF-8 stem")?;
        if fixture.expected.is_none() {
            bypass_cases += 1;
            let error = model
                .infer(fixture.input)
                .expect_err("Q=0 must use the classification bypass");
            assert!(
                error.to_string().contains("must bypass"),
                "{case_id}: {error}"
            );
            continue;
        }
        let actual = model.infer(fixture.input)?;
        let expected = fixture.expected.context("head fixture has no outputs")?;
        assert_outputs(&actual, &expected);
        head_cases += 1;
        eprintln!("{case_id}: marginal outputs ok");
    }
    assert_eq!(head_cases, 24);
    assert_eq!(bypass_cases, 6);
    Ok(())
}
