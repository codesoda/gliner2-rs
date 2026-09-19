use std::{env, fs, fs::File, path::Path};

use anyhow::{Context, Result, bail, ensure};
use gliner2_rs::{
    boundary::classification::decode_classification,
    classification::{ClassAct, ClassificationOutput},
};
use ndarray::Array1;
use ndarray_npy::NpzReader;
use serde_json::Value;

fn labels(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 1e-4 + 1e-3 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
    );
}

#[test]
fn multilabel_hits_remain_in_schema_order() -> Result<()> {
    let output = decode_classification(
        &labels(&["low", "high", "middle"]),
        &[1.0, 3.0, 2.0],
        true,
        0.7,
        ClassAct::Sigmoid,
        1.0,
    )?;
    let ClassificationOutput::Multi { labels: chosen } = output else {
        panic!("expected multi-label output");
    };
    assert_eq!(
        chosen
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<Vec<_>>(),
        ["low", "high", "middle"]
    );
    assert!(chosen[0].1 < chosen[2].1 && chosen[2].1 < chosen[1].1);
    Ok(())
}

#[test]
fn argmax_ties_and_multilabel_fallback_select_first_label() -> Result<()> {
    let names = labels(&["first", "second", "third"]);
    let multi = decode_classification(
        &names,
        &[-2.0, -2.0, -3.0],
        true,
        1.0,
        ClassAct::Sigmoid,
        1.0,
    )?;
    let ClassificationOutput::Multi { labels: chosen } = multi else {
        panic!("expected multi-label output");
    };
    assert_eq!(chosen.len(), 1);
    assert_eq!(chosen[0].0, "first");

    // The threshold does not gate single-label output.
    let single =
        decode_classification(&names, &[2.0, 2.0, 1.0], false, 1.0, ClassAct::Softmax, 1.0)?;
    let ClassificationOutput::Single { label, .. } = single else {
        panic!("expected single-label output");
    };
    assert_eq!(label, "first");
    Ok(())
}

#[test]
fn threshold_equality_is_retained() -> Result<()> {
    let output = decode_classification(
        &labels(&["equal", "below"]),
        &[0.0, -1.0],
        true,
        0.5,
        ClassAct::Sigmoid,
        1.0,
    )?;
    let ClassificationOutput::Multi { labels: chosen } = output else {
        panic!("expected multi-label output");
    };
    assert_eq!(chosen, vec![("equal".to_owned(), 0.5)]);
    Ok(())
}

#[test]
fn temperature_precedes_explicit_and_auto_activations() -> Result<()> {
    let binary = labels(&["a", "b"]);

    let sigmoid = decode_classification(&binary, &[0.0, 2.0], false, 0.5, ClassAct::Sigmoid, 2.0)?;
    let ClassificationOutput::Single { label, confidence } = sigmoid else {
        panic!("expected single-label output");
    };
    assert_eq!(label, "b");
    assert_close(confidence, 0.731_058_6);

    let softmax = decode_classification(&binary, &[0.0, 2.0], false, 0.5, ClassAct::Softmax, 2.0)?;
    let ClassificationOutput::Single { confidence, .. } = softmax else {
        panic!("expected single-label output");
    };
    assert_close(confidence, 0.731_058_6);

    let auto_single = decode_classification(&binary, &[0.0, 2.0], false, 0.5, ClassAct::Auto, 2.0)?;
    assert_eq!(auto_single, softmax);

    let auto_multi = decode_classification(&binary, &[0.0, 2.0], true, 0.7, ClassAct::Auto, 2.0)?;
    let explicit_multi =
        decode_classification(&binary, &[0.0, 2.0], true, 0.7, ClassAct::Sigmoid, 2.0)?;
    assert_eq!(auto_multi, explicit_multi);
    let ClassificationOutput::Multi { labels: chosen } = auto_multi else {
        panic!("expected multi-label output");
    };
    assert_eq!(chosen.len(), 1);
    assert_eq!(chosen[0].0, "b");
    assert_close(chosen[0].1, 0.731_058_6);
    Ok(())
}

#[test]
fn malformed_inputs_return_errors() {
    let one = labels(&["one"]);
    assert!(decode_classification(&[], &[], false, 0.5, ClassAct::Auto, 1.0).is_err());
    assert!(decode_classification(&one, &[], false, 0.5, ClassAct::Auto, 1.0).is_err());

    for logit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(decode_classification(&one, &[logit], false, 0.5, ClassAct::Auto, 1.0).is_err());
    }
    // Finite inputs can overflow during temperature scaling; never return NaN
    // confidence from softmax(infinity).
    assert!(
        decode_classification(
            &one,
            &[f32::MAX],
            false,
            0.5,
            ClassAct::Softmax,
            f32::MIN_POSITIVE
        )
        .is_err()
    );
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            decode_classification(&one, &[0.0], false, 0.5, ClassAct::Auto, temperature).is_err()
        );
    }
    for threshold in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        assert!(
            decode_classification(&one, &[0.0], false, threshold, ClassAct::Auto, 1.0).is_err()
        );
    }
}

fn assert_fixture(path: &Path) -> Result<()> {
    let metadata: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    ensure!(metadata["status"] == "ok", "fixture is not successful");
    let configs = metadata["schema_spec"]["classifications"]
        .as_array()
        .context("classification configs")?;
    let array_metadata = metadata["arrays"].as_object().context("array metadata")?;
    let expected = metadata["final_result_utf8_offsets"]
        .as_object()
        .context("final classification outputs")?;
    let npz_path = path.with_extension("npz");
    let mut npz = NpzReader::new(File::open(&npz_path)?)?;

    for (index, config) in configs.iter().enumerate() {
        let task = config["task"].as_str().context("classification task")?;
        let label_values = config["labels"]
            .as_array()
            .context("classification labels")?;
        let task_labels: Vec<_> = label_values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .context("classification label string")
            })
            .collect::<Result<_>>()?;
        let key = format!("classifier_{index}_raw_logits");
        ensure!(
            array_metadata.contains_key(&key),
            "{} metadata is missing {key}",
            path.display()
        );
        let logits: Array1<f32> = npz.by_name(&key)?;
        let multi_label = config["multi_label"].as_bool().unwrap_or(false);
        let threshold = config["cls_threshold"].as_f64().unwrap_or(0.5) as f32;
        let activation = config["class_act"]
            .as_str()
            .and_then(ClassAct::parse)
            .unwrap_or(ClassAct::Auto);
        let actual = decode_classification(
            &task_labels,
            logits.as_slice().context("contiguous classifier logits")?,
            multi_label,
            threshold,
            activation,
            1.0,
        )?;
        let expected_task = expected
            .get(task)
            .with_context(|| format!("missing final output for {task}"))?;

        match actual {
            ClassificationOutput::Single { label, confidence } => {
                ensure!(!multi_label, "{task}: expected multi-label output");
                assert_eq!(
                    label,
                    expected_task["label"].as_str().context("final label")?,
                    "{task}"
                );
                let expected_confidence = expected_task["confidence"]
                    .as_f64()
                    .context("final confidence")? as f32;
                assert!(
                    (confidence - expected_confidence).abs() <= 1e-3,
                    "{task}: confidence {confidence} != {expected_confidence}"
                );
            }
            ClassificationOutput::Multi { labels } => {
                ensure!(multi_label, "{task}: expected single-label output");
                let expected_rows = expected_task.as_array().context("final multi labels")?;
                assert_eq!(labels.len(), expected_rows.len(), "{task}");
                for ((label, confidence), row) in labels.iter().zip(expected_rows) {
                    assert_eq!(
                        label,
                        row["label"].as_str().context("final label")?,
                        "{task}"
                    );
                    let expected_confidence =
                        row["confidence"].as_f64().context("final confidence")? as f32;
                    assert!(
                        (*confidence - expected_confidence).abs() <= 1e-3,
                        "{task}: confidence {confidence} != {expected_confidence}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn committed_classifier_logits_reproduce_final_multi_task_output() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/gliner2.5-base-v1-subset/classification_multi_task.json");
    assert_fixture(&path)
}

#[test]
fn opt_in_full_classification_fixtures_reproduce_final_outputs() -> Result<()> {
    if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1") {
        return Ok(());
    }
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1");
    let mut paths: Vec<_> = fs::read_dir(&directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
            && path
                .file_stem()
                .is_some_and(|stem| stem.to_string_lossy().starts_with("classification"))
    });
    paths.sort();
    if paths.is_empty() {
        bail!("no full classification fixtures in {}", directory.display());
    }
    for path in &paths {
        assert_fixture(path)?;
    }
    assert_eq!(paths.len(), 5);
    Ok(())
}
