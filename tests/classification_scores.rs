//! Complete-distribution classification against pinned upstream goldens.
//!
//! `fixtures/classification-scores/<bundle>.json` holds, per case and task,
//! the raw logits observed on the public Python `extract` path and the
//! probabilities computed independently with torch. For every bundle present
//! under `onnx/`, this file checks the classifier-only pipeline against those
//! goldens, then checks the complete pipeline produces identical numbers.
//!
//! Tolerances are fixed here, before any comparison, and reuse the existing
//! classifier export gate: raw logits within `1e-4 + 1e-3 * |expected|`,
//! probabilities within `1e-4`, argmax identical, and the historical winner
//! equal to the public Python result.

mod common;

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use gliner2_rs::{
    Activation, ClassificationPipeline, ClassificationRequest, ClassificationScores,
    RuntimeOptions,
    boundary::BoundaryPipeline,
    classification::{ClassAct, ClassificationOutput},
};
use serde::Deserialize;

const LOGIT_ATOL: f32 = 1e-4;
const LOGIT_RTOL: f32 = 1e-3;
const PROBABILITY_ATOL: f32 = 1e-4;

const BUNDLES: &[&str] = &[
    "gliner2.5-small-v1",
    "gliner2.5-base-v1",
    "gliner2.5-multi-v1",
];

#[derive(Deserialize)]
struct Golden {
    bundle_name: String,
    classification_temperature: f32,
    max_len: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    text: String,
    input_tokens: usize,
    truncated_words: usize,
    tasks: Vec<Task>,
}

#[derive(Deserialize)]
struct Task {
    task: String,
    labels: Vec<String>,
    instruction: Option<String>,
    label_descriptions: Vec<(String, String)>,
    multi_label: bool,
    cls_threshold: f32,
    class_act: String,
    activation: String,
    raw_logits: Vec<f32>,
    probabilities: Vec<f32>,
    public_result: serde_json::Value,
}

impl Task {
    fn request(&self) -> ClassificationRequest {
        let activation = match self.activation.as_str() {
            "softmax" => Activation::Softmax,
            "sigmoid" => Activation::Sigmoid,
            other => panic!("unknown activation {other}"),
        };
        let mut request = ClassificationRequest::new(&self.task, self.labels.clone(), activation);
        if let Some(instruction) = &self.instruction {
            request = request.with_instruction(instruction.clone());
        }
        request.with_label_descriptions(self.label_descriptions.clone())
    }

    fn class_act(&self) -> ClassAct {
        ClassAct::parse(&self.class_act).expect("class_act")
    }
}

fn golden_path(bundle: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/classification-scores/{bundle}.json"))
}

fn load_golden(bundle: &str) -> Result<Golden> {
    let path = golden_path(bundle);
    let golden: Golden = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?;
    ensure!(golden.bundle_name == bundle, "golden bundle name mismatch");
    Ok(golden)
}

/// Bundles present locally. Absent bundles skip loudly unless strict mode.
fn available_bundles() -> Result<Vec<(String, PathBuf)>> {
    let root = common::model_root();
    let mut found = Vec::new();
    for bundle in BUNDLES {
        let dir = root.join("onnx").join(bundle);
        let paths = ClassificationPipeline::required_paths(&dir);
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        if common::artifacts_available(&refs)? {
            found.push(((*bundle).to_owned(), dir));
        }
    }
    Ok(found)
}

fn assert_scores(case: &Case, task: &Task, scores: &ClassificationScores, temperature: f32) {
    let id = format!("{}/{}", case.id, task.task);
    assert_eq!(scores.task(), task.task, "{id}");
    assert_eq!(scores.labels(), task.labels.as_slice(), "{id}: label order");
    assert_eq!(scores.temperature(), temperature, "{id}");
    assert_eq!(scores.activation().as_str(), task.activation, "{id}");
    assert_eq!(scores.raw_logits().len(), task.raw_logits.len(), "{id}");
    for (index, (actual, expected)) in scores.raw_logits().iter().zip(&task.raw_logits).enumerate()
    {
        let limit = LOGIT_ATOL + LOGIT_RTOL * expected.abs();
        assert!(
            (actual - expected).abs() <= limit,
            "{id}: logit {index} {actual} vs {expected} exceeds {limit}"
        );
    }
    for (index, (actual, expected)) in scores
        .probabilities()
        .iter()
        .zip(&task.probabilities)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= PROBABILITY_ATOL,
            "{id}: probability {index} {actual} vs {expected}"
        );
    }
    let expected_argmax = task
        .probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, value)| {
            if *value > task.probabilities[best] {
                index
            } else {
                best
            }
        });
    assert_eq!(scores.argmax(), expected_argmax, "{id}: argmax");
    assert_eq!(
        scores.is_categorical(),
        task.activation == "softmax",
        "{id}"
    );
    assert_eq!(
        scores.usage().input_tokens,
        case.input_tokens,
        "{id}: input tokens"
    );
    assert_eq!(
        scores.usage().truncated_words,
        case.truncated_words,
        "{id}: truncation"
    );
}

fn assert_public_winner(case: &Case, task: &Task, scores: &ClassificationScores) {
    let id = format!("{}/{}", case.id, task.task);
    let decoded = if task.multi_label {
        scores.select_multi(task.cls_threshold)
    } else {
        scores.select_single()
    };
    match decoded {
        ClassificationOutput::Single { label, .. } => {
            assert_eq!(task.public_result["label"], label, "{id}: public winner");
        }
        ClassificationOutput::Multi { labels } => {
            let expected: Vec<&str> = task
                .public_result
                .as_array()
                .expect("multi result")
                .iter()
                .map(|row| row["label"].as_str().expect("label"))
                .collect();
            let actual: Vec<&str> = labels.iter().map(|(label, _)| label.as_str()).collect();
            assert_eq!(actual, expected, "{id}: public multi-label set");
        }
    }
}

#[test]
fn classifier_only_pipeline_matches_upstream_goldens() -> Result<()> {
    for (bundle, dir) in available_bundles()? {
        let golden = load_golden(&bundle)?;
        let pipeline = ClassificationPipeline::from_dir(&dir)?;
        assert_eq!(pipeline.max_len(), golden.max_len);
        assert_eq!(
            pipeline.classification_temperature(),
            golden.classification_temperature
        );
        let mut max_logit = 0f32;
        let mut max_probability = 0f32;
        for case in &golden.cases {
            let requests: Vec<_> = case.tasks.iter().map(Task::request).collect();
            let scores = pipeline.score_classifications(&case.text, &requests)?;
            assert_eq!(scores.len(), case.tasks.len(), "{}", case.id);
            for (task, scores) in case.tasks.iter().zip(&scores) {
                assert_scores(case, task, scores, golden.classification_temperature);
                assert_public_winner(case, task, scores);
                for (a, b) in scores.raw_logits().iter().zip(&task.raw_logits) {
                    max_logit = max_logit.max((a - b).abs());
                }
                for (a, b) in scores.probabilities().iter().zip(&task.probabilities) {
                    max_probability = max_probability.max((a - b).abs());
                }
            }
        }
        eprintln!(
            "{bundle}: {} cases, max |Δlogit|={max_logit:.3e}, max |Δp|={max_probability:.3e}",
            golden.cases.len()
        );
    }
    Ok(())
}

#[test]
fn complete_pipeline_scores_are_identical_to_classifier_only() -> Result<()> {
    for (bundle, dir) in available_bundles()? {
        let full_paths = [
            "boundary_marginals.onnx",
            "boundary_scorer.onnx",
            "boundary_explicit_scorer.onnx",
            "boundary_records.onnx",
            "boundary_relations.onnx",
        ]
        .map(|name| dir.join(name));
        let refs: Vec<&Path> = full_paths.iter().map(PathBuf::as_path).collect();
        if !common::artifacts_available(&refs)? {
            continue;
        }
        let golden = load_golden(&bundle)?;
        let only = ClassificationPipeline::from_dir(&dir)?;
        let full = BoundaryPipeline::from_dir(&dir)?;
        for case in &golden.cases {
            let requests: Vec<_> = case.tasks.iter().map(Task::request).collect();
            let a = only.score_classifications(&case.text, &requests)?;
            let b = full.score_classifications(&case.text, &requests)?;
            assert_eq!(a, b, "{bundle}/{}: pipelines differ", case.id);
            if case.tasks.len() != 1 {
                // The historical API scores one task per prompt; a multi-task
                // prompt is a different encoder input.
                continue;
            }
            for (task, scores) in case.tasks.iter().zip(&a) {
                // The historical API must agree with the distribution's own
                // selection on the same request.
                let legacy = full.classify_with_descriptions_and_options(
                    &case.text,
                    &task.task,
                    &task.labels,
                    &task.label_descriptions,
                    task.multi_label,
                    task.cls_threshold,
                    task.class_act(),
                )?;
                if task.instruction.is_none() {
                    let selected = if task.multi_label {
                        scores.select_multi(task.cls_threshold)
                    } else {
                        scores.select_single()
                    };
                    assert_eq!(legacy, selected, "{bundle}/{}", case.id);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn classifier_only_loads_from_a_directory_without_extraction_heads() -> Result<()> {
    let Some((bundle, dir)) = available_bundles()?.into_iter().next() else {
        return Ok(());
    };
    let golden = load_golden(&bundle)?;
    let minimal = tempfile::tempdir()?;
    for path in ClassificationPipeline::required_paths(&dir) {
        let name = path.file_name().unwrap();
        // A hard link keeps this cheap; copy if the filesystem refuses.
        if fs::hard_link(&path, minimal.path().join(name)).is_err() {
            fs::copy(&path, minimal.path().join(name))?;
        }
    }
    let names: Vec<_> = fs::read_dir(minimal.path())?
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names.len(), 4, "{names:?}");
    assert!(names.iter().all(|name| !name.starts_with("boundary_")));

    let pipeline = ClassificationPipeline::from_dir_with_options(
        minimal.path(),
        RuntimeOptions::default().with_intra_threads(1),
    )?;
    let report = pipeline.runtime_report();
    assert_eq!(report.intra_threads, 1);
    assert_eq!(report.sessions, ["encoder", "classifier"]);
    assert!(
        report.native_runtime.contains("1.28"),
        "{}",
        report.native_runtime
    );

    let case = &golden.cases[0];
    let scores = pipeline.score_classification(&case.text, &case.tasks[0].request())?;
    assert_scores(
        case,
        &case.tasks[0],
        &scores,
        golden.classification_temperature,
    );

    // The same directory is not a complete extraction bundle.
    assert!(BoundaryPipeline::from_dir(minimal.path()).is_err());
    Ok(())
}

#[test]
fn differently_configured_instances_do_not_share_settings() -> Result<()> {
    let Some((bundle, dir)) = available_bundles()?.into_iter().next() else {
        return Ok(());
    };
    let golden = load_golden(&bundle)?;
    let one = ClassificationPipeline::from_dir_with_options(
        &dir,
        RuntimeOptions::default().with_intra_threads(1),
    )?;
    let two = ClassificationPipeline::from_dir_with_options(
        &dir,
        RuntimeOptions::default()
            .with_intra_threads(2)
            .with_inter_threads(Some(2))
            .with_optimization_level(gliner2_rs::OptimizationLevel::Basic),
    )?;
    assert_eq!(one.runtime_report().intra_threads, 1);
    assert_eq!(two.runtime_report().intra_threads, 2);
    assert_eq!(two.runtime_report().inter_threads, Some(2));
    assert_eq!(
        two.runtime_report().optimization_level,
        gliner2_rs::OptimizationLevel::Basic
    );
    assert_eq!(one.runtime_options(), one.runtime_report().into_options());

    // Thread count and optimization level may reorder float reductions, so
    // decisions must agree and values must stay inside the golden gate; bit
    // equality is not required.
    for case in golden.cases.iter().take(4) {
        let request = case.tasks[0].request();
        let a = one.score_classification(&case.text, &request)?;
        let b = two.score_classification(&case.text, &request)?;
        assert_eq!(a.argmax(), b.argmax(), "{}", case.id);
        assert_scores(case, &case.tasks[0], &b, golden.classification_temperature);
    }
    Ok(())
}
