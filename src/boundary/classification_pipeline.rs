//! Classifier-only view of a boundary bundle.
//!
//! [`ClassificationPipeline`] opens exactly three assets: the tokenizer, the
//! encoder graph and the classifier graph. It never touches the extraction
//! heads, so a directory that holds only those files loads, and one that lacks
//! any of them is rejected with the missing name. The full-bundle validator is
//! unchanged: such a directory is not a complete extraction bundle.
//!
//! Scoring shares the same prompt assembly, encoder call, label-state gather
//! and temperature/activation arithmetic as [`super::BoundaryPipeline`]. Given the
//! same checkpoint, options and input, both produce the same logits.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};

use super::{
    classification::decode_classification,
    config::BoundaryRuntimeConfig,
    pipeline::score_classifications,
    preprocessing::{BoundaryPreprocessingPolicy, WordSplitter},
};
use crate::{
    classification::{ClassAct, ClassificationOutput},
    classifier::Classifier,
    encoder::Encoder,
    options::{RuntimeOptions, RuntimeReport},
    scores::{ClassificationRequest, ClassificationScores},
    tokenizer::RuntimeTokenizer,
};

/// Bundle-relative files a classifier-only load reads. `config.json` is read
/// for the architecture check, `max_len` and the classification temperature.
pub const CLASSIFICATION_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "encoder.onnx",
    "classifier.onnx",
];

/// Names of the sessions this pipeline opens, in load order.
const SESSIONS: &[&str] = &["encoder", "classifier"];

/// Boundary classification without extraction heads.
pub struct ClassificationPipeline {
    tokenizer: RuntimeTokenizer,
    encoder: Encoder,
    classifier: Classifier,
    runtime: BoundaryRuntimeConfig,
    preprocessing: BoundaryPreprocessingPolicy,
    options: RuntimeOptions,
}

impl ClassificationPipeline {
    /// Load with the default CPU runtime options.
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with_options(bundle, RuntimeOptions::default())
    }

    /// Load the tokenizer, encoder and classifier from `bundle` with explicit
    /// runtime options. Options are validated before any file is opened.
    pub fn from_dir_with_options(
        bundle: impl AsRef<Path>,
        options: RuntimeOptions,
    ) -> Result<Self> {
        options
            .validate()
            .context("invalid classification runtime options")?;
        let bundle = bundle.as_ref();
        let missing = missing_files(bundle);
        ensure!(
            missing.is_empty(),
            "classification bundle {} is missing {}",
            bundle.display(),
            missing.join(", ")
        );
        let runtime = BoundaryRuntimeConfig::from_dir(bundle)?;
        let preprocessing =
            BoundaryPreprocessingPolicy::new(runtime.max_len, WordSplitter::Whitespace)
                .map_err(|error| anyhow!(error))?;
        let encoder_path = bundle.join("encoder.onnx");
        let classifier_path = bundle.join("classifier.onnx");
        Ok(Self {
            tokenizer: RuntimeTokenizer::from_dir(bundle).with_context(|| {
                format!(
                    "failed to load boundary tokenizer from {}",
                    bundle.display()
                )
            })?,
            encoder: Encoder::new_with_options(&encoder_path, options).with_context(|| {
                format!(
                    "failed to load boundary encoder at {}",
                    encoder_path.display()
                )
            })?,
            classifier: Classifier::new_with_options(&classifier_path, options).with_context(
                || {
                    format!(
                        "failed to load boundary classifier at {}",
                        classifier_path.display()
                    )
                },
            )?,
            runtime,
            preprocessing,
            options,
        })
    }

    /// Files under `bundle` that a classifier-only load needs but cannot find.
    pub fn missing_files(bundle: impl AsRef<Path>) -> Vec<String> {
        missing_files(bundle.as_ref())
    }

    /// Absolute paths of the files this pipeline reads from `bundle`.
    pub fn required_paths(bundle: impl AsRef<Path>) -> Vec<PathBuf> {
        let bundle = bundle.as_ref();
        CLASSIFICATION_FILES
            .iter()
            .map(|name| bundle.join(name))
            .collect()
    }

    pub fn runtime_report(&self) -> RuntimeReport {
        RuntimeReport::new(self.options, SESSIONS.to_vec())
    }

    pub const fn runtime_options(&self) -> RuntimeOptions {
        self.options
    }

    /// Word cap applied to the text before the prompt is added. Words past
    /// this cap are dropped and reported through `ScoringUsage`.
    pub const fn max_len(&self) -> usize {
        self.runtime.max_len
    }

    /// Checkpoint classification temperature applied once before activation.
    pub const fn classification_temperature(&self) -> f32 {
        self.runtime.classification_temperature
    }

    /// Complete distribution for one task. See
    /// [`super::BoundaryPipeline::score_classification`].
    pub fn score_classification(
        &self,
        text: &str,
        request: &ClassificationRequest,
    ) -> Result<ClassificationScores> {
        let mut scores = self.score_classifications(text, std::slice::from_ref(request))?;
        ensure!(
            scores.len() == 1,
            "single classification request produced {} task outputs",
            scores.len()
        );
        Ok(scores.remove(0))
    }

    /// Complete distributions for several tasks over one encoder pass. See
    /// [`super::BoundaryPipeline::score_classifications`].
    pub fn score_classifications(
        &self,
        text: &str,
        requests: &[ClassificationRequest],
    ) -> Result<Vec<ClassificationScores>> {
        score_classifications(
            &self.tokenizer,
            &self.encoder,
            &self.classifier,
            self.preprocessing,
            self.runtime.classification_temperature,
            text,
            requests,
        )
    }

    /// Historical winner/threshold output, decoded from the same scores.
    pub fn classify_with_options(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        let request = ClassificationRequest::new(
            task,
            labels.to_vec(),
            crate::scores::Activation::from_class_act(class_act, multi_label),
        );
        let scores = self.score_classification(text, &request)?;
        decode_classification(
            labels,
            scores.raw_logits(),
            multi_label,
            cls_threshold,
            class_act,
            self.runtime.classification_temperature,
        )
    }

    pub fn classify(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
    ) -> Result<ClassificationOutput> {
        self.classify_with_options(
            text,
            task,
            labels,
            multi_label,
            cls_threshold,
            ClassAct::Auto,
        )
    }
}

fn missing_files(bundle: &Path) -> Vec<String> {
    CLASSIFICATION_FILES
        .iter()
        .filter(|name| !bundle.join(name).is_file())
        .map(|name| (*name).to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn reports_every_missing_required_file() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let missing = ClassificationPipeline::missing_files(dir.path());
        assert_eq!(missing, ["config.json", "encoder.onnx", "classifier.onnx"]);
        let message = match ClassificationPipeline::from_dir(dir.path()) {
            Ok(_) => panic!("loaded from an incomplete directory"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("config.json"), "{message}");
        assert!(message.contains("classifier.onnx"), "{message}");
    }

    #[test]
    fn required_paths_never_include_extraction_heads() {
        let paths = ClassificationPipeline::required_paths("/bundle");
        let names: Vec<_> = paths
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "config.json",
                "tokenizer.json",
                "encoder.onnx",
                "classifier.onnx"
            ]
        );
        assert!(names.iter().all(|name| !name.starts_with("boundary_")));
    }

    #[test]
    fn invalid_options_fail_before_files_are_checked() {
        let dir = tempdir().unwrap();
        let message = match ClassificationPipeline::from_dir_with_options(
            dir.path(),
            RuntimeOptions::default().with_intra_threads(0),
        ) {
            Ok(_) => panic!("loaded with zero threads"),
            Err(error) => format!("{error:#}"),
        };
        assert!(
            message.contains("thread count must be positive"),
            "{message}"
        );
    }
}
