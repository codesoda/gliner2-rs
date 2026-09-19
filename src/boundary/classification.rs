//! Source-faithful classification decoding for the boundary architecture.
//!
//! This intentionally differs from the legacy v2 decoder: multi-label results
//! remain in schema order, and equal-probability argmax ties select the first
//! label.

use anyhow::{Result, ensure};

use crate::classification::{ClassAct, ClassificationOutput};

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<_> = logits
        .iter()
        .map(|&value| (value - maximum).exp())
        .collect();
    let total: f32 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

fn first_argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

/// Decode one boundary-classifier output after temperature scaling.
///
/// Multi-label hits use an inclusive threshold and retain the input label
/// order. If none pass, the first maximum is retained. Single-label decoding
/// always returns the first maximum. Invalid inputs are errors rather than
/// indexing or assertion panics.
pub fn decode_classification(
    labels: &[String],
    logits: &[f32],
    multi_label: bool,
    cls_threshold: f32,
    class_act: ClassAct,
    classification_temperature: f32,
) -> Result<ClassificationOutput> {
    ensure!(
        !labels.is_empty(),
        "classification labels must be non-empty"
    );
    ensure!(
        labels.len() == logits.len(),
        "classification labels/logits length mismatch: {} != {}",
        labels.len(),
        logits.len()
    );
    ensure!(
        classification_temperature.is_finite() && classification_temperature > 0.0,
        "classification temperature must be finite and positive, got {classification_temperature}"
    );
    ensure!(
        cls_threshold.is_finite() && (0.0..=1.0).contains(&cls_threshold),
        "classification threshold must be finite and in [0,1], got {cls_threshold}"
    );
    for (index, &logit) in logits.iter().enumerate() {
        ensure!(
            logit.is_finite(),
            "classification logit {index} must be finite, got {logit}"
        );
    }

    let scaled: Vec<_> = logits
        .iter()
        .map(|&logit| logit / classification_temperature)
        .collect();
    let probabilities: Vec<f32> = match class_act {
        ClassAct::Sigmoid => scaled.into_iter().map(sigmoid).collect(),
        ClassAct::Softmax => softmax(&scaled),
        ClassAct::Auto if multi_label => scaled.into_iter().map(sigmoid).collect(),
        ClassAct::Auto => softmax(&scaled),
    };
    ensure!(
        probabilities.iter().all(|value| value.is_finite()),
        "classification activation produced non-finite probabilities after temperature scaling"
    );

    if multi_label {
        let mut chosen: Vec<_> = labels
            .iter()
            .cloned()
            .zip(probabilities.iter().copied())
            .filter(|(_, probability)| *probability >= cls_threshold)
            .collect();
        if chosen.is_empty() {
            let best = first_argmax(&probabilities);
            chosen.push((labels[best].clone(), probabilities[best]));
        }
        Ok(ClassificationOutput::Multi { labels: chosen })
    } else {
        let best = first_argmax(&probabilities);
        Ok(ClassificationOutput::Single {
            label: labels[best].clone(),
            confidence: probabilities[best],
        })
    }
}
