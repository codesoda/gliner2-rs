//! Source-faithful classification decoding for the boundary architecture.
//!
//! This intentionally differs from the legacy v2 decoder: multi-label results
//! remain in schema order, and equal-probability argmax ties select the first
//! label.

use anyhow::{Result, ensure};

use crate::classification::{ClassAct, ClassificationOutput};
use crate::scores::{Activation, ClassificationScores};

/// Decode one boundary-classifier output after temperature scaling.
///
/// Multi-label hits use an inclusive threshold and retain the input label
/// order. If none pass, the first maximum is retained. Single-label decoding
/// always returns the first maximum. Invalid inputs are errors rather than
/// indexing or assertion panics.
///
/// This is the selection view of [`ClassificationScores`]; both share one
/// arithmetic path. Duplicate labels are accepted here for compatibility with
/// the historical decoder, so the distribution is formed on de-duplicated
/// positions and then re-expanded.
pub fn decode_classification(
    labels: &[String],
    logits: &[f32],
    multi_label: bool,
    cls_threshold: f32,
    class_act: ClassAct,
    classification_temperature: f32,
) -> Result<ClassificationOutput> {
    ensure!(
        cls_threshold.is_finite() && (0.0..=1.0).contains(&cls_threshold),
        "classification threshold must be finite and in [0,1], got {cls_threshold}"
    );
    // Positional stand-in names keep the arithmetic identical when a caller
    // repeats a label string; the real labels are restored afterwards.
    let positions: Vec<String> = (0..labels.len()).map(|index| index.to_string()).collect();
    let scores = ClassificationScores::new(
        "",
        positions,
        logits.to_vec(),
        classification_temperature,
        Activation::from_class_act(class_act, multi_label),
    )?;
    let restore = |position: &str| labels[position.parse::<usize>().expect("position")].clone();
    Ok(if multi_label {
        match scores.select_multi(cls_threshold) {
            ClassificationOutput::Multi { labels: chosen } => ClassificationOutput::Multi {
                labels: chosen
                    .into_iter()
                    .map(|(position, probability)| (restore(&position), probability))
                    .collect(),
            },
            single => single,
        }
    } else {
        match scores.select_single() {
            ClassificationOutput::Single { label, confidence } => ClassificationOutput::Single {
                label: restore(&label),
                confidence,
            },
            multi => multi,
        }
    })
}
