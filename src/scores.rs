//! Complete ordered classification distributions.
//!
//! [`ClassificationScores`] keeps every caller label in the caller's order with
//! its raw logit, temperature-scaled logit and activated probability. Nothing
//! is filtered, sorted or renormalised. Selection helpers reproduce the
//! historical winner and multi-label outputs from the same vector, so a
//! consumer that needs the distribution and one that needs a decision share
//! exactly one arithmetic path.
//!
//! Two activations exist and they mean different things:
//!
//! - [`Activation::Softmax`] gives one categorical distribution over the
//!   labels. Probabilities are non-negative and sum to one within float error.
//! - [`Activation::Sigmoid`] gives one independent Bernoulli probability per
//!   label. These do **not** sum to one and must not be treated as, or
//!   renormalised into, a categorical distribution.

use std::{error::Error, fmt};

use crate::classification::{ClassAct, ClassificationOutput};

/// Activation applied to the temperature-scaled logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Activation {
    Softmax,
    Sigmoid,
}

impl Activation {
    /// Resolve the legacy `ClassAct` request the way the pinned upstream does:
    /// `Auto` is sigmoid for multi-label tasks and softmax otherwise.
    pub const fn from_class_act(class_act: ClassAct, multi_label: bool) -> Self {
        match class_act {
            ClassAct::Sigmoid => Self::Sigmoid,
            ClassAct::Softmax => Self::Softmax,
            ClassAct::Auto if multi_label => Self::Sigmoid,
            ClassAct::Auto => Self::Softmax,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::Sigmoid => "sigmoid",
        }
    }

    /// Whether the activated vector is one categorical distribution.
    pub const fn is_categorical(self) -> bool {
        matches!(self, Self::Softmax)
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why a distribution could not be formed.
#[derive(Clone, Debug, PartialEq)]
pub enum ScoresError {
    EmptyLabels,
    LengthMismatch { labels: usize, logits: usize },
    DuplicateLabel(String),
    ReservedMarker { field: &'static str, value: String },
    UnknownDescription(String),
    InvalidTemperature(f32),
    NonFiniteLogit { index: usize, value: f32 },
    NonFiniteProbability { index: usize },
    NotNormalized { sum: f32, tolerance: f32 },
}

impl fmt::Display for ScoresError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLabels => formatter.write_str("classification labels must be non-empty"),
            Self::LengthMismatch { labels, logits } => write!(
                formatter,
                "classification labels/logits length mismatch: {labels} != {logits}"
            ),
            Self::DuplicateLabel(label) => write!(
                formatter,
                "classification label {label:?} appears more than once; a distribution needs distinct labels"
            ),
            Self::ReservedMarker { field, value } => write!(
                formatter,
                "classification {field} {value:?} contains a reserved prompt marker"
            ),
            Self::UnknownDescription(label) => write!(
                formatter,
                "classification description for {label:?} does not match any requested label"
            ),
            Self::InvalidTemperature(value) => write!(
                formatter,
                "classification temperature must be finite and positive, got {value}"
            ),
            Self::NonFiniteLogit { index, value } => write!(
                formatter,
                "classification logit {index} must be finite, got {value}"
            ),
            Self::NonFiniteProbability { index } => write!(
                formatter,
                "classification activation produced a non-finite probability at index {index}"
            ),
            Self::NotNormalized { sum, tolerance } => write!(
                formatter,
                "softmax probabilities sum to {sum}, outside 1 ± {tolerance}"
            ),
        }
    }
}

impl Error for ScoresError {}

/// Tokens the prompt formatter treats as structure. A label or instruction
/// containing one would move marker positions and misalign the label states.
pub const RESERVED_MARKERS: &[&str] = &["[P]", "[L]", "[E]", "[C]", "[R]", "[DESCRIPTION]"];

/// Categorical sums are checked against `1 ± SOFTMAX_SUM_TOLERANCE`.
///
/// f32 softmax over a few dozen labels accumulates error well below 1e-5;
/// the tolerance is fixed here so that consumers do not each pick their own.
pub const SOFTMAX_SUM_TOLERANCE: f32 = 1e-4;

/// One classification task to score in full.
///
/// `labels` are returned in this order. `instruction` is rendered after the
/// task name exactly as the upstream `prompt` field (`"{task}: {instruction}"`).
/// `label_descriptions` follow the upstream `[DESCRIPTION] label: text` form
/// and are matched to labels by exact string; unknown descriptions are
/// rejected instead of silently dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationRequest {
    pub task: String,
    pub labels: Vec<String>,
    pub instruction: Option<String>,
    pub label_descriptions: Vec<(String, String)>,
    pub activation: Activation,
}

impl ClassificationRequest {
    pub fn new(task: impl Into<String>, labels: Vec<String>, activation: Activation) -> Self {
        Self {
            task: task.into(),
            labels,
            instruction: None,
            label_descriptions: Vec::new(),
            activation,
        }
    }

    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    pub fn with_label_descriptions(mut self, descriptions: Vec<(String, String)>) -> Self {
        self.label_descriptions = descriptions;
        self
    }

    /// Structural checks that need no model: non-empty distinct labels, no
    /// reserved markers anywhere, and descriptions only for declared labels.
    pub fn validate(&self) -> Result<(), ScoresError> {
        if self.labels.is_empty() {
            return Err(ScoresError::EmptyLabels);
        }
        check_reserved("task", &self.task)?;
        for (index, label) in self.labels.iter().enumerate() {
            if self.labels[..index].contains(label) {
                return Err(ScoresError::DuplicateLabel(label.clone()));
            }
            check_reserved("label", label)?;
        }
        if let Some(instruction) = &self.instruction {
            check_reserved("instruction", instruction)?;
        }
        for (label, description) in &self.label_descriptions {
            if !self.labels.contains(label) {
                return Err(ScoresError::UnknownDescription(label.clone()));
            }
            check_reserved("description", description)?;
        }
        Ok(())
    }
}

/// Encoder-side facts a consumer needs to judge one scored request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScoringUsage {
    /// Sub-word tokens in the encoder sequence, prompt included.
    pub input_tokens: usize,
    /// Normalized text words dropped by the checkpoint's `max_len` cap.
    pub truncated_words: usize,
}

impl ScoringUsage {
    pub const fn truncated(self) -> bool {
        self.truncated_words > 0
    }
}

/// Complete distribution for one classification task.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassificationScores {
    task: String,
    labels: Vec<String>,
    raw_logits: Vec<f32>,
    temperature: f32,
    scaled_logits: Vec<f32>,
    activation: Activation,
    probabilities: Vec<f32>,
    usage: ScoringUsage,
}

impl ClassificationScores {
    /// Form the distribution from the classifier's raw per-label logits.
    ///
    /// `temperature` is the checkpoint's classification temperature; it is
    /// applied once before the activation. Labels must be distinct.
    pub fn new(
        task: impl Into<String>,
        labels: Vec<String>,
        raw_logits: Vec<f32>,
        temperature: f32,
        activation: Activation,
    ) -> Result<Self, ScoresError> {
        if labels.is_empty() {
            return Err(ScoresError::EmptyLabels);
        }
        if labels.len() != raw_logits.len() {
            return Err(ScoresError::LengthMismatch {
                labels: labels.len(),
                logits: raw_logits.len(),
            });
        }
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(ScoresError::InvalidTemperature(temperature));
        }
        if let Some((index, &value)) = raw_logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ScoresError::NonFiniteLogit { index, value });
        }
        for (index, label) in labels.iter().enumerate() {
            if labels[..index].contains(label) {
                return Err(ScoresError::DuplicateLabel(label.clone()));
            }
        }

        let scaled_logits: Vec<f32> = raw_logits
            .iter()
            .map(|&logit| logit / temperature)
            .collect();
        let probabilities = match activation {
            Activation::Sigmoid => scaled_logits.iter().copied().map(sigmoid).collect(),
            Activation::Softmax => softmax(&scaled_logits),
        };
        if let Some(index) = probabilities.iter().position(|value| !value.is_finite()) {
            return Err(ScoresError::NonFiniteProbability { index });
        }
        if activation.is_categorical() {
            let sum: f32 = probabilities.iter().sum();
            if (sum - 1.0).abs() > SOFTMAX_SUM_TOLERANCE {
                return Err(ScoresError::NotNormalized {
                    sum,
                    tolerance: SOFTMAX_SUM_TOLERANCE,
                });
            }
        }

        Ok(Self {
            task: task.into(),
            labels,
            raw_logits,
            temperature,
            scaled_logits,
            activation,
            probabilities,
            usage: ScoringUsage::default(),
        })
    }

    pub(crate) fn with_usage(mut self, usage: ScoringUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Encoder usage for the request that produced these scores. Zero for
    /// distributions formed directly from logits.
    pub fn usage(&self) -> ScoringUsage {
        self.usage
    }

    pub fn task(&self) -> &str {
        &self.task
    }

    /// Labels in the caller's order. Every other vector is aligned to this.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn raw_logits(&self) -> &[f32] {
        &self.raw_logits
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// `raw_logits / temperature`, the activation's input.
    pub fn scaled_logits(&self) -> &[f32] {
        &self.scaled_logits
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Activated values, aligned with [`labels`](Self::labels). Categorical
    /// only when [`is_categorical`](Self::is_categorical) is true.
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }

    pub fn is_categorical(&self) -> bool {
        self.activation.is_categorical()
    }

    /// Index of the first maximum probability. Ties go to the earlier label,
    /// matching the pinned upstream `argmax`.
    pub fn argmax(&self) -> usize {
        first_argmax(&self.probabilities)
    }

    /// Historical single-label output: the first maximum.
    pub fn select_single(&self) -> ClassificationOutput {
        let best = self.argmax();
        ClassificationOutput::Single {
            label: self.labels[best].clone(),
            confidence: self.probabilities[best],
        }
    }

    /// Historical multi-label output: labels at or above `threshold`, in the
    /// caller's order; the first maximum if none pass.
    pub fn select_multi(&self, threshold: f32) -> ClassificationOutput {
        let mut chosen: Vec<_> = self
            .labels
            .iter()
            .cloned()
            .zip(self.probabilities.iter().copied())
            .filter(|(_, probability)| *probability >= threshold)
            .collect();
        if chosen.is_empty() {
            let best = self.argmax();
            chosen.push((self.labels[best].clone(), self.probabilities[best]));
        }
        ClassificationOutput::Multi { labels: chosen }
    }
}

/// Reject a task name, label, instruction or description that would be read
/// as prompt structure.
pub fn check_reserved(field: &'static str, value: &str) -> Result<(), ScoresError> {
    if RESERVED_MARKERS.iter().any(|marker| value.contains(marker)) {
        return Err(ScoresError::ReservedMarker {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
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

pub(crate) fn first_argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn scores(logits: &[f32], temperature: f32, activation: Activation) -> ClassificationScores {
        let names: Vec<String> = (0..logits.len()).map(|index| format!("l{index}")).collect();
        ClassificationScores::new("task", names, logits.to_vec(), temperature, activation).unwrap()
    }

    #[test]
    fn softmax_keeps_order_and_sums_to_one() {
        let result = scores(&[2.0, -1.0, 0.5, -30.0], 1.0, Activation::Softmax);
        assert_eq!(result.labels(), labels(&["l0", "l1", "l2", "l3"]));
        assert_eq!(result.raw_logits(), [2.0, -1.0, 0.5, -30.0]);
        assert!(result.is_categorical());
        let sum: f32 = result.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(
            result.probabilities()[3] > 0.0,
            "tiny probabilities are kept"
        );
        assert_eq!(result.argmax(), 0);
    }

    #[test]
    fn sigmoid_is_independent_and_not_renormalized() {
        let result = scores(&[3.0, 3.0], 1.0, Activation::Sigmoid);
        assert!(!result.is_categorical());
        let sum: f32 = result.probabilities().iter().sum();
        assert!(sum > 1.5, "two confident sigmoids sum above one: {sum}");
        assert_eq!(result.probabilities()[0], result.probabilities()[1]);
    }

    #[test]
    fn temperature_divides_before_activation() {
        let hot = scores(&[2.0, 0.0], 2.0, Activation::Softmax);
        let cold = scores(&[1.0, 0.0], 1.0, Activation::Softmax);
        assert_eq!(hot.scaled_logits(), [1.0, 0.0]);
        assert_eq!(hot.probabilities(), cold.probabilities());
    }

    #[test]
    fn extreme_logits_stay_finite() {
        let result = scores(&[1000.0, -1000.0, 999.0], 1.0, Activation::Softmax);
        assert!(result.probabilities().iter().all(|value| value.is_finite()));
        let result = scores(&[1000.0, -1000.0], 1.0, Activation::Sigmoid);
        assert_eq!(result.probabilities(), [1.0, 0.0]);
    }

    #[test]
    fn ties_select_the_first_label() {
        let result = scores(&[0.5, 0.5, 0.5], 1.0, Activation::Softmax);
        assert_eq!(result.argmax(), 0);
        assert_eq!(
            result.select_single(),
            ClassificationOutput::Single {
                label: "l0".into(),
                confidence: result.probabilities()[0]
            }
        );
    }

    #[test]
    fn multi_selection_keeps_order_and_falls_back_to_first_max() {
        let result = scores(&[-2.0, 2.0, 1.0], 1.0, Activation::Sigmoid);
        match result.select_multi(0.5) {
            ClassificationOutput::Multi { labels } => {
                let names: Vec<_> = labels.iter().map(|(label, _)| label.as_str()).collect();
                assert_eq!(names, ["l1", "l2"]);
            }
            other => panic!("unexpected {other:?}"),
        }
        match result.select_multi(0.999) {
            ClassificationOutput::Multi { labels } => assert_eq!(labels[0].0, "l1"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_inputs_with_context() {
        let make = |names: &[&str], logits: &[f32], temperature: f32| {
            ClassificationScores::new(
                "task",
                labels(names),
                logits.to_vec(),
                temperature,
                Activation::Softmax,
            )
        };
        assert_eq!(make(&[], &[], 1.0).unwrap_err(), ScoresError::EmptyLabels);
        assert_eq!(
            make(&["a"], &[1.0, 2.0], 1.0).unwrap_err(),
            ScoresError::LengthMismatch {
                labels: 1,
                logits: 2
            }
        );
        assert_eq!(
            make(&["a", "a"], &[1.0, 2.0], 1.0).unwrap_err(),
            ScoresError::DuplicateLabel("a".into())
        );
        assert_eq!(
            make(&["a"], &[1.0], 0.0).unwrap_err(),
            ScoresError::InvalidTemperature(0.0)
        );
        assert!(matches!(
            make(&["a", "b"], &[1.0, f32::NAN], 1.0).unwrap_err(),
            ScoresError::NonFiniteLogit { index: 1, .. }
        ));
    }

    #[test]
    fn reserved_markers_are_rejected() {
        assert!(check_reserved("label", "positive").is_ok());
        let error = check_reserved("label", "x [L] y").unwrap_err();
        assert!(error.to_string().contains("reserved prompt marker"));
        assert!(check_reserved("instruction", "see [DESCRIPTION]").is_err());
    }

    #[test]
    fn request_validation_rejects_structural_problems() {
        let ok =
            ClassificationRequest::new("intent", labels(&["buy", "sell"]), Activation::Softmax)
                .with_instruction("Pick the user's goal.")
                .with_label_descriptions(vec![("buy".into(), "wants to purchase".into())]);
        ok.validate().unwrap();
        let bad =
            ClassificationRequest::new("intent", labels(&["buy", "buy"]), Activation::Softmax);
        assert_eq!(
            bad.validate().unwrap_err(),
            ScoresError::DuplicateLabel("buy".into())
        );
        let bad = ClassificationRequest::new("intent", labels(&["buy"]), Activation::Softmax)
            .with_label_descriptions(vec![("sell".into(), "x".into())]);
        assert_eq!(
            bad.validate().unwrap_err(),
            ScoresError::UnknownDescription("sell".into())
        );
        let bad = ClassificationRequest::new("in[P]tent", labels(&["buy"]), Activation::Softmax);
        assert!(matches!(
            bad.validate().unwrap_err(),
            ScoresError::ReservedMarker { field: "task", .. }
        ));
        let bad = ClassificationRequest::new("intent", vec![], Activation::Softmax);
        assert_eq!(bad.validate().unwrap_err(), ScoresError::EmptyLabels);
    }

    #[test]
    fn from_class_act_matches_upstream_auto_rule() {
        assert_eq!(
            Activation::from_class_act(ClassAct::Auto, true),
            Activation::Sigmoid
        );
        assert_eq!(
            Activation::from_class_act(ClassAct::Auto, false),
            Activation::Softmax
        );
        assert_eq!(
            Activation::from_class_act(ClassAct::Sigmoid, false),
            Activation::Sigmoid
        );
        assert_eq!(
            Activation::from_class_act(ClassAct::Softmax, true),
            Activation::Softmax
        );
    }
}
