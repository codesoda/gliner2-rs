#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationOutput {
    Single { label: String, confidence: f32 },
    Multi { labels: Vec<(String, f32)> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassAct {
    Sigmoid,
    Softmax,
    #[default]
    Auto,
}

impl ClassAct {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sigmoid" => Some(Self::Sigmoid),
            "softmax" => Some(Self::Softmax),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FormattedClassification {
    Single(String),
    SingleWithConfidence { label: String, confidence: f32 },
    Multi(Vec<String>),
    MultiWithConfidence(Vec<(String, f32)>),
}

impl ClassificationOutput {
    pub fn format(&self, include_confidence: bool) -> FormattedClassification {
        match self {
            Self::Single { label, confidence } => {
                if include_confidence {
                    FormattedClassification::SingleWithConfidence {
                        label: label.clone(),
                        confidence: *confidence,
                    }
                } else {
                    FormattedClassification::Single(label.clone())
                }
            }
            Self::Multi { labels } => {
                if include_confidence {
                    FormattedClassification::MultiWithConfidence(labels.clone())
                } else {
                    FormattedClassification::Multi(labels.iter().map(|(l, _)| l.clone()).collect())
                }
            }
        }
    }
}

pub fn build_classification_schema_tokens(
    task: &str,
    labels: &[String],
    prompt: Option<&str>,
) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let prompt_str = match prompt {
        Some(p) if !p.is_empty() => format!("{task}: {p}"),
        _ => task.to_string(),
    };
    tokens.push(prompt_str);

    tokens.push("(".to_string());
    for label in labels {
        tokens.push("[L]".to_string());
        tokens.push(label.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

pub fn build_classification_schema_tokens_with_descriptions(
    task: &str,
    labels: &[String],
    prompt: Option<&str>,
    label_descriptions: &[(String, String)],
) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let mut prompt_str = match prompt {
        Some(p) if !p.is_empty() => format!("{task}: {p}"),
        _ => task.to_string(),
    };

    // Match Python `SchemaTransformer.transform_schema()` behavior:
    // append `[DESCRIPTION] label: description` in the given order.
    for (label, desc) in label_descriptions {
        if labels.iter().any(|l| l == label) {
            prompt_str.push_str(" [DESCRIPTION] ");
            prompt_str.push_str(label);
            prompt_str.push_str(": ");
            prompt_str.push_str(desc);
        }
    }

    tokens.push(prompt_str);

    tokens.push("(".to_string());
    for label in labels {
        tokens.push("[L]".to_string());
        tokens.push(label.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

pub fn decode_classification(
    labels: &[String],
    logits: &[f32],
    multi_label: bool,
    cls_threshold: f32,
    class_act: ClassAct,
) -> ClassificationOutput {
    assert_eq!(
        labels.len(),
        logits.len(),
        "labels and logits length must match"
    );

    let probs: Vec<f32> = match class_act {
        ClassAct::Sigmoid => logits.iter().copied().map(sigmoid).collect(),
        ClassAct::Softmax => softmax(logits),
        ClassAct::Auto => {
            if multi_label {
                logits.iter().copied().map(sigmoid).collect()
            } else {
                softmax(logits)
            }
        }
    };

    if multi_label {
        let mut chosen: Vec<(String, f32)> = labels
            .iter()
            .cloned()
            .zip(probs.iter().copied())
            .filter(|(_, p)| *p >= cls_threshold)
            .collect();

        if chosen.is_empty() && !probs.is_empty() {
            let (best_idx, best_p) = probs
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(&b))
                .unwrap();
            chosen.push((labels[best_idx].clone(), best_p));
        }

        chosen.sort_by(|a, b| b.1.total_cmp(&a.1));
        ClassificationOutput::Multi { labels: chosen }
    } else {
        if probs.is_empty() {
            return ClassificationOutput::Single {
                label: String::new(),
                confidence: 0.0,
            };
        }
        let (best_idx, best_p) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(&b))
            .unwrap();
        ClassificationOutput::Single {
            label: labels[best_idx].clone(),
            confidence: best_p,
        }
    }
}
