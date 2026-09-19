use std::collections::HashSet;

use crate::schema_spec::FieldDtype;

#[derive(Debug, Clone, PartialEq)]
pub struct EntitySpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityMatches {
    pub label: String,
    pub spans: Vec<EntitySpan>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FormattedEntitySpan {
    Text(String),
    TextWithConfidence {
        text: String,
        confidence: f32,
    },
    TextWithSpans {
        text: String,
        start: usize,
        end: usize,
    },
    TextWithConfidenceAndSpans {
        text: String,
        confidence: f32,
        start: usize,
        end: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FormattedEntityValue {
    List(Vec<FormattedEntitySpan>),
    Single(Option<FormattedEntitySpan>),
}

impl EntitySpan {
    pub fn format(&self, include_confidence: bool, include_spans: bool) -> FormattedEntitySpan {
        match (include_confidence, include_spans) {
            (false, false) => FormattedEntitySpan::Text(self.text.clone()),
            (true, false) => FormattedEntitySpan::TextWithConfidence {
                text: self.text.clone(),
                confidence: self.score,
            },
            (false, true) => FormattedEntitySpan::TextWithSpans {
                text: self.text.clone(),
                start: self.start,
                end: self.end,
            },
            (true, true) => FormattedEntitySpan::TextWithConfidenceAndSpans {
                text: self.text.clone(),
                confidence: self.score,
                start: self.start,
                end: self.end,
            },
        }
    }
}

pub fn format_entity_spans(
    spans: &[EntitySpan],
    dtype: FieldDtype,
    include_confidence: bool,
    include_spans: bool,
) -> FormattedEntityValue {
    // Match Python's `format_results()` behavior: dedupe by case-insensitive text.
    let mut seen = HashSet::<String>::new();
    let mut unique: Vec<&EntitySpan> = Vec::new();
    for span in spans {
        let key = span.text.to_ascii_lowercase();
        if span.text.is_empty() || !seen.insert(key) {
            continue;
        }
        unique.push(span);
    }

    match dtype {
        FieldDtype::List => FormattedEntityValue::List(
            unique
                .into_iter()
                .map(|s| s.format(include_confidence, include_spans))
                .collect(),
        ),
        FieldDtype::Str => FormattedEntityValue::Single(
            unique
                .first()
                .map(|s| s.format(include_confidence, include_spans)),
        ),
    }
}

pub fn build_entities_schema_tokens(labels: &[String], prompt: Option<&str>) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let prompt_str = match prompt {
        Some(p) if !p.is_empty() => format!("entities: {p}"),
        _ => "entities".to_string(),
    };
    tokens.push(prompt_str);

    tokens.push("(".to_string());
    for label in labels {
        tokens.push("[E]".to_string());
        tokens.push(label.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

pub fn build_entities_schema_tokens_with_descriptions(
    labels: &[String],
    prompt: Option<&str>,
    label_descriptions: &[(String, String)],
) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let mut prompt_str = match prompt {
        Some(p) if !p.is_empty() => format!("entities: {p}"),
        _ => "entities".to_string(),
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
        tokens.push("[E]".to_string());
        tokens.push(label.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}
