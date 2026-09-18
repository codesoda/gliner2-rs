use std::collections::BTreeMap;

use crate::entities::FormattedEntitySpan;

#[derive(Debug, Clone, PartialEq)]
pub struct FormattedRelationPair {
    pub head: FormattedEntitySpan,
    pub tail: FormattedEntitySpan,
}

pub type FormattedRelationExtraction = BTreeMap<String, Vec<FormattedRelationPair>>;

/// Mirrors Python default output: tuples `(head, tail)` per relation type.
pub type RelationExtraction = BTreeMap<String, Vec<(String, String)>>;

pub fn build_relation_schema_tokens(relation: &str, description: Option<&str>) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let prompt_str = match description {
        Some(d) if !d.is_empty() => format!("{relation}: {d}"),
        _ => relation.to_string(),
    };
    tokens.push(prompt_str);

    tokens.push("(".to_string());
    tokens.push("[R]".to_string());
    tokens.push("head".to_string());
    tokens.push("[R]".to_string());
    tokens.push("tail".to_string());
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}
