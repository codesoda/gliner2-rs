use std::collections::BTreeMap;

use anyhow::anyhow;

use crate::{
    Result,
    entities::FormattedEntityValue,
    schema_spec::{FieldDtype, StructureFieldSpec},
};

pub type JsonRecord = BTreeMap<String, FormattedEntityValue>;
pub type JsonExtraction = BTreeMap<String, Vec<JsonRecord>>;

#[derive(Debug, Clone, PartialEq)]
pub struct JsonStructureSpec {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct JsonSchema {
    pub structures: Vec<JsonStructureSpec>,
}

impl JsonSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn structure(mut self, name: impl Into<String>, fields: Vec<String>) -> Self {
        self.structures.push(JsonStructureSpec {
            name: name.into(),
            fields,
        });
        self
    }
}

/// Parse a single `extract_json()` field spec string into a `StructureFieldSpec`.
///
/// Mirrors the tutorial format:
/// - `"field"` defaults to `dtype=list`
/// - `"field::str"` / `"field::list"`
/// - `"field::[a|b|c]"` defaults to `dtype=str`
/// - `"field::<type>::<description>"`
/// - `"field::[a|b|c]::<type>::<description>"`
/// - `"field::<description>"` defaults to `dtype=list`
pub fn parse_field_spec(field_spec: &str) -> Result<StructureFieldSpec> {
    let raw = field_spec.trim();
    if raw.is_empty() {
        return Err(anyhow!("field spec is empty"));
    }

    let parts: Vec<&str> = raw.split("::").collect();
    let field_name = parts
        .first()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| anyhow!("field spec '{raw}' has an empty field name"))?;

    let mut dtype = FieldDtype::List;
    let mut saw_dtype = false;
    let mut choices: Vec<String> = Vec::new();
    let mut description_parts: Vec<&str> = Vec::new();

    for part in parts.iter().skip(1) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        match part.to_ascii_lowercase().as_str() {
            "str" => {
                dtype = FieldDtype::Str;
                saw_dtype = true;
                continue;
            }
            "list" => {
                dtype = FieldDtype::List;
                saw_dtype = true;
                continue;
            }
            _ => {}
        }

        if part.starts_with('[') && part.ends_with(']') {
            let inner = &part[1..part.len() - 1];
            let parsed: Vec<String> = inner
                .split('|')
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .map(|c| c.to_string())
                .collect();
            if parsed.is_empty() {
                return Err(anyhow!("choices part '{part}' is empty in '{raw}'"));
            }
            choices = parsed;
            continue;
        }

        description_parts.push(part);
    }

    // Per tutorial: if choices are present and no explicit dtype was provided, default to `str`.
    if !choices.is_empty() && !saw_dtype {
        dtype = FieldDtype::Str;
    }

    let mut spec = StructureFieldSpec::new(field_name)
        .dtype(dtype)
        .choices(choices);
    if !description_parts.is_empty() {
        spec = spec.description(description_parts.join("::"));
    }
    Ok(spec)
}
