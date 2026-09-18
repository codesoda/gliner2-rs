use crate::schema_spec::StructureSpec;

pub fn build_structure_schema_tokens(spec: &StructureSpec, prompt: Option<&str>) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("(".to_string());
    tokens.push("[P]".to_string());

    let mut prompt_str = match prompt {
        Some(p) if !p.is_empty() => format!("{}: {p}", spec.name),
        _ => spec.name.clone(),
    };

    // Match Python `SchemaTransformer.transform_schema()` behavior:
    // append `[DESCRIPTION] field: description` in field order.
    for field in &spec.fields {
        if let Some(desc) = field.description.as_ref() {
            prompt_str.push_str(" [DESCRIPTION] ");
            prompt_str.push_str(&field.name);
            prompt_str.push_str(": ");
            prompt_str.push_str(desc);
        }
    }

    tokens.push(prompt_str);

    tokens.push("(".to_string());
    for field in &spec.fields {
        tokens.push("[C]".to_string());
        tokens.push(field.name.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

pub fn build_structure_choice_prefix(spec: &StructureSpec) -> Vec<String> {
    let classification_fields: Vec<_> = spec
        .fields
        .iter()
        .filter(|f| !f.choices.is_empty())
        .collect();
    if classification_fields.is_empty() {
        return Vec::new();
    }

    let mut inner: Vec<String> = Vec::new();
    for field in classification_fields {
        inner.push(field.name.clone());
        inner.push("(".to_string());
        for (idx, choice) in field.choices.iter().enumerate() {
            inner.push(choice.clone());
            if idx + 1 < field.choices.len() {
                inner.push("|".to_string());
            }
        }
        inner.push(")".to_string());
        inner.push(",".to_string());
    }
    if inner.last().is_some_and(|t| t == ",") {
        inner.pop();
    }

    let mut prefix: Vec<String> = Vec::new();
    prefix.push("(".to_string());
    prefix.push(format!("{}:", spec.name));
    prefix.extend(inner);
    prefix.push(")".to_string());
    prefix
}

