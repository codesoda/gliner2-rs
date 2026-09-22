use std::{collections::BTreeSet, env, fs, path::PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::{
    boundary::BoundaryPipeline,
    classification::FormattedClassification,
    entities::{FormattedEntitySpan, FormattedEntityValue},
    schema_spec::{
        ExtractionResult, FieldDtype, RelationSpec, SchemaBuilder, SchemaSpec, StructureFieldSpec,
    },
};
use serde_json::{Map, Value, json};

const CASE_IDS: [&str; 2] = ["unicode_relations", "mixed_choice_relations"];
const CASES_SHA256: &str = "629e92b96e4755efc2f7f64de9c0eb5fac09484cb235df6180c6b3ad7ba56c7a";
const SOURCE_HASHES: [(&str, &str); 4] = [
    (
        "config.json",
        "0eb92d00584d613aab32b2178f84a85176b62c87ae3689ce9084e83f6eba64d1",
    ),
    (
        "encoder_config/config.json",
        "d36a845b9f25dcaf1ec45a1c4bdf65ea4ac20596537e14530ec9f660a63aeca4",
    ),
    (
        "model.safetensors",
        "7274094de2e0c2a37a386f55fc4e23061a954da5bd7a335e7dfe56f2743c277a",
    ),
    (
        "tokenizer.json",
        "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
    ),
];

fn root() -> PathBuf {
    env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn strict() -> bool {
    env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1")
        || env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1")
}

fn fixture_path() -> PathBuf {
    root()
        .join("fixtures/gliner2.5-base-v1/relation-aux")
        .join("relation-pipeline-vectors.json")
}

fn load_fixture() -> Result<Option<Value>> {
    let path = fixture_path();
    if !path.is_file() {
        let message = format!("missing relation pipeline fixture {}", path.display());
        if strict() {
            return Err(anyhow!(message));
        }
        eprintln!("SKIP: {message}");
        return Ok(None);
    }
    Ok(Some(
        serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("invalid fixture {}", path.display()))?,
    ))
}

fn bundle() -> Result<Option<PathBuf>> {
    let bundle = root().join("onnx/gliner2.5-base-v1");
    let metadata = ["config.json", "tokenizer.json"];
    let graphs = [
        "encoder.onnx",
        "classifier.onnx",
        "boundary_marginals.onnx",
        "boundary_scorer.onnx",
        "boundary_explicit_scorer.onnx",
        "boundary_records.onnx",
        "boundary_relations.onnx",
    ];
    ensure!(
        graphs.len() == 7,
        "boundary bundle graph list must stay exact"
    );
    let missing: Vec<_> = metadata
        .into_iter()
        .chain(graphs)
        .map(|name| bundle.join(name))
        .filter(|path| !path.is_file())
        .collect();
    if missing.is_empty() {
        return Ok(Some(bundle));
    }
    let message = format!(
        "missing relation mixed boundary bundle files: {}",
        missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if strict() {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn object_keys(value: &Value) -> Result<BTreeSet<&str>> {
    Ok(value
        .as_object()
        .context("expected JSON object")?
        .keys()
        .map(String::as_str)
        .collect())
}

fn validate_expected_spans(value: &Value, text: &str, path: &str) -> Result<()> {
    match value {
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_expected_spans(value, text, &format!("{path}[{index}]"))?;
            }
        }
        Value::Object(object) => {
            let start = object.get("start");
            let end = object.get("end");
            ensure!(
                start.is_some() == end.is_some(),
                "{path}: incomplete span coordinates"
            );
            if let (Some(start), Some(end)) = (start, end) {
                let start = start.as_u64().context("span start")? as usize;
                let end = end.as_u64().context("span end")? as usize;
                ensure!(
                    object.get("offset_unit").and_then(Value::as_str) == Some("utf8_byte"),
                    "{path}: expected UTF-8 offset unit"
                );
                let surface = object
                    .get("text")
                    .and_then(Value::as_str)
                    .context("span text")?;
                ensure!(
                    text.get(start..end) == Some(surface),
                    "{path}: [{start},{end}) does not slice original text to {surface:?}"
                );
            }
            for (key, value) in object {
                validate_expected_spans(value, text, &format!("{path}.{key}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_fixture(fixture: &Value) -> Result<&Vec<Value>> {
    ensure!(fixture["format_version"] == 1);
    ensure!(fixture["case_count"] == 2);
    ensure!(fixture["cases_sha256"] == CASES_SHA256);
    let provenance = &fixture["provenance"];
    ensure!(
        provenance["gliner2_repository"] == "https://github.com/fastino-ai/GLiNER2"
            && provenance["gliner2_commit"] == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
            && provenance["model_id"] == "fastino/gliner2.5-base-v1"
            && provenance["hf_revision"] == "78cea040597df251eedefa9d7ee2a756af39fe64"
            && provenance["architecture"] == "boundary"
            && provenance["architecture_version"] == 1
            && provenance["device"] == "cpu"
            && provenance["dtype"] == "float32"
            && provenance["autocast"] == false
    );
    let source_hashes = provenance["source_file_sha256"]
        .as_object()
        .context("source hashes")?;
    ensure!(source_hashes.len() == SOURCE_HASHES.len());
    for (name, hash) in SOURCE_HASHES {
        ensure!(source_hashes.get(name).and_then(Value::as_str) == Some(hash));
    }

    let cases = fixture["cases"].as_array().context("fixture cases")?;
    ensure!(cases.len() == 2, "expected exactly two auxiliary cases");
    let ids = cases
        .iter()
        .map(|case| case["case_id"].as_str().context("case ID"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(ids == CASE_IDS, "auxiliary IDs differ: {ids:?}");

    let extract_arguments = json!({
        "threshold": 0.3,
        "format_results": true,
        "include_confidence": true,
        "include_spans": true,
        "max_len": 4096
    });
    ensure!(
        cases[0]["original_text"]
            == "Zoë works for Café Labs in São Paulo. 李雷 works for 東京研究所."
            && cases[0]["source_schema_call_order"] == json!(["relations"])
            && cases[0]["source_schema_arguments"] == json!({"relations": [{"name": "works for"}]})
            && cases[0]["extract_arguments"] == extract_arguments
    );
    ensure!(
        cases[1]["source_schema_call_order"]
            == json!(["structures", "entities", "relations", "classifications"])
    );
    ensure!(
        cases[1]["source_schema_arguments"]
            == json!({
                "structures": [{
                    "name": "metadata",
                    "fields": [{
                        "name": "kind",
                        "dtype": "str",
                        "choices": ["employment", "other"]
                    }]
                }],
                "entities": ["person", "organization"],
                "relations": [{
                    "name": "works for",
                    "description": "employment relationship"
                }],
                "classifications": [{
                    "task": "sentiment",
                    "labels": ["positive", "negative"]
                }]
            })
            && cases[1]["extract_arguments"] == extract_arguments
            && cases[1]["original_text"] == "Alice works for Acme Corp."
    );

    for case in cases {
        let id = case["case_id"].as_str().context("case ID")?;
        let text = case["original_text"].as_str().context("original text")?;
        let expected = &case["upstream_formatted_result_utf8_offsets"];
        validate_expected_spans(expected, text, id)?;
        ensure!(
            expected["relation_extraction"]["works for"]
                .as_array()
                .is_some_and(|pairs| !pairs.is_empty()),
            "{id}: relation oracle must be nonempty"
        );
    }
    ensure!(!cases[0]["original_text"].as_str().unwrap().is_ascii());
    ensure!(
        object_keys(&cases[0]["upstream_formatted_result_utf8_offsets"])?
            == BTreeSet::from(["relation_extraction"])
    );
    ensure!(
        object_keys(&cases[1]["upstream_formatted_result_utf8_offsets"])?
            == BTreeSet::from(["entities", "metadata", "relation_extraction", "sentiment"]),
        "mixed oracle must contain every task family"
    );
    Ok(cases)
}

fn schema_for(case_id: &str) -> Result<SchemaSpec> {
    match case_id {
        "unicode_relations" => Ok(SchemaBuilder::new()
            .relations(vec![RelationSpec::new("works for")])
            .build()),
        "mixed_choice_relations" => Ok(SchemaBuilder::new()
            .structure("metadata")
            .field(
                StructureFieldSpec::new("kind")
                    .dtype(FieldDtype::Str)
                    .choices(vec!["employment".to_owned(), "other".to_owned()]),
            )
            .finish()
            .entities(vec!["person".to_owned(), "organization".to_owned()])
            .relations(vec![
                RelationSpec::new("works for").description("employment relationship"),
            ])
            .classification(
                "sentiment",
                vec!["positive".to_owned(), "negative".to_owned()],
            )
            .build()),
        other => Err(anyhow!("unexpected relation auxiliary case {other:?}")),
    }
}

fn span_value(span: &FormattedEntitySpan, source: &str, path: &str) -> Result<Value> {
    Ok(match span {
        FormattedEntitySpan::Text(text) => Value::String(text.clone()),
        FormattedEntitySpan::TextWithConfidence { text, confidence } => {
            json!({"text": text, "confidence": confidence})
        }
        FormattedEntitySpan::TextWithSpans { text, start, end } => {
            ensure!(
                source.get(*start..*end) == Some(text),
                "{path}: Rust byte coordinates do not slice original text"
            );
            json!({"text": text, "start": start, "end": end, "offset_unit": "utf8_byte"})
        }
        FormattedEntitySpan::TextWithConfidenceAndSpans {
            text,
            confidence,
            start,
            end,
        } => {
            ensure!(
                source.get(*start..*end) == Some(text),
                "{path}: Rust byte coordinates do not slice original text"
            );
            json!({
                "text": text,
                "confidence": confidence,
                "start": start,
                "end": end,
                "offset_unit": "utf8_byte"
            })
        }
    })
}

fn entity_value(value: &FormattedEntityValue, source: &str, path: &str) -> Result<Value> {
    match value {
        FormattedEntityValue::List(spans) => Ok(Value::Array(
            spans
                .iter()
                .enumerate()
                .map(|(index, span)| span_value(span, source, &format!("{path}[{index}]")))
                .collect::<Result<Vec<_>>>()?,
        )),
        FormattedEntityValue::Single(None) => Ok(Value::Null),
        FormattedEntityValue::Single(Some(span)) => span_value(span, source, path),
    }
}

fn classification_value(value: &FormattedClassification) -> Value {
    match value {
        FormattedClassification::Single(label) => Value::String(label.clone()),
        FormattedClassification::SingleWithConfidence { label, confidence } => {
            json!({"label": label, "confidence": confidence})
        }
        FormattedClassification::Multi(labels) => {
            Value::Array(labels.iter().cloned().map(Value::String).collect())
        }
        FormattedClassification::MultiWithConfidence(labels) => Value::Array(
            labels
                .iter()
                .map(|(label, confidence)| json!({"label": label, "confidence": confidence}))
                .collect(),
        ),
    }
}

fn comparable_result(actual: &ExtractionResult, source: &str) -> Result<Value> {
    let mut result = Map::new();
    for (structure, records) in &actual.structures {
        let records = records
            .iter()
            .enumerate()
            .map(|(record_index, record)| {
                let fields = record
                    .iter()
                    .map(|(field, value)| {
                        Ok((
                            field.clone(),
                            entity_value(
                                value,
                                source,
                                &format!("{structure}[{record_index}].{field}"),
                            )?,
                        ))
                    })
                    .collect::<Result<Map<String, Value>>>()?;
                Ok(Value::Object(fields))
            })
            .collect::<Result<Vec<_>>>()?;
        result.insert(structure.clone(), Value::Array(records));
    }
    if !actual.entities.is_empty() {
        let entities = actual
            .entities
            .iter()
            .map(|(label, value)| {
                Ok((
                    label.clone(),
                    entity_value(value, source, &format!("entities.{label}"))?,
                ))
            })
            .collect::<Result<Map<String, Value>>>()?;
        result.insert("entities".to_owned(), Value::Object(entities));
    }
    if !actual.relations.is_empty() {
        let relations = actual
            .relations
            .iter()
            .map(|(name, pairs)| {
                let pairs = pairs
                    .iter()
                    .enumerate()
                    .map(|(index, pair)| {
                        Ok(json!({
                            "head": span_value(
                                &pair.head,
                                source,
                                &format!("relation_extraction.{name}[{index}].head"),
                            )?,
                            "tail": span_value(
                                &pair.tail,
                                source,
                                &format!("relation_extraction.{name}[{index}].tail"),
                            )?,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((name.clone(), Value::Array(pairs)))
            })
            .collect::<Result<Map<String, Value>>>()?;
        result.insert("relation_extraction".to_owned(), Value::Object(relations));
    }
    for (task, value) in &actual.classifications {
        result.insert(task.clone(), classification_value(value));
    }
    Ok(Value::Object(result))
}

fn compare_value(
    actual: &Value,
    expected: &Value,
    path: &str,
    max_confidence_error: &mut f64,
) -> Result<()> {
    if path.ends_with(".confidence") {
        let actual = actual.as_f64().context("actual confidence")?;
        let expected = expected.as_f64().context("expected confidence")?;
        let error = (actual - expected).abs();
        *max_confidence_error = max_confidence_error.max(error);
        ensure!(
            actual.is_finite() && error <= 1e-3,
            "{path}: confidence {actual} != {expected}; abs error={error:e}"
        );
        return Ok(());
    }
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            let actual_keys: BTreeSet<_> = actual.keys().collect();
            let expected_keys: BTreeSet<_> = expected.keys().collect();
            ensure!(
                actual_keys == expected_keys,
                "{path}: keys differ: actual={actual_keys:?}, expected={expected_keys:?}"
            );
            for (key, expected) in expected {
                compare_value(
                    actual.get(key).context("key checked above")?,
                    expected,
                    &format!("{path}.{key}"),
                    max_confidence_error,
                )?;
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            ensure!(
                actual.len() == expected.len(),
                "{path}: array length {} != {}",
                actual.len(),
                expected.len()
            );
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare_value(
                    actual,
                    expected,
                    &format!("{path}[{index}]"),
                    max_confidence_error,
                )?;
            }
        }
        _ => ensure!(actual == expected, "{path}: {actual} != {expected}"),
    }
    Ok(())
}

#[test]
fn relation_auxiliary_fixture_has_exact_model_free_contract() -> Result<()> {
    let Some(fixture) = load_fixture()? else {
        return Ok(());
    };
    validate_fixture(&fixture)?;
    Ok(())
}

#[test]
fn two_relation_pipeline_oracles_match_every_task_family() -> Result<()> {
    let Some(fixture) = load_fixture()? else {
        return Ok(());
    };
    let cases = validate_fixture(&fixture)?;
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let mut tested = BTreeSet::new();
    let mut maximum_error = 0.0_f64;
    for case in cases {
        let case_id = case["case_id"].as_str().context("case ID")?;
        ensure!(tested.insert(case_id), "duplicate tested case {case_id}");
        let source = case["original_text"].as_str().context("original text")?;
        let threshold = case["extract_arguments"]["threshold"]
            .as_f64()
            .context("threshold")? as f32;
        let actual =
            pipeline.extract_with_confidence_and_spans(source, &schema_for(case_id)?, threshold)?;
        let actual = comparable_result(&actual, source)?;
        compare_value(
            &actual,
            &case["upstream_formatted_result_utf8_offsets"],
            case_id,
            &mut maximum_error,
        )?;
    }
    ensure!(
        tested == BTreeSet::from(CASE_IDS),
        "tested auxiliary IDs differ: {tested:?}"
    );
    eprintln!("relation mixed public-oracle maximum confidence absolute error={maximum_error:e}");
    Ok(())
}
