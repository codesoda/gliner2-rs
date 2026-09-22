use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use gliner2_rs::{
    boundary::{
        BoundaryPipeline,
        decode::OverlapPolicy,
        record_decode::Cardinality,
        record_schema::{RecordConfig, RecordFieldOptions, RecordMetadata},
    },
    classification::{ClassAct, FormattedClassification},
    entities::{FormattedEntitySpan, FormattedEntityValue},
    schema_spec::{
        EntitySpec, FieldDtype, RelationSpec, SchemaSpec, StructureFieldSpec, StructureSpec,
    },
};
use serde_json::{Map, Value, json};

const GRAPH_FILES: [&str; 7] = [
    "encoder.onnx",
    "classifier.onnx",
    "boundary_marginals.onnx",
    "boundary_scorer.onnx",
    "boundary_explicit_scorer.onnx",
    "boundary_records.onnx",
    "boundary_relations.onnx",
];
const CASE_IDS: [&str; 30] = [
    "ner_simple",
    "ner_duplicate_mentions",
    "ner_nested_overlap",
    "ner_crossing_candidates",
    "ner_long_span",
    "ner_scalar_and_list",
    "ner_email_product",
    "ner_repeated_person",
    "classification_sentiment_positive",
    "classification_sentiment_negative",
    "classification_intent",
    "classification_multilabel",
    "classification_multi_task",
    "json_natural_people",
    "json_natural_choice",
    "json_legacy_event",
    "json_latent_products",
    "json_anchorless_list",
    "relation_employment",
    "relation_founded",
    "relation_location",
    "relation_multiple_types",
    "unicode_combining_emoji",
    "unicode_cjk",
    "unicode_japanese_arabic",
    "long_1000_words",
    "long_2000_words",
    "long_3000_words",
    "edge_punctuation_empty_schema",
    "edge_empty_text",
];
const ADDITIONAL_IDS: [&str; 2] = ["mixed_unicode_all_tasks", "explicit_unicode_duplicates"];

struct Profile {
    name: &'static str,
    model_id: &'static str,
    revision: &'static str,
    bundle_name: &'static str,
}

const PROFILES: [Profile; 3] = [
    Profile {
        name: "small",
        model_id: "fastino/gliner2.5-small-v1",
        revision: "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
        bundle_name: "gliner2.5-small-v1",
    },
    Profile {
        name: "base",
        model_id: "fastino/gliner2.5-base-v1",
        revision: "78cea040597df251eedefa9d7ee2a756af39fe64",
        bundle_name: "gliner2.5-base-v1",
    },
    Profile {
        name: "multi",
        model_id: "fastino/gliner2.5-multi-v1",
        revision: "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        bundle_name: "gliner2.5-multi-v1",
    },
];

struct Settings {
    profile: &'static Profile,
    graph_dir: PathBuf,
    fixture_dir: PathBuf,
    report_json: Option<PathBuf>,
}

#[derive(Default)]
struct Stats {
    cases: usize,
    maximum_confidence_abs: f64,
    maximum_explicit_logit_abs: f64,
    categories: BTreeSet<String>,
    task_families: BTreeSet<String>,
}

fn strict() -> bool {
    env::var("GLINER2_STRICT_BUNDLE_VALIDATION").as_deref() == Ok("1")
}

fn settings() -> Result<Option<Settings>> {
    let model = env::var("GLINER2_BUNDLE_MODEL").ok();
    let graph_dir = env::var_os("GLINER2_BUNDLE_GRAPH_DIR").map(PathBuf::from);
    let fixture_dir = env::var_os("GLINER2_BUNDLE_FIXTURE_DIR").map(PathBuf::from);
    let report_json = env::var_os("GLINER2_BUNDLE_REPORT_JSON").map(PathBuf::from);
    if model.is_none() && graph_dir.is_none() && fixture_dir.is_none() {
        if strict() {
            bail!(
                "strict bundle validation requires GLINER2_BUNDLE_MODEL, \
                 GLINER2_BUNDLE_GRAPH_DIR and GLINER2_BUNDLE_FIXTURE_DIR"
            );
        }
        eprintln!(
            "SKIP: set GLINER2_BUNDLE_MODEL plus explicit graph/fixture directories \
             to run heavyweight bundle parity"
        );
        return Ok(None);
    }
    let model = model.context("GLINER2_BUNDLE_MODEL is required when bundle paths are set")?;
    let profile = PROFILES
        .iter()
        .find(|profile| profile.name == model)
        .ok_or_else(|| {
            anyhow!("unknown bundle model ID {model:?}; expected small, base or multi")
        })?;
    let graph_dir = graph_dir.context("GLINER2_BUNDLE_GRAPH_DIR is required")?;
    let fixture_dir = fixture_dir.context("GLINER2_BUNDLE_FIXTURE_DIR is required")?;
    let missing: Vec<_> = ["config.json", "tokenizer.json", "export_manifest.json"]
        .into_iter()
        .chain(GRAPH_FILES)
        .map(|name| graph_dir.join(name))
        .filter(|path| !path.is_file())
        .collect();
    if !missing.is_empty() {
        let message = format!("bundle graph directory is incomplete: {missing:?}");
        if strict() {
            bail!(message);
        }
        eprintln!("SKIP: {message}");
        return Ok(None);
    }
    if !fixture_dir.join("manifest.json").is_file() {
        let message = format!(
            "bundle fixture directory lacks {}",
            fixture_dir.join("manifest.json").display()
        );
        if strict() {
            bail!(message);
        }
        eprintln!("SKIP: {message}");
        return Ok(None);
    }
    Ok(Some(Settings {
        profile,
        graph_dir,
        fixture_dir,
        report_json,
    }))
}

fn string_list(value: &Value, context: &str) -> Result<Vec<String>> {
    value
        .as_array()
        .with_context(|| context.to_owned())?
        .iter()
        .map(|item| {
            item.as_str()
                .with_context(|| format!("{context} item"))
                .map(str::to_owned)
        })
        .collect()
}

fn field_dtype(value: Option<&Value>) -> Result<FieldDtype> {
    match value.and_then(Value::as_str).unwrap_or("list") {
        "str" => Ok(FieldDtype::Str),
        "list" => Ok(FieldDtype::List),
        other => bail!("unknown field dtype {other:?}"),
    }
}

fn cardinality(value: &str) -> Result<Cardinality> {
    match value {
        "optional_one" => Ok(Cardinality::OptionalOne),
        "required_one" => Ok(Cardinality::RequiredOne),
        "zero_or_more" => Ok(Cardinality::ZeroOrMore),
        "one_or_more" => Ok(Cardinality::OneOrMore),
        other => bail!("unknown record cardinality {other:?}"),
    }
}

fn schema_and_records(specification: &Value) -> Result<(SchemaSpec, RecordMetadata)> {
    let mut schema = SchemaSpec::default();
    if let Some(labels) = specification.get("entities") {
        schema.entities = string_list(labels, "entity labels")?
            .into_iter()
            .map(EntitySpec::new)
            .collect();
    }
    if let Some(config) = specification
        .get("entities_config")
        .and_then(Value::as_object)
    {
        for (name, options) in config {
            let mut entity = EntitySpec::new(name).dtype(field_dtype(options.get("dtype"))?);
            if let Some(description) = options.get("description").and_then(Value::as_str) {
                entity = entity.description(description);
            }
            if let Some(threshold) = options.get("threshold").and_then(Value::as_f64) {
                entity = entity.threshold(threshold as f32);
            }
            schema.entities.push(entity);
        }
    }
    if let Some(tasks) = specification
        .get("classifications")
        .and_then(Value::as_array)
    {
        for task in tasks {
            let activation = task
                .get("class_act")
                .and_then(Value::as_str)
                .map(|value| {
                    ClassAct::parse(value)
                        .ok_or_else(|| anyhow!("unknown classification activation {value:?}"))
                })
                .transpose()?
                .unwrap_or_default();
            schema
                .classifications
                .push(gliner2_rs::schema_spec::ClassificationSpec {
                    task: task["task"]
                        .as_str()
                        .context("classification task name")?
                        .to_owned(),
                    labels: string_list(&task["labels"], "classification labels")?,
                    label_descriptions: Vec::new(),
                    multi_label: task
                        .get("multi_label")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    cls_threshold: task
                        .get("cls_threshold")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.5) as f32,
                    class_act: activation,
                    prompt: task
                        .get("prompt")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                });
        }
    }

    let mut records = RecordMetadata::new();
    if let Some(structures) = specification.get("structures").and_then(Value::as_array) {
        for structure in structures {
            let name = structure["name"]
                .as_str()
                .context("structure name")?
                .to_owned();
            let mut fields = Vec::new();
            let mut record = match structure.get("mode").and_then(Value::as_str) {
                None => None,
                Some("natural") => Some(RecordConfig::natural(
                    structure["anchor"]
                        .as_str()
                        .context("natural record anchor")?,
                )),
                Some("latent") => Some(RecordConfig::latent()),
                Some("anchorless") => Some(RecordConfig::anchorless()),
                Some(other) => bail!("unknown record mode {other:?}"),
            };
            for field in structure["fields"].as_array().context("structure fields")? {
                let field_name = field["name"].as_str().context("field name")?;
                let mut field_spec =
                    StructureFieldSpec::new(field_name).dtype(field_dtype(field.get("dtype"))?);
                if let Some(description) = field.get("description").and_then(Value::as_str) {
                    field_spec = field_spec.description(description);
                }
                if let Some(threshold) = field.get("threshold").and_then(Value::as_f64) {
                    field_spec = field_spec.threshold(threshold as f32);
                }
                if let Some(choices) = field.get("choices") {
                    field_spec = field_spec.choices(string_list(choices, "field choices")?);
                }
                fields.push(field_spec);
                if let Some(config) = record.take() {
                    let mut options = RecordFieldOptions::default().exclusive(
                        field
                            .get("exclusive")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    );
                    if let Some(value) = field.get("cardinality").and_then(Value::as_str) {
                        options = options.cardinality(cardinality(value)?);
                    }
                    record = Some(config.field(field_name, options));
                }
            }
            schema.structures.push(StructureSpec {
                name: name.clone(),
                fields,
            });
            if let Some(config) = record {
                ensure!(records.insert(name, config).is_none());
            }
        }
    }
    if let Some(relations) = specification.get("relations").and_then(Value::as_array) {
        for relation in relations {
            let spec = match relation {
                Value::String(name) => RelationSpec::new(name),
                Value::Object(object) => {
                    let name = object
                        .get("name")
                        .and_then(Value::as_str)
                        .context("relation name")?;
                    let mut spec = RelationSpec::new(name);
                    if let Some(description) = object.get("description").and_then(Value::as_str) {
                        spec = spec.description(description);
                    }
                    if let Some(threshold) = object.get("threshold").and_then(Value::as_f64) {
                        spec = spec.threshold(threshold as f32);
                    }
                    spec
                }
                _ => bail!("malformed relation specification"),
            };
            schema.relations.push(spec);
        }
    }
    Ok((schema, records))
}

fn span_value(span: &FormattedEntitySpan, source: &str, context: &str) -> Result<Value> {
    let FormattedEntitySpan::TextWithConfidenceAndSpans {
        text,
        confidence,
        start,
        end,
    } = span
    else {
        bail!("{context}: typed output omitted confidence or spans")
    };
    ensure!(
        confidence.is_finite(),
        "{context}: confidence is non-finite"
    );
    let slice = source
        .get(*start..*end)
        .with_context(|| format!("{context}: [{start},{end}) is not a UTF-8 byte range"))?;
    ensure!(
        slice.trim() == text,
        "{context}: original UTF-8 slice {slice:?} does not preserve output text {text:?}"
    );
    Ok(json!({
        "text": text,
        "confidence": confidence,
        "start": start,
        "end": end,
        "offset_unit": "utf8_byte",
    }))
}

fn entity_value(value: &FormattedEntityValue, source: &str, context: &str) -> Result<Value> {
    match value {
        FormattedEntityValue::List(spans) => spans
            .iter()
            .enumerate()
            .map(|(index, span)| span_value(span, source, &format!("{context}[{index}]")))
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        FormattedEntityValue::Single(Some(span)) => span_value(span, source, context),
        FormattedEntityValue::Single(None) => Ok(Value::Null),
    }
}

fn classification_value(value: &FormattedClassification, context: &str) -> Result<Value> {
    match value {
        FormattedClassification::SingleWithConfidence { label, confidence } => {
            ensure!(
                confidence.is_finite(),
                "{context}: confidence is non-finite"
            );
            Ok(json!({"label": label, "confidence": confidence}))
        }
        FormattedClassification::MultiWithConfidence(labels) => labels
            .iter()
            .map(|(label, confidence)| {
                ensure!(
                    confidence.is_finite(),
                    "{context}: confidence is non-finite"
                );
                Ok(json!({"label": label, "confidence": confidence}))
            })
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        _ => bail!("{context}: typed output omitted requested confidence"),
    }
}

fn result_value(result: &gliner2_rs::schema_spec::ExtractionResult, source: &str) -> Result<Value> {
    let mut root = Map::new();
    if !result.entities.is_empty() {
        let mut entities = Map::new();
        for (name, value) in &result.entities {
            entities.insert(name.clone(), entity_value(value, source, name)?);
        }
        root.insert("entities".to_owned(), Value::Object(entities));
    }
    for (name, value) in &result.classifications {
        root.insert(name.clone(), classification_value(value, name)?);
    }
    for (name, instances) in &result.structures {
        let mut converted = Vec::with_capacity(instances.len());
        for (instance_index, instance) in instances.iter().enumerate() {
            let mut fields = Map::new();
            for (field, value) in instance {
                fields.insert(
                    field.clone(),
                    entity_value(value, source, &format!("{name}[{instance_index}].{field}"))?,
                );
            }
            converted.push(Value::Object(fields));
        }
        root.insert(name.clone(), Value::Array(converted));
    }
    if !result.relations.is_empty() {
        let mut relations = Map::new();
        for (name, pairs) in &result.relations {
            let converted = pairs
                .iter()
                .enumerate()
                .map(|(index, pair)| {
                    Ok(json!({
                        "head": span_value(
                            &pair.head,
                            source,
                            &format!("{name}[{index}].head"),
                        )?,
                        "tail": span_value(
                            &pair.tail,
                            source,
                            &format!("{name}[{index}].tail"),
                        )?,
                    }))
                })
                .collect::<Result<Vec<_>>>()?;
            relations.insert(name.clone(), Value::Array(converted));
        }
        root.insert("relation_extraction".to_owned(), Value::Object(relations));
    }
    Ok(Value::Object(root))
}

fn compare_json(actual: &Value, expected: &Value, path: &str, stats: &mut Stats) -> Result<()> {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            let actual_keys: BTreeSet<_> = actual.keys().collect();
            let expected_keys: BTreeSet<_> = expected.keys().collect();
            ensure!(
                actual_keys == expected_keys,
                "{path}: object keys differ; actual={actual_keys:?}, expected={expected_keys:?}"
            );
            for (key, expected) in expected {
                compare_json(
                    actual
                        .get(key)
                        .context("key disappeared during comparison")?,
                    expected,
                    &format!("{path}.{key}"),
                    stats,
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
                compare_json(actual, expected, &format!("{path}[{index}]"), stats)?;
            }
        }
        (Value::Number(actual), Value::Number(expected)) if path.ends_with(".confidence") => {
            let actual = actual.as_f64().context("actual confidence")?;
            let expected = expected.as_f64().context("expected confidence")?;
            ensure!(
                actual.is_finite() && expected.is_finite(),
                "{path}: confidence is non-finite"
            );
            let difference = (actual - expected).abs();
            stats.maximum_confidence_abs = stats.maximum_confidence_abs.max(difference);
            ensure!(
                difference <= 1e-3,
                "{path}: confidence {actual} != {expected}; abs={difference}"
            );
        }
        _ => ensure!(
            actual == expected,
            "{path}: actual={actual} != expected={expected}"
        ),
    }
    Ok(())
}

fn expected_strings(value: &Value, key: &str) -> Result<Vec<String>> {
    string_list(&value[key], key)
}

fn run_explicit(
    pipeline: &mut BoundaryPipeline,
    fixture_dir: &Path,
    stats: &mut Stats,
) -> Result<()> {
    let path = fixture_dir
        .join("additional")
        .join("explicit_unicode_duplicates.json");
    let fixture: Value = serde_json::from_slice(&fs::read(&path)?)?;
    ensure!(fixture["case_id"] == "explicit_unicode_duplicates");
    let text = fixture["original_text"].as_str().context("explicit text")?;
    let labels = expected_strings(&fixture, "labels")?;
    let spans = fixture["spans"]
        .as_array()
        .context("explicit spans")?
        .iter()
        .map(|span| {
            let bounds = span["requested_utf8"]
                .as_array()
                .context("explicit UTF-8 bounds")?;
            ensure!(bounds.len() == 2);
            Ok([
                bounds[0].as_u64().context("explicit start")? as usize,
                bounds[1].as_u64().context("explicit end")? as usize,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(labels.len() == 2);
    ensure!(spans.len() == 4 && spans[0] == spans[3]);
    pipeline.set_word_splitter(gliner2_rs::boundary::preprocessing::WordSplitter::Whitespace)?;
    let actual = pipeline.score_explicit_spans(text, &labels, &spans)?;
    let logits = fixture["logits"].as_array().context("explicit logits")?;
    let probabilities = fixture["probabilities"]
        .as_array()
        .context("explicit probabilities")?;
    ensure!(actual.len() == labels.len());
    for (query, group) in actual.iter().enumerate() {
        ensure!(group.label == labels[query], "explicit label order changed");
        ensure!(group.spans.len() == spans.len());
        let expected_logits = logits[query].as_array().context("query logits")?;
        let expected_probabilities = probabilities[query]
            .as_array()
            .context("query probabilities")?;
        for (candidate, score) in group.spans.iter().enumerate() {
            let [start, end] = spans[candidate];
            ensure!((score.start, score.end) == (start, end));
            ensure!(
                text.get(start..end) == Some(score.text.as_str()),
                "explicit score does not preserve original UTF-8 text"
            );
            ensure!(score.logit.is_finite() && score.confidence.is_finite());
            let expected_logit = expected_logits[candidate]
                .as_f64()
                .context("expected explicit logit")?;
            let logit_error = (f64::from(score.logit) - expected_logit).abs();
            stats.maximum_explicit_logit_abs = stats.maximum_explicit_logit_abs.max(logit_error);
            ensure!(
                logit_error <= 1e-4 + 1e-3 * expected_logit.abs(),
                "explicit [{query},{candidate}] logit differs by {logit_error}"
            );
            let expected_probability = expected_probabilities[candidate]
                .as_f64()
                .context("expected explicit probability")?;
            let confidence_error = (f64::from(score.confidence) - expected_probability).abs();
            stats.maximum_confidence_abs = stats.maximum_confidence_abs.max(confidence_error);
            ensure!(
                confidence_error <= 1e-3,
                "explicit [{query},{candidate}] confidence differs by {confidence_error}"
            );
        }
    }
    stats.task_families.insert("explicit_spans".to_owned());
    Ok(())
}

fn string_array(value: &Value, context: &str) -> Result<Vec<String>> {
    string_list(value, context)
}

fn validate_manifests(settings: &Settings) -> Result<(Value, Value)> {
    let export: Value =
        serde_json::from_slice(&fs::read(settings.graph_dir.join("export_manifest.json"))?)?;
    ensure!(export["manifest_version"] == 1);
    ensure!(export["architecture"] == "boundary");
    ensure!(export["architecture_version"] == 1);
    ensure!(export["hf_model"] == settings.profile.model_id);
    ensure!(export["hf_revision"] == settings.profile.revision);
    ensure!(export["gliner2_commit"] == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455");
    ensure!(export["opset"] == 17 && export["precision"] == "fp32");
    let graphs = export["graphs"].as_object().context("bundle graph map")?;
    let graph_ids: BTreeSet<_> = graphs.keys().map(String::as_str).collect();
    ensure!(graph_ids == BTreeSet::from(GRAPH_FILES));

    let fixtures: Value =
        serde_json::from_slice(&fs::read(settings.fixture_dir.join("manifest.json"))?)?;
    ensure!(fixtures["format_version"] == 2);
    ensure!(fixtures["status"] == "complete");
    ensure!(fixtures["bundle_profile"] == settings.profile.name);
    ensure!(fixtures["bundle_name"] == settings.profile.bundle_name);
    ensure!(fixtures["model_id"] == settings.profile.model_id);
    ensure!(fixtures["hf_revision"] == settings.profile.revision);
    ensure!(fixtures["gliner2_commit"] == export["gliner2_commit"]);
    ensure!(fixtures["source_file_sha256"] == export["source_file_sha256"]);
    ensure!(fixtures["case_count"] == 30);
    ensure!(fixtures["successful_case_count"] == 30);
    ensure!(fixtures["error_case_count"] == 0);
    ensure!(
        string_array(&fixtures["case_ids"], "manifest case IDs")? == CASE_IDS.map(str::to_owned)
    );
    ensure!(
        string_array(&fixtures["additional_case_ids"], "manifest additional IDs")?
            == ADDITIONAL_IDS.map(str::to_owned)
    );
    Ok((export, fixtures))
}

fn run_fixture(pipeline: &mut BoundaryPipeline, path: &Path, stats: &mut Stats) -> Result<()> {
    let fixture: Value = serde_json::from_slice(&fs::read(path)?)?;
    let case_id = fixture["case_id"].as_str().context("fixture case ID")?;
    let text = fixture["original_text"]
        .as_str()
        .context("fixture original text")?;
    ensure!(fixture["text"].as_str() == Some(text));
    let threshold = fixture["threshold"].as_f64().context("fixture threshold")? as f32;
    let (schema, records) = schema_and_records(&fixture["schema_spec"])
        .with_context(|| format!("{case_id}: schema conversion"))?;
    if let Some(policy) = fixture.get("overlap_policy").and_then(Value::as_str) {
        pipeline.set_overlap_policy(OverlapPolicy::from_str(policy)?);
    }
    let actual = pipeline.extract_with_records(text, &schema, &records, threshold, true, true)?;
    let actual = result_value(&actual, text)?;
    compare_json(
        &actual,
        &fixture["final_result_utf8_offsets"],
        case_id,
        stats,
    )?;
    stats.cases += 1;
    stats.categories.insert(
        fixture["category"]
            .as_str()
            .context("fixture category")?
            .to_owned(),
    );
    if !schema.entities.is_empty() {
        stats.task_families.insert("entities".to_owned());
    }
    if !schema.classifications.is_empty() {
        stats.task_families.insert("classifications".to_owned());
    }
    if !schema.structures.is_empty() {
        stats.task_families.insert("structures".to_owned());
    }
    if !schema.relations.is_empty() {
        stats.task_families.insert("relations".to_owned());
    }
    Ok(())
}

#[test]
fn synthetic_schema_conversion_covers_all_task_families_and_record_metadata() -> Result<()> {
    let specification = json!({
        "entities_config": {
            "person": {"dtype": "list", "description": "a person", "threshold": 0.4}
        },
        "classifications": [{
            "task": "sentiment",
            "labels": ["positive", "negative"],
            "multi_label": true,
            "cls_threshold": 0.25,
            "class_act": "sigmoid"
        }],
        "structures": [{
            "name": "person_record",
            "mode": "natural",
            "anchor": "name",
            "fields": [
                {"name": "name", "dtype": "str", "cardinality": "required_one", "exclusive": true},
                {"name": "city", "dtype": "list", "choices": ["東京"], "cardinality": "zero_or_more"}
            ]
        }],
        "relations": [{"name": "lives in", "description": "person to city", "threshold": 0.3}]
    });
    let (schema, records) = schema_and_records(&specification)?;
    ensure!(schema.entities.len() == 1);
    ensure!(schema.classifications.len() == 1);
    ensure!(schema.structures.len() == 1);
    ensure!(schema.relations.len() == 1);
    ensure!(records.len() == 1);
    ensure!(schema.structures[0].fields[1].choices == ["東京"]);
    Ok(())
}

#[test]
fn synthetic_typed_conversion_preserves_utf8_and_confidence_policy() -> Result<()> {
    let text = "Zoë met 李雷";
    let zoe = FormattedEntitySpan::TextWithConfidenceAndSpans {
        text: "Zoë".to_owned(),
        confidence: 0.75,
        start: 0,
        end: 4,
    };
    let li = FormattedEntitySpan::TextWithConfidenceAndSpans {
        text: "李雷".to_owned(),
        confidence: 0.625,
        start: 9,
        end: 15,
    };
    let mut result = gliner2_rs::schema_spec::ExtractionResult::default();
    result.entities.insert(
        "person".to_owned(),
        FormattedEntityValue::List(vec![zoe.clone(), li.clone()]),
    );
    result.classifications.insert(
        "language".to_owned(),
        FormattedClassification::SingleWithConfidence {
            label: "multilingual".to_owned(),
            confidence: 0.9,
        },
    );
    result.structures.insert(
        "meeting".to_owned(),
        vec![BTreeMap::from([(
            "attendee".to_owned(),
            FormattedEntityValue::Single(Some(zoe.clone())),
        )])],
    );
    result.relations.insert(
        "met".to_owned(),
        vec![gliner2_rs::relations::FormattedRelationPair {
            head: zoe,
            tail: li,
        }],
    );
    let actual = result_value(&result, text)?;
    let mut expected = actual.clone();
    expected["entities"]["person"][0]["confidence"] = json!(0.7505);
    let mut stats = Stats::default();
    compare_json(&actual, &expected, "synthetic", &mut stats)?;
    ensure!(stats.maximum_confidence_abs > 0.0);
    ensure!(actual["entities"]["person"][1]["start"] == 9);
    ensure!(actual["entities"]["person"][1]["end"] == 15);
    Ok(())
}

#[test]
fn per_checkpoint_bundle_matches_independent_source_oracles() -> Result<()> {
    let Some(settings) = settings()? else {
        return Ok(());
    };
    let (export_manifest, fixture_manifest) = validate_manifests(&settings)?;
    let mut pipeline = BoundaryPipeline::from_dir(&settings.graph_dir)?;
    let default_overlap = pipeline.overlap_policy();
    let mut stats = Stats::default();
    for case_id in CASE_IDS {
        pipeline.set_overlap_policy(default_overlap);
        run_fixture(
            &mut pipeline,
            &settings.fixture_dir.join(format!("{case_id}.json")),
            &mut stats,
        )
        .with_context(|| case_id.to_owned())?;
    }
    pipeline.set_overlap_policy(default_overlap);
    run_fixture(
        &mut pipeline,
        &settings
            .fixture_dir
            .join("additional")
            .join("mixed_unicode_all_tasks.json"),
        &mut stats,
    )
    .context("mixed_unicode_all_tasks")?;
    run_explicit(&mut pipeline, &settings.fixture_dir, &mut stats)?;

    ensure!(stats.cases == 31);
    ensure!(
        stats.task_families
            == BTreeSet::from([
                "classifications".to_owned(),
                "entities".to_owned(),
                "explicit_spans".to_owned(),
                "relations".to_owned(),
                "structures".to_owned(),
            ]),
        "typed task-family coverage is incomplete: {:?}",
        stats.task_families
    );

    if let Some(path) = settings.report_json {
        let report = json!({
            "format_version": 1,
            "kind": "native-bundle-parity",
            "status": "passed",
            "release_ready": false,
            "profile": settings.profile.name,
            "bundle_name": settings.profile.bundle_name,
            "hf_model": settings.profile.model_id,
            "hf_revision": settings.profile.revision,
            "gliner2_commit": export_manifest["gliner2_commit"],
            "source_file_sha256": export_manifest["source_file_sha256"],
            "graph_files": export_manifest["files"],
            "fixture_generator_sha256": fixture_manifest["provenance"]["generator_sha256"],
            "fixture_entries": fixture_manifest["entries"],
            "fixture_additional_entries": fixture_manifest["additional_entries"],
            "runtime": {
                "ort_api_minor": ort::MINOR_VERSION,
                "onnx_runtime_build_info": ort::info(),
                "rust_package_version": env!("CARGO_PKG_VERSION"),
            },
            "checks": {
                "corpus_cases": 30,
                "mixed_cases": 1,
                "explicit_cases": 1,
                "exact_discrete_and_utf8_outputs": true,
                "typed_task_families": stats.task_families,
                "categories": stats.categories,
                "maximum_confidence_abs": stats.maximum_confidence_abs,
                "maximum_explicit_logit_abs": stats.maximum_explicit_logit_abs,
                "confidence_atol": 1e-3,
                "stage_atol": 1e-4,
                "stage_rtol": 1e-3,
            },
        });
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(())
}
