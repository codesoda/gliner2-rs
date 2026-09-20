use std::{
    collections::BTreeMap,
    env, fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::{
    boundary::{
        pipeline::BoundaryPipeline,
        record_decode::Cardinality,
        record_schema::{RecordConfig, RecordFieldOptions, RecordMetadata},
    },
    entities::{FormattedEntitySpan, FormattedEntityValue},
    json::JsonSchema,
    pipeline::{AutoPipeline, SpanPipeline},
    schema_spec::{EntitySpec, FieldDtype, SchemaSpec, StructureFieldSpec, StructureSpec},
};
use serde_json::Value;

static MAX_CONFIDENCE_DELTA: AtomicU32 = AtomicU32::new(0);

const CASE_IDS: [&str; 4] = [
    "json_natural_people",
    "json_natural_choice",
    "json_latent_products",
    "json_anchorless_list",
];

fn strict() -> bool {
    env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1")
        || env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1")
}

fn root() -> PathBuf {
    env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn bundle() -> Result<Option<PathBuf>> {
    let bundle = root().join("onnx/gliner2.5-base-v1");
    let required = [
        "config.json",
        "tokenizer.json",
        "encoder.onnx",
        "classifier.onnx",
        "boundary_marginals.onnx",
        "boundary_scorer.onnx",
        "boundary_explicit_scorer.onnx",
        "boundary_records.onnx",
    ];
    let missing: Vec<_> = required
        .iter()
        .map(|name| bundle.join(name))
        .filter(|path| !path.is_file())
        .collect();
    if missing.is_empty() {
        return Ok(Some(bundle));
    }
    let message = format!("missing boundary record artifacts: {missing:?}");
    if strict() {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn fixtures() -> Result<Option<Vec<Value>>> {
    let directory = root().join("fixtures/gliner2.5-base-v1");
    let mut fixtures = Vec::with_capacity(CASE_IDS.len());
    for id in CASE_IDS {
        let path = directory.join(format!("{id}.json"));
        if !path.is_file() {
            let message = format!("missing boundary record fixture {}", path.display());
            if strict() {
                return Err(anyhow!(message));
            }
            eprintln!("SKIP: {message}");
            return Ok(None);
        }
        let fixture: Value = serde_json::from_slice(&fs::read(&path)?)?;
        ensure!(
            fixture["case_id"] == id,
            "fixture ID mismatch at {}",
            path.display()
        );
        fixtures.push(fixture);
    }
    Ok(Some(fixtures))
}

fn cardinality(value: &str) -> Result<Cardinality> {
    match value {
        "optional_one" => Ok(Cardinality::OptionalOne),
        "required_one" => Ok(Cardinality::RequiredOne),
        "zero_or_more" => Ok(Cardinality::ZeroOrMore),
        "one_or_more" => Ok(Cardinality::OneOrMore),
        other => Err(anyhow!("unknown fixture cardinality {other:?}")),
    }
}

fn schema_and_metadata(fixture: &Value) -> Result<(SchemaSpec, RecordMetadata)> {
    let structures = fixture["schema_spec"]["structures"]
        .as_array()
        .context("fixture structures")?;
    let mut schema_structures = Vec::with_capacity(structures.len());
    let mut metadata = RecordMetadata::new();
    for structure in structures {
        let name = structure["name"].as_str().context("structure name")?;
        let mode = structure["mode"].as_str().context("record mode")?;
        let mut config = match mode {
            "natural" => {
                RecordConfig::natural(structure["anchor"].as_str().context("natural anchor")?)
            }
            "latent" => RecordConfig::latent(),
            "anchorless" => RecordConfig::anchorless(),
            other => return Err(anyhow!("unknown fixture record mode {other:?}")),
        };
        let mut fields = Vec::new();
        for field in structure["fields"].as_array().context("record fields")? {
            let field_name = field["name"].as_str().context("field name")?;
            let dtype = match field["dtype"].as_str().context("field dtype")? {
                "str" => FieldDtype::Str,
                "list" => FieldDtype::List,
                other => return Err(anyhow!("unknown fixture dtype {other:?}")),
            };
            let choices = field
                .get("choices")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| value.as_str().context("choice").map(str::to_owned))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default();
            fields.push(
                StructureFieldSpec::new(field_name)
                    .dtype(dtype)
                    .choices(choices),
            );
            config = config.field(
                field_name,
                RecordFieldOptions::default()
                    .cardinality(cardinality(
                        field["cardinality"].as_str().context("field cardinality")?,
                    )?)
                    .exclusive(
                        field
                            .get("exclusive")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    ),
            );
        }
        schema_structures.push(StructureSpec {
            name: name.to_owned(),
            fields,
        });
        ensure!(metadata.insert(name.to_owned(), config).is_none());
    }
    Ok((
        SchemaSpec {
            structures: schema_structures,
            ..SchemaSpec::default()
        },
        metadata,
    ))
}

fn assert_span(
    actual: &FormattedEntitySpan,
    expected: &Value,
    include_confidence: bool,
    include_spans: bool,
    context: &str,
) -> Result<()> {
    let expected_text = expected["text"].as_str().context("expected text")?;
    let expected_confidence = expected["confidence"]
        .as_f64()
        .context("expected confidence")? as f32;
    let expected_start = expected["start"].as_u64().context("expected start")? as usize;
    let expected_end = expected["end"].as_u64().context("expected end")? as usize;
    let (text, confidence, spans) = match actual {
        FormattedEntitySpan::Text(text) => (text, None, None),
        FormattedEntitySpan::TextWithConfidence { text, confidence } => {
            (text, Some(*confidence), None)
        }
        FormattedEntitySpan::TextWithSpans { text, start, end } => {
            (text, None, Some((*start, *end)))
        }
        FormattedEntitySpan::TextWithConfidenceAndSpans {
            text,
            confidence,
            start,
            end,
        } => (text, Some(*confidence), Some((*start, *end))),
    };
    ensure!(
        text == expected_text,
        "{context}: text {text:?} != {expected_text:?}"
    );
    ensure!(
        confidence.is_some() == include_confidence,
        "{context}: confidence shape"
    );
    ensure!(spans.is_some() == include_spans, "{context}: span shape");
    if let Some(confidence) = confidence {
        ensure!(
            (confidence - expected_confidence).abs() <= 1e-3,
            "{context}: confidence {confidence} != {expected_confidence}"
        );
        MAX_CONFIDENCE_DELTA.fetch_max(
            (confidence - expected_confidence).abs().to_bits(),
            Ordering::Relaxed,
        );
    }
    if let Some((start, end)) = spans {
        ensure!(
            (start, end) == (expected_start, expected_end),
            "{context}: [{start},{end}) != [{expected_start},{expected_end})"
        );
    }
    Ok(())
}

fn assert_result(
    actual: &gliner2_rs::json::JsonExtraction,
    expected: &Value,
    include_confidence: bool,
    include_spans: bool,
) -> Result<()> {
    let expected = expected.as_object().context("expected record result")?;
    ensure!(actual.len() == expected.len(), "structure count differs");
    for (name, expected_instances) in expected {
        let actual_instances = actual
            .get(name)
            .with_context(|| format!("structure {name}"))?;
        let expected_instances = expected_instances
            .as_array()
            .context("expected instances")?;
        ensure!(
            actual_instances.len() == expected_instances.len(),
            "{name}: instance count"
        );
        for (record_index, (actual_record, expected_record)) in
            actual_instances.iter().zip(expected_instances).enumerate()
        {
            let expected_record = expected_record.as_object().context("expected record")?;
            ensure!(
                actual_record.len() == expected_record.len(),
                "{name}[{record_index}]: fields"
            );
            for (field, expected_value) in expected_record {
                let actual_value = actual_record
                    .get(field)
                    .with_context(|| format!("{name}[{record_index}].{field}"))?;
                let context = format!("{name}[{record_index}].{field}");
                match (actual_value, expected_value) {
                    (FormattedEntityValue::Single(None), Value::Null) => {}
                    (FormattedEntityValue::Single(Some(actual)), Value::Object(_)) => {
                        assert_span(
                            actual,
                            expected_value,
                            include_confidence,
                            include_spans,
                            &context,
                        )?;
                    }
                    (FormattedEntityValue::List(actual), Value::Array(expected)) => {
                        ensure!(actual.len() == expected.len(), "{context}: list count");
                        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                            assert_span(
                                actual,
                                expected,
                                include_confidence,
                                include_spans,
                                &format!("{context}[{index}]"),
                            )?;
                        }
                    }
                    _ => return Err(anyhow!("{context}: value shape differs")),
                }
            }
        }
    }
    Ok(())
}

#[test]
fn four_frozen_record_fixtures_match_all_final_values_and_order() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let Some(fixtures) = fixtures()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    for fixture in fixtures {
        let id = fixture["case_id"].as_str().context("case ID")?;
        let text = fixture["text"].as_str().context("fixture text")?;
        let threshold = fixture["threshold"].as_f64().unwrap_or(0.5) as f32;
        let (schema, metadata) = schema_and_metadata(&fixture)?;
        MAX_CONFIDENCE_DELTA.store(0, Ordering::Relaxed);
        let actual = pipeline
            .extract_with_records(text, &schema, &metadata, threshold, true, true)?
            .structures;
        assert_result(&actual, &fixture["final_result_utf8_offsets"], true, true)
            .with_context(|| id.to_owned())?;

        if id == "json_natural_choice" {
            let json_schema = JsonSchema::new().structure(
                "vendor",
                vec![
                    "company::str".into(),
                    "category::[books|hardware|software]::str".into(),
                ],
            );
            let json_result = pipeline.extract_json_with_records(
                text,
                &json_schema,
                &metadata,
                threshold,
                true,
                true,
            )?;
            assert_result(
                &json_result,
                &fixture["final_result_utf8_offsets"],
                true,
                true,
            )?;

            let no_spans = pipeline
                .extract_with_records(text, &schema, &metadata, threshold, true, false)?
                .structures;
            assert_result(
                &no_spans,
                &fixture["final_result_utf8_offsets"],
                true,
                false,
            )?;
            let plain = pipeline
                .extract_with_records(text, &schema, &metadata, threshold, false, false)?
                .structures;
            assert_result(&plain, &fixture["final_result_utf8_offsets"], false, false)?;
        }
        eprintln!(
            "M5 record {id}: maximum confidence error = {:e}",
            f32::from_bits(MAX_CONFIDENCE_DELTA.load(Ordering::Relaxed))
        );
    }
    Ok(())
}

#[test]
fn annotated_record_routes_with_other_query_families_in_one_schema() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let Some(fixtures) = fixtures()? else {
        return Ok(());
    };
    let fixture = &fixtures[0];
    let (mut schema, metadata) = schema_and_metadata(fixture)?;
    schema.entities.push(EntitySpec::new("location"));
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let result = pipeline.extract_with_records(
        fixture["text"].as_str().context("fixture text")?,
        &schema,
        &metadata,
        0.5,
        true,
        true,
    )?;
    ensure!(result.structures.contains_key("person"));
    ensure!(result.entities.contains_key("location"));
    Ok(())
}

#[test]
fn sidecar_preflight_and_auto_forwarding_are_explicit() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(&bundle)?;
    let schema = SchemaSpec {
        structures: vec![StructureSpec {
            name: "record".into(),
            fields: vec![StructureFieldSpec::new("value").dtype(FieldDtype::Str)],
        }],
        ..SchemaSpec::default()
    };
    let invalid = BTreeMap::from([("typo".into(), RecordConfig::latent())]);
    let error = pipeline
        .extract_with_records("text", &schema, &invalid, 0.5, false, false)
        .unwrap_err()
        .to_string();
    ensure!(error.contains("unknown structure"), "{error}");

    let empty = RecordMetadata::new();
    ensure!(
        pipeline.extract("text", &schema, 0.5)?
            == pipeline.extract_with_records("text", &schema, &empty, 0.5, false, false)?,
        "empty sidecar must preserve legacy behavior"
    );

    let auto = AutoPipeline::from_dir(bundle)?;
    let forwarded = auto.extract_with_records("text", &schema, &empty, 0.5, false, false)?;
    ensure!(forwarded == pipeline.extract("text", &schema, 0.5)?);
    let json_schema = JsonSchema::new().structure("record", vec!["value::str".into()]);
    ensure!(
        auto.extract_json_with_records("text", &json_schema, &empty, 0.5, false, false,)?
            == pipeline.extract_json("text", &json_schema)?
    );

    // Existing v2 test artifacts deliberately separate tokenizer/config from
    // ONNX graphs. Exercise the span branch rather than skipping a usable split
    // bundle merely because it is not yet packaged as a release directory.
    let span_bundle = root().join("onnx/gliner2-base-v1");
    let span_model = root().join("models/gliner2-base-v1");
    let missing: Vec<_> = [
        span_model.join("config.json"),
        span_model.join("tokenizer.json"),
        span_bundle.join("encoder.onnx"),
        span_bundle.join("extractor_padded.onnx"),
    ]
    .into_iter()
    .filter(|path| !path.is_file())
    .collect();
    if missing.is_empty() {
        let span = AutoPipeline::Span(Box::new(SpanPipeline::new(
            &span_model,
            span_bundle.join("encoder.onnx"),
            span_bundle.join("extractor_padded.onnx"),
        )?));
        let error = span
            .extract_with_records("text", &schema, &empty, 0.5, false, false)
            .unwrap_err()
            .to_string();
        ensure!(error.contains("unsupported for span models"), "{error}");
        let error = span
            .extract_json_with_records("text", &json_schema, &empty, 0.5, false, false)
            .unwrap_err()
            .to_string();
        ensure!(error.contains("unsupported for span models"), "{error}");
    } else {
        ensure!(
            env::var("GLINER2_REQUIRE_MODELS").as_deref() != Ok("1"),
            "missing required v2 unsupported-path test artifacts: {missing:?}"
        );
        eprintln!("SKIP: missing v2 unsupported-path test artifacts: {missing:?}");
    }
    Ok(())
}
