use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::{
    boundary::pipeline::BoundaryPipeline,
    entities::{FormattedEntitySpan, FormattedEntityValue},
    json::JsonSchema,
    schema_spec::{FieldDtype, SchemaBuilder, SchemaSpec, StructureFieldSpec, StructureSpec},
    validators::RegexValidator,
};
use serde_json::Value;

fn bundle() -> Result<Option<PathBuf>> {
    let root = env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let bundle = root.join("onnx/gliner2.5-base-v1");
    let required = [
        "config.json",
        "tokenizer.json",
        "encoder.onnx",
        "classifier.onnx",
        "boundary_marginals.onnx",
        "boundary_scorer.onnx",
        "boundary_explicit_scorer.onnx",
        "boundary_records.onnx",
        "boundary_relations.onnx",
    ];
    let missing: Vec<_> = required
        .iter()
        .map(|name| bundle.join(name))
        .filter(|path| !path.is_file())
        .collect();
    if missing.is_empty() {
        return Ok(Some(bundle));
    }
    let message = format!(
        "missing boundary legacy-JSON artifacts: {}",
        missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn legacy_fixture() -> Result<Value> {
    let root = env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let path = root.join("fixtures/gliner2.5-base-v1/json_legacy_event.json");
    if !path.is_file() {
        ensure!(
            env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1"),
            "missing required full fixture {}",
            path.display()
        );
        return Err(anyhow!("SKIP: missing {}", path.display()));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn codepoint_to_byte(text: &str, codepoint: usize) -> Result<usize> {
    if codepoint == text.chars().count() {
        return Ok(text.len());
    }
    text.char_indices()
        .nth(codepoint)
        .map(|(byte, _)| byte)
        .context("golden Python code-point offset is out of range")
}

fn assert_span(actual: &FormattedEntitySpan, expected: &Value, text: &str) -> Result<()> {
    let FormattedEntitySpan::TextWithConfidenceAndSpans {
        text: actual_text,
        confidence,
        start,
        end,
    } = actual
    else {
        anyhow::bail!("expected confidence-and-spans structure value");
    };
    ensure!(actual_text == expected["text"].as_str().context("golden text")?);
    let expected_start = codepoint_to_byte(
        text,
        expected["start"].as_u64().context("golden start")? as usize,
    )?;
    let expected_end = codepoint_to_byte(
        text,
        expected["end"].as_u64().context("golden end")? as usize,
    )?;
    ensure!(*start == expected_start && *end == expected_end);
    ensure!(text.get(*start..*end) == Some(actual_text.as_str()));
    let reference = expected["confidence"]
        .as_f64()
        .context("golden confidence")? as f32;
    ensure!((*confidence - reference).abs() <= 1e-3);
    Ok(())
}

fn legacy_schema() -> SchemaSpec {
    SchemaBuilder::new()
        .structure("event")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("date").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("location").dtype(FieldDtype::Str))
        .finish()
        .build()
}

#[test]
fn actual_legacy_event_matches_upstream_single_instance_fixture() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let golden = match legacy_fixture() {
        Ok(value) => value,
        Err(error) if error.to_string().starts_with("SKIP:") => {
            eprintln!("{error}");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let text = golden["text"].as_str().context("fixture text")?;
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let actual = pipeline.extract_with_confidence_and_spans(text, &legacy_schema(), 0.5)?;
    ensure!(actual.structures.len() == 1);
    let instances = actual.structures.get("event").context("event output")?;
    ensure!(instances.len() == 1, "legacy JSON must emit one instance");
    let instance = &instances[0];
    let expected = &golden["final_result_python_offsets"]["event"][0];
    ensure!(instance.len() == 3);
    for field in ["name", "date", "location"] {
        let value = instance.get(field).with_context(|| field.to_owned())?;
        match (value, &expected[field]) {
            (FormattedEntityValue::Single(Some(actual)), Value::Object(_)) => {
                assert_span(actual, &expected[field], text)?;
            }
            (FormattedEntityValue::List(actual), Value::Array(expected)) => {
                ensure!(actual.len() == expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_span(actual, expected, text)?;
                }
            }
            _ => anyhow::bail!("{field}: output shape differs from fixture"),
        }
    }
    Ok(())
}

#[test]
fn json_schema_delegates_and_mixed_global_query_routing_is_stable() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let text = "The conference begins Monday in Tokyo and ends Wednesday.";
    let quick = JsonSchema::new().structure(
        "event",
        vec![
            "name::str".into(),
            "date::list".into(),
            "location::str".into(),
        ],
    );
    let quick_result = pipeline.extract_json_with_confidence_and_spans(text, &quick, 0.5)?;
    let typed_result = pipeline
        .extract_with_confidence_and_spans(text, &legacy_schema(), 0.5)?
        .structures;
    ensure!(quick_result == typed_result);

    let mixed = SchemaBuilder::new()
        .structure("event")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("date").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("location").dtype(FieldDtype::Str))
        .finish()
        .entities(vec!["location".to_owned()])
        .classification(
            "sentiment",
            vec!["positive".to_owned(), "negative".to_owned()],
        )
        .build();
    let mixed_result = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;
    ensure!(mixed_result.structures.contains_key("event"));
    ensure!(mixed_result.entities.contains_key("location"));
    ensure!(mixed_result.classifications.contains_key("sentiment"));
    Ok(())
}

#[test]
fn choices_have_schema_shapes_without_synthetic_spans() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let schema = SchemaBuilder::new()
        .structure("ticket")
        .field(
            StructureFieldSpec::new("tags")
                .dtype(FieldDtype::List)
                .choices(vec![
                    "very urgent".into(),
                    "low".into(),
                    "very urgent".into(),
                ])
                .threshold(0.0),
        )
        .field(
            StructureFieldSpec::new("priority")
                .dtype(FieldDtype::Str)
                .choices(vec!["low".into(), "high".into()])
                .threshold(0.0),
        )
        .finish()
        .build();
    let result = pipeline.extract_with_confidence_and_spans(
        "This ticket is high priority.",
        &schema,
        0.5,
    )?;
    let instance = &result.structures["ticket"][0];
    let FormattedEntityValue::List(tags) = &instance["tags"] else {
        anyhow::bail!("choice list shape");
    };
    ensure!(tags.len() == 2, "exact duplicate choices must be removed");
    ensure!(
        tags.iter()
            .all(|value| matches!(value, FormattedEntitySpan::TextWithConfidence { .. }))
    );
    let FormattedEntityValue::Single(Some(priority)) = &instance["priority"] else {
        anyhow::bail!("choice scalar shape");
    };
    ensure!(matches!(
        priority,
        FormattedEntitySpan::TextWithConfidence { .. }
    ));
    Ok(())
}

#[test]
fn configured_thresholds_and_validators_can_empty_an_entire_structure() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    let schema = SchemaBuilder::new()
        .structure("event")
        .field(
            StructureFieldSpec::new("name")
                .dtype(FieldDtype::Str)
                .threshold(0.0)
                .validators(vec![RegexValidator::new("^NEVER$")]),
        )
        .field(
            StructureFieldSpec::new("kind")
                .dtype(FieldDtype::List)
                .choices(vec!["conference".into(), "meeting".into()])
                .threshold(1.0),
        )
        .finish()
        .build();
    ensure!(
        pipeline
            .extract("The conference begins Monday.", &schema, 0.5)?
            .structures
            .is_empty()
    );

    let malformed = SchemaSpec {
        structures: vec![StructureSpec {
            name: "bad".into(),
            fields: vec![StructureFieldSpec::new("field").threshold(f32::NAN)],
        }],
        ..SchemaSpec::default()
    };
    let error = pipeline
        .extract("must reject", &malformed, 0.5)
        .unwrap_err();
    ensure!(error.to_string().contains("finite and in [0,1]"));
    Ok(())
}
