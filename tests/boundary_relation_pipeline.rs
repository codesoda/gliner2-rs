use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::{
    boundary::BoundaryPipeline,
    entities::FormattedEntitySpan,
    pipeline::AutoPipeline,
    schema_spec::{FieldDtype, RelationSpec, SchemaBuilder, SchemaSpec, StructureFieldSpec},
};
use serde_json::Value;

const RELATION_CASE_IDS: [&str; 4] = [
    "relation_employment",
    "relation_founded",
    "relation_location",
    "relation_multiple_types",
];

fn root() -> PathBuf {
    env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn strict_boundary() -> bool {
    env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1")
        || env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1")
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
    let message = format!("missing boundary relation artifacts: {missing:?}");
    if strict_boundary() {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn fixture(case_id: &str) -> Result<Value> {
    ensure!(
        RELATION_CASE_IDS.contains(&case_id),
        "relation fixture ID {case_id:?} is not in the frozen case set"
    );
    let directory = if case_id == "relation_employment" {
        "fixtures/gliner2.5-base-v1-subset"
    } else {
        "fixtures/gliner2.5-base-v1"
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(directory)
        .join(format!("{case_id}.json"));
    let fixture: Value = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("missing frozen fixture {}", path.display()))?,
    )?;
    ensure!(fixture["case_id"] == case_id);
    ensure!(fixture["category"] == "relation");
    Ok(fixture)
}

fn relation_specs(fixture: &Value) -> Result<Vec<RelationSpec>> {
    fixture["schema_spec"]["relations"]
        .as_array()
        .context("fixture relation schema")?
        .iter()
        .map(|value| match value {
            Value::String(name) => Ok(RelationSpec::new(name.clone())),
            Value::Object(object) => {
                let name = object
                    .get("name")
                    .and_then(Value::as_str)
                    .context("fixture relation name")?;
                let mut spec = RelationSpec::new(name);
                if let Some(description) = object.get("description").and_then(Value::as_str) {
                    spec = spec.description(description);
                }
                Ok(spec)
            }
            _ => Err(anyhow!("malformed fixture relation spec: {value}")),
        })
        .collect()
}

fn assert_span(actual: &FormattedEntitySpan, expected: &Value, context: &str) -> Result<()> {
    let FormattedEntitySpan::TextWithConfidenceAndSpans {
        text,
        confidence,
        start,
        end,
    } = actual
    else {
        return Err(anyhow!("{context}: expected confidence-and-span value"));
    };
    let expected_text = expected["text"]
        .as_str()
        .context("expected relation text")?;
    let expected_start = expected["start"]
        .as_u64()
        .context("expected relation start")? as usize;
    let expected_end = expected["end"].as_u64().context("expected relation end")? as usize;
    let expected_confidence = expected["confidence"]
        .as_f64()
        .context("expected relation confidence")? as f32;
    ensure!(
        (text.as_str(), *start, *end) == (expected_text, expected_start, expected_end),
        "{context}: source span mismatch"
    );
    let error = (confidence - expected_confidence).abs();
    ensure!(
        confidence.is_finite() && error <= 1e-3,
        "{context}: confidence {confidence} != {expected_confidence}"
    );
    eprintln!("{context}: relation_confidence_abs_error={error:e}");
    Ok(())
}

fn assert_fixture(pipeline: &BoundaryPipeline, fixture: &Value) -> Result<()> {
    let text = fixture["text"].as_str().context("fixture text")?;
    let threshold = fixture["threshold"].as_f64().context("fixture threshold")? as f32;
    let schema = SchemaSpec {
        relations: relation_specs(fixture)?,
        ..SchemaSpec::default()
    };
    let actual = pipeline.extract_with_confidence_and_spans(text, &schema, threshold)?;
    ensure!(actual.entities.is_empty());
    ensure!(actual.classifications.is_empty());
    ensure!(actual.structures.is_empty());

    let expected = fixture["final_result_utf8_offsets"]["relation_extraction"]
        .as_object()
        .context("fixture relation result")?;
    ensure!(
        actual.relations.len() == expected.len(),
        "relation key count"
    );
    for (name, expected_pairs) in expected {
        let actual_pairs = actual
            .relations
            .get(name)
            .with_context(|| format!("missing relation {name:?}"))?;
        let expected_pairs = expected_pairs
            .as_array()
            .context("fixture relation pairs")?;
        ensure!(
            actual_pairs.len() == expected_pairs.len(),
            "{name}: pair count"
        );
        for (index, (actual, expected)) in actual_pairs.iter().zip(expected_pairs).enumerate() {
            assert_span(
                &actual.head,
                &expected["head"],
                &format!("{name}[{index}].head"),
            )?;
            assert_span(
                &actual.tail,
                &expected["tail"],
                &format!("{name}[{index}].tail"),
            )?;
        }
    }
    Ok(())
}

#[test]
fn committed_and_opt_in_full_relation_fixtures_match_upstream_outputs() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    // A downloaded bundle alone must still exercise the committed real oracle.
    // The full gate requires all four IDs, never a discovered/vacuously empty set.
    let case_ids: &[&str] = if env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1") {
        &RELATION_CASE_IDS
    } else {
        &["relation_employment"]
    };
    let fixtures = case_ids
        .iter()
        .map(|case_id| fixture(case_id))
        .collect::<Result<Vec<_>>>()?;
    ensure!(fixtures.len() == case_ids.len());
    let pipeline = BoundaryPipeline::from_dir(bundle)?;
    for fixture in &fixtures {
        assert_fixture(&pipeline, fixture)
            .with_context(|| fixture["case_id"].as_str().unwrap_or("unknown").to_owned())?;
    }
    Ok(())
}

#[test]
fn relation_public_options_batch_and_auto_forwarding_are_consistent() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let employment = fixture("relation_employment")?;
    let pipeline = BoundaryPipeline::from_dir(&bundle)?;
    let text = employment["text"].as_str().context("employment text")?;
    let relation_types = vec!["works for".to_owned()];

    let plain = pipeline.extract_relations(text, &relation_types, 0.5)?;
    ensure!(
        plain["works for"]
            == [
                ("Alice".to_owned(), "Acme Corp".to_owned()),
                ("Bob".to_owned(), "Acme Corp".to_owned()),
                ("Bob".to_owned(), "Northwind".to_owned()),
            ]
    );
    let confidence = pipeline.extract_relations_with_confidence(text, &relation_types, 0.5)?;
    let spans = pipeline.extract_relations_with_spans(text, &relation_types, 0.5)?;
    let both = pipeline.extract_relations_with_confidence_and_spans(text, &relation_types, 0.5)?;
    let options =
        pipeline.extract_relations_with_options(text, &relation_types, 0.5, true, true)?;
    ensure!(both == options);
    ensure!(confidence["works for"].len() == plain["works for"].len());
    ensure!(spans["works for"].len() == plain["works for"].len());

    let texts = [text, "Carol works for Example Labs."];
    let batch = pipeline.batch_extract_relations(&texts, &relation_types, 0.5, 1)?;
    ensure!(batch.len() == 2 && batch[0] == plain);
    ensure!(batch[1] == pipeline.extract_relations(texts[1], &relation_types, 0.5)?);
    let error = pipeline
        .batch_extract_relations::<&str>(&[], &relation_types, 0.5, 0)
        .unwrap_err()
        .to_string();
    ensure!(error.contains("batch_size") && error.contains("> 0"));

    let auto = AutoPipeline::from_dir(bundle)?;
    ensure!(auto.extract_relations(text, &relation_types, 0.5)? == plain);
    ensure!(auto.extract_relations_with_confidence(text, &relation_types, 0.5)? == confidence);
    ensure!(auto.extract_relations_with_spans(text, &relation_types, 0.5)? == spans);
    ensure!(auto.extract_relations_with_confidence_and_spans(text, &relation_types, 0.5)? == both);
    ensure!(auto.extract_relations_with_options(text, &relation_types, 0.5, true, true)? == both);
    ensure!(
        auto.batch_extract_relations(&texts, &relation_types, 0.5, 2)?
            .len()
            == 2
    );
    Ok(())
}

#[test]
fn relation_schema_options_preflight_and_joint_routing_are_supported() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let pipeline = BoundaryPipeline::from_dir(bundle)?;

    ensure!(pipeline.extract_relations("text", &[], 0.5)?.is_empty());
    let invalid_default = pipeline
        .extract_relations("must fail before inference", &[], f32::NAN)
        .unwrap_err()
        .to_string();
    ensure!(
        invalid_default.contains("default threshold"),
        "{invalid_default}"
    );

    let invalid_override = SchemaSpec {
        relations: vec![RelationSpec::new("works for").threshold(1.1)],
        ..SchemaSpec::default()
    };
    let error = pipeline
        .extract("must fail before inference", &invalid_override, 0.5)
        .unwrap_err()
        .to_string();
    ensure!(error.contains("relation threshold") && error.contains("[0,1]"));

    let text = "Alice works for Acme Corp while Bob works for Northwind.";
    let suppressed = SchemaSpec {
        relations: vec![RelationSpec::new("works for").threshold(1.0)],
        ..SchemaSpec::default()
    };
    let suppressed_output = pipeline.extract(text, &suppressed, 0.0)?;
    ensure!(suppressed_output.relations.len() == 1);
    ensure!(suppressed_output.relations["works for"].is_empty());
    // Empty input and the public relation formatter retain requested labels too.
    let empty = pipeline.extract_relations("", &["works for".to_owned()], 0.5)?;
    ensure!(empty.len() == 1);
    ensure!(empty["works for"].is_empty());

    let described = SchemaSpec {
        relations: vec![RelationSpec::new("works for").description("employment relationship")],
        ..SchemaSpec::default()
    };
    let described_output = pipeline.extract(text, &described, 0.5)?;
    ensure!(described_output.relations.contains_key("works for"));
    ensure!(
        !described_output
            .relations
            .contains_key("works for: employment relationship")
    );

    let duplicate = SchemaSpec {
        relations: vec![
            RelationSpec::new("works for"),
            RelationSpec::new("works for"),
        ],
        ..SchemaSpec::default()
    };
    let duplicated = pipeline.extract_with_spans(text, &duplicate, 0.5)?;
    ensure!(duplicated.relations.len() == 1);
    let duplicate_pairs = &duplicated.relations["works for"];
    let coordinates: BTreeSet<_> = duplicate_pairs
        .iter()
        .map(|pair| match (&pair.head, &pair.tail) {
            (
                FormattedEntitySpan::TextWithSpans {
                    start: head_start,
                    end: head_end,
                    ..
                },
                FormattedEntitySpan::TextWithSpans {
                    start: tail_start,
                    end: tail_end,
                    ..
                },
            ) => (*head_start, *head_end, *tail_start, *tail_end),
            _ => unreachable!("span extraction requested"),
        })
        .collect();
    ensure!(coordinates.len() == duplicate_pairs.len());

    // This is a routing/output-presence check, not an independent numerical oracle.
    let mixed = SchemaBuilder::new()
        .structure("metadata")
        .field(
            StructureFieldSpec::new("kind")
                .dtype(FieldDtype::Str)
                .choices(vec!["employment".to_owned()])
                .threshold(0.0),
        )
        .finish()
        .entities(vec!["person".to_owned(), "organization".to_owned()])
        .relations(vec!["works for".to_owned()])
        .classification(
            "sentiment",
            vec!["positive".to_owned(), "negative".to_owned()],
        )
        .build();
    let mixed_output = pipeline.extract_with_spans(text, &mixed, 0.0)?;
    ensure!(mixed_output.structures.contains_key("metadata"));
    ensure!(mixed_output.entities.len() == 2);
    ensure!(mixed_output.classifications.contains_key("sentiment"));
    let mixed_edges = &mixed_output.relations["works for"];
    ensure!(!mixed_edges.is_empty());
    for edge in mixed_edges {
        for endpoint in [&edge.head, &edge.tail] {
            let FormattedEntitySpan::TextWithSpans {
                text: surface,
                start,
                end,
            } = endpoint
            else {
                unreachable!("span extraction requested")
            };
            ensure!(
                text.get(*start..*end)
                    .is_some_and(|source| source.trim() == surface)
            );
        }
    }

    Ok(())
}
