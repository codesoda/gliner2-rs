use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use gliner2_rs::{
    boundary::{
        config::BoundaryRuntimeConfig, decode::OverlapPolicy, pipeline::BoundaryPipeline,
        preprocessing::BoundaryPreprocessingPolicy,
    },
    classification::FormattedClassification,
    embeddings::{extract_boundary_queries, extract_embeddings},
    encoder::Encoder,
    entities::{FormattedEntitySpan, FormattedEntityValue},
    pipeline::{AutoPipeline, BoundaryPipeline as PublicBoundaryPipeline},
    schema::format_input_with_mapping,
    schema_spec::{
        ClassificationOptions, EntitySpec, FieldDtype, RelationSpec, SchemaBuilder, SchemaSpec,
        StructureSpec,
    },
    tokenizer::RuntimeTokenizer,
};
use ndarray::{Array2, Array3, Axis};
use ndarray_npy::NpzReader;
use serde_json::Value;
use tempfile::tempdir;

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
        "missing boundary pipeline artifacts: {}",
        missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if env::var("GLINER2_REQUIRE_BOUNDARY_MODELS").as_deref() == Ok("1") {
        anyhow::bail!(message);
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn fixture_dir(full: bool) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(if full {
        "fixtures/gliner2.5-base-v1"
    } else {
        "fixtures/gliner2.5-base-v1-subset"
    })
}

fn corpus_cases() -> Result<BTreeMap<String, Value>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/parity/boundary_corpus.json");
    let document: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let cases = document["cases"].as_array().context("corpus cases")?;
    let mut by_id = BTreeMap::new();
    for case in cases {
        let id = case["id"].as_str().context("corpus case id")?.to_owned();
        ensure!(
            by_id.insert(id.clone(), case.clone()).is_none(),
            "duplicate corpus case ID {id}"
        );
    }
    Ok(by_id)
}

fn mixed_fixture() -> Result<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/boundary-mixed.json");
    serde_json::from_slice(&fs::read(path)?).map_err(Into::into)
}

fn schema_from_fixture(value: &Value) -> Result<SchemaSpec> {
    let schema = &value["schema_spec"];
    let mut builder = SchemaBuilder::new();
    if let Some(labels) = schema.get("entities").and_then(Value::as_array) {
        builder = builder.entities(
            labels
                .iter()
                .map(|label| label.as_str().context("entity label").map(str::to_owned))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    if let Some(config) = schema.get("entities_config").and_then(Value::as_object) {
        let mut specs = Vec::new();
        for (name, settings) in config {
            let dtype = match settings
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("list")
            {
                "str" => FieldDtype::Str,
                "list" => FieldDtype::List,
                other => anyhow::bail!("unknown fixture entity dtype {other}"),
            };
            let mut spec = EntitySpec::new(name).dtype(dtype);
            if let Some(threshold) = settings.get("threshold").and_then(Value::as_f64) {
                spec = spec.threshold(threshold as f32);
            }
            if let Some(description) = settings.get("description").and_then(Value::as_str) {
                spec = spec.description(description);
            }
            specs.push(spec);
        }
        builder = builder.entities(specs);
    }
    if let Some(tasks) = schema.get("classifications").and_then(Value::as_array) {
        for task in tasks {
            let name = task["task"].as_str().context("classification task")?;
            let labels = task["labels"]
                .as_array()
                .context("classification labels")?
                .iter()
                .map(|label| {
                    label
                        .as_str()
                        .context("classification label")
                        .map(str::to_owned)
                })
                .collect::<Result<Vec<_>>>()?;
            builder = builder.classification_with_options(
                name,
                labels,
                ClassificationOptions {
                    multi_label: task
                        .get("multi_label")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    cls_threshold: task
                        .get("cls_threshold")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.5) as f32,
                    ..ClassificationOptions::default()
                },
            );
        }
    }
    Ok(builder.build())
}

fn assert_confidence(actual: f32, expected: &Value, context: &str) -> Result<()> {
    let expected = expected.as_f64().context("expected confidence")? as f32;
    let difference = (actual - expected).abs();
    ensure!(
        difference <= 1e-3,
        "{context}: confidence {actual} != {expected}"
    );
    eprintln!("{context}: final_confidence_abs_error={difference:e}");
    Ok(())
}

fn assert_entity_span(actual: &FormattedEntitySpan, expected: &Value, context: &str) -> Result<()> {
    let FormattedEntitySpan::TextWithConfidenceAndSpans {
        text,
        confidence,
        start,
        end,
    } = actual
    else {
        anyhow::bail!("{context}: expected confidence-and-span formatting");
    };
    ensure!(
        text == expected["text"].as_str().context("expected text")?,
        "{context}: text"
    );
    ensure!(
        *start == expected["start"].as_u64().context("expected start")? as usize,
        "{context}: start"
    );
    ensure!(
        *end == expected["end"].as_u64().context("expected end")? as usize,
        "{context}: end"
    );
    assert_confidence(*confidence, &expected["confidence"], context)
}

fn assert_final_result(
    actual: &gliner2_rs::schema_spec::ExtractionResult,
    golden: &Value,
) -> Result<()> {
    let expected = &golden["final_result_utf8_offsets"];
    if let Some(entities) = expected.get("entities").and_then(Value::as_object) {
        ensure!(
            actual.entities.len() == entities.len(),
            "entity label count"
        );
        for (label, expected_value) in entities {
            let value = actual
                .entities
                .get(label)
                .with_context(|| format!("entity {label}"))?;
            match (value, expected_value) {
                (FormattedEntityValue::List(actual), Value::Array(expected)) => {
                    ensure!(actual.len() == expected.len(), "{label}: span count");
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        assert_entity_span(actual, expected, &format!("{label}[{index}]"))?;
                    }
                }
                (FormattedEntityValue::Single(Some(actual)), Value::Object(_)) => {
                    assert_entity_span(actual, expected_value, label)?;
                }
                (FormattedEntityValue::Single(None), Value::Null) => {}
                _ => anyhow::bail!("{label}: scalar/list shape differs from golden"),
            }
        }
    } else {
        ensure!(actual.entities.is_empty(), "unexpected entity output");
    }

    let expected_object = expected.as_object().context("final result golden object")?;
    let expected_tasks: BTreeSet<_> = expected_object
        .keys()
        .filter(|key| key.as_str() != "entities")
        .map(String::as_str)
        .collect();
    let actual_tasks: BTreeSet<_> = actual.classifications.keys().map(String::as_str).collect();
    ensure!(
        actual_tasks == expected_tasks,
        "classification task keys differ: actual={actual_tasks:?}, expected={expected_tasks:?}"
    );
    ensure!(
        actual.classifications.len() == expected_tasks.len(),
        "classification count"
    );
    for task in expected_tasks {
        let output = actual
            .classifications
            .get(task)
            .with_context(|| format!("classification {task}"))?;
        let golden = expected_object
            .get(task)
            .with_context(|| format!("classification golden {task}"))?;
        match output {
            FormattedClassification::SingleWithConfidence { label, confidence } => {
                ensure!(
                    label == golden["label"].as_str().context("golden label")?,
                    "{task}: label"
                );
                assert_confidence(*confidence, &golden["confidence"], task)?;
            }
            FormattedClassification::MultiWithConfidence(actual) => {
                let expected = golden.as_array().context("golden multi-label array")?;
                ensure!(actual.len() == expected.len(), "{task}: label count");
                for (index, ((label, confidence), expected)) in
                    actual.iter().zip(expected).enumerate()
                {
                    ensure!(
                        label == expected["label"].as_str().context("golden label")?,
                        "{task}[{index}]: label order"
                    );
                    assert_confidence(
                        *confidence,
                        &expected["confidence"],
                        &format!("{task}[{index}]"),
                    )?;
                }
            }
            _ => anyhow::bail!("{task}: expected confidence formatting"),
        }
    }
    Ok(())
}

#[test]
fn official_configs_are_runtime_compatible_and_invalid_variants_fail_early() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for name in ["small", "base", "multi"] {
        let source = root.join(format!("docs/checkpoints/{name}-config.json"));
        let dir = tempdir()?;
        fs::copy(source, dir.path().join("config.json"))?;
        BoundaryRuntimeConfig::from_dir(dir.path())?;
    }

    let base: Value =
        serde_json::from_slice(&fs::read(root.join("docs/checkpoints/base-config.json"))?)?;
    let invalid = [
        ("pair_temperature", serde_json::json!(1e-300), "positive"),
        (
            "boundary_ffn_multiplier",
            serde_json::json!("2.0"),
            "expected a number",
        ),
        (
            "boundary_ffn_multiplier",
            serde_json::json!(3),
            "unsupported boundary graph setting",
        ),
        ("rotary_base", serde_json::json!(false), "expected a number"),
        (
            "rotary_base",
            serde_json::json!(9999),
            "unsupported boundary graph setting",
        ),
    ];
    for (name, value, expected_error) in invalid {
        let mut config = base.clone();
        config["boundary_head"][name] = value;
        let dir = tempdir()?;
        fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
        let error = BoundaryRuntimeConfig::from_dir(dir.path())
            .unwrap_err()
            .to_string();
        ensure!(
            error.contains(name) && error.contains(expected_error),
            "unexpected {name} validation error: {error}"
        );
    }

    let mut integer_graph_values = base;
    integer_graph_values["boundary_head"]["boundary_ffn_multiplier"] = serde_json::json!(2);
    integer_graph_values["boundary_head"]["rotary_base"] = serde_json::json!(10000);
    let dir = tempdir()?;
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&integer_graph_values)?,
    )?;
    BoundaryRuntimeConfig::from_dir(dir.path())?;

    let dir = tempdir()?;
    fs::write(
        dir.path().join("config.json"),
        r#"{"architecture":"boundary","architecture_version":1,"token_pooling":"mean","boundary_head":{"candidate_pool":"shared"}}"#,
    )?;
    let error = BoundaryRuntimeConfig::from_dir(dir.path())
        .unwrap_err()
        .to_string();
    ensure!(error.contains("token_pooling") && error.contains("first"));
    Ok(())
}

#[test]
fn committed_and_opt_in_full_m4_fixtures_match_formatted_outputs() -> Result<()> {
    let corpus = corpus_cases()?;
    let eligible: BTreeMap<_, _> = corpus
        .into_iter()
        .filter(|(_, case)| !matches!(case["category"].as_str(), Some("json") | Some("relation")))
        .collect();
    ensure!(
        eligible.len() == 21,
        "expected exactly 21 M4-eligible corpus IDs, got {}",
        eligible.len()
    );

    let full = env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() == Ok("1");
    let expected_ids: BTreeSet<String> = if full {
        eligible.keys().cloned().collect()
    } else {
        ["classification_multi_task", "unicode_combining_emoji"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    };
    ensure!(
        expected_ids.len() == if full { 21 } else { 2 },
        "unexpected expected fixture ID count"
    );

    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let mut pipeline = BoundaryPipeline::from_dir(bundle)?;
    let directory = fixture_dir(full);
    let mut tested_ids = BTreeSet::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json")
            || path.file_name().and_then(|value| value.to_str()) == Some("manifest.json")
        {
            continue;
        }
        let golden: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let case_id = golden["case_id"].as_str().context("fixture case_id")?;
        let corpus_case = eligible.get(case_id);
        if corpus_case.is_none() {
            ensure!(
                matches!(golden["category"].as_str(), Some("json") | Some("relation")),
                "fixture {case_id} is absent from the exact M4-eligible corpus set"
            );
            continue;
        }
        let corpus_case = corpus_case.context("eligible corpus case")?;
        ensure!(
            tested_ids.insert(case_id.to_owned()),
            "duplicate tested fixture ID {case_id}"
        );
        let schema = schema_from_fixture(&golden)?;
        let overlap = corpus_case
            .get("overlap_policy")
            .and_then(Value::as_str)
            .map(str::parse)
            .transpose()?
            .unwrap_or(OverlapPolicy::Disallow);
        pipeline.set_overlap_policy(overlap);
        let text = golden["text"].as_str().context("fixture text")?;
        let threshold = golden["threshold"].as_f64().unwrap_or(0.5) as f32;
        let actual = pipeline.extract_with_confidence_and_spans(text, &schema, threshold)?;
        assert_final_result(&actual, &golden).with_context(|| path.display().to_string())?;
    }
    eprintln!("tested boundary fixture IDs: {tested_ids:?}");
    ensure!(
        tested_ids == expected_ids,
        "tested fixture IDs differ: tested={tested_ids:?}, expected={expected_ids:?}"
    );
    Ok(())
}

#[test]
fn final_result_assertion_rejects_a_missing_classification_without_a_model() -> Result<()> {
    let golden = serde_json::json!({
        "final_result_utf8_offsets": {
            "sentiment": {"label": "positive", "confidence": 0.9}
        }
    });
    let actual = gliner2_rs::schema_spec::ExtractionResult::default();
    let error = assert_final_result(&actual, &golden)
        .expect_err("dropping all classifications must fail")
        .to_string();
    ensure!(error.contains("classification task keys differ"));
    Ok(())
}

#[test]
fn mixed_public_oracle_fixture_has_pinned_model_free_integrity() -> Result<()> {
    let fixture = mixed_fixture()?;
    ensure!(fixture["format_version"] == 1);
    let provenance = &fixture["provenance"];
    ensure!(
        provenance["gliner2_repository"] == "https://github.com/fastino-ai/GLiNER2"
            && provenance["gliner2_commit"] == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
            && provenance["model_id"] == "fastino/gliner2.5-base-v1"
            && provenance["hf_revision"] == "78cea040597df251eedefa9d7ee2a756af39fe64"
    );
    let expected_hashes = [
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
    let source_hashes = provenance["source_file_sha256"]
        .as_object()
        .context("mixed source hashes")?;
    ensure!(source_hashes.len() == expected_hashes.len());
    for (name, hash) in expected_hashes {
        ensure!(source_hashes.get(name).and_then(Value::as_str) == Some(hash));
    }

    let cases = fixture["cases"].as_array().context("mixed cases")?;
    ensure!(cases.len() == 2, "mixed fixture case count");
    let ids: Vec<_> = cases
        .iter()
        .map(|case| case["case_id"].as_str().context("mixed case ID"))
        .collect::<Result<_>>()?;
    ensure!(ids == ["mixed_english", "mixed_unicode_repeated"]);
    ensure!(cases[0]["text"] == "Ada Lovelace enjoyed her visit to London.");
    let repeated_text = cases[1]["text"].as_str().context("repeated text")?;
    ensure!(
        repeated_text.matches("Zoë").count() == 2
            && repeated_text.matches("José").count() == 2
            && repeated_text.matches("São Paulo").count() == 2
            && !repeated_text.is_ascii()
    );

    for case in cases {
        let text = case["text"].as_str().context("mixed text")?;
        let schema = &case["schema_spec"];
        ensure!(
            schema["entities"]
                .as_array()
                .is_some_and(|labels| !labels.is_empty())
                && schema["classifications"]
                    .as_array()
                    .is_some_and(|tasks| !tasks.is_empty()),
            "mixed case must contain both task families"
        );
        let final_result = case["final_result_utf8_offsets"]
            .as_object()
            .context("mixed final result")?;
        let task_names: BTreeSet<_> = schema["classifications"]
            .as_array()
            .context("mixed classification specs")?
            .iter()
            .map(|task| task["task"].as_str().context("mixed task name"))
            .collect::<Result<_>>()?;
        let result_task_names: BTreeSet<_> = final_result
            .keys()
            .filter(|name| name.as_str() != "entities")
            .map(String::as_str)
            .collect();
        ensure!(task_names == result_task_names);

        for spans in final_result["entities"]
            .as_object()
            .context("mixed entities")?
            .values()
        {
            for span in spans.as_array().context("mixed entity span list")? {
                ensure!(span["offset_unit"] == "utf8_byte");
                let start = span["start"].as_u64().context("mixed span start")? as usize;
                let end = span["end"].as_u64().context("mixed span end")? as usize;
                ensure!(
                    text.get(start..end) == span["text"].as_str(),
                    "mixed UTF-8 span does not slice to its pinned text"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn mixed_tasks_match_pinned_public_oracle_outputs() -> Result<()> {
    let fixture = mixed_fixture()?;
    let cases = fixture["cases"].as_array().context("mixed cases")?;
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let mut pipeline = BoundaryPipeline::from_dir(bundle)?;
    for case in cases {
        pipeline.set_overlap_policy(OverlapPolicy::Disallow);
        let schema = schema_from_fixture(case)?;
        let text = case["text"].as_str().context("mixed fixture text")?;
        let threshold = case["threshold"].as_f64().context("mixed threshold")? as f32;
        let actual = pipeline.extract_with_confidence_and_spans(text, &schema, threshold)?;
        assert_final_result(&actual, case)
            .with_context(|| case["case_id"].as_str().unwrap_or("unknown").to_owned())?;
    }
    Ok(())
}

#[test]
fn formatted_input_and_gathered_states_match_committed_stage_fixture() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let directory = fixture_dir(false);
    let golden: Value =
        serde_json::from_slice(&fs::read(directory.join("unicode_combining_emoji.json"))?)?;
    let schema_tokens: Vec<Vec<String>> = serde_json::from_value(golden["schema_tokens"].clone())?;
    let prepared = BoundaryPreprocessingPolicy::default()
        .prepare(golden["text"].as_str().context("fixture text")?, &[]);
    let expected_tokens: Vec<String> = serde_json::from_value(golden["text_tokens"].clone())?;
    ensure!(
        prepared.text_tokens == expected_tokens,
        "prepared text tokens"
    );

    let tokenizer = RuntimeTokenizer::from_dir(&bundle)?;
    let formatted = format_input_with_mapping(&tokenizer, &schema_tokens, &prepared.text_tokens)?;
    let mut npz = NpzReader::new(File::open(directory.join("unicode_combining_emoji.npz"))?)?;
    let expected_ids: Array2<i64> = npz.by_name("input_ids")?;
    ensure!(
        formatted.input_ids == expected_ids.row(0).to_vec(),
        "formatted input ids"
    );

    let sequence = formatted.input_ids.len();
    let hidden = Encoder::new(bundle.join("encoder.onnx"))?.infer(
        Array2::from_shape_vec((1, sequence), formatted.input_ids.clone())?,
        Array2::from_shape_vec((1, sequence), formatted.attention_mask.clone())?,
    )?;
    let extracted = extract_embeddings(&hidden, &formatted, schema_tokens.len())?;
    let queries = extract_boundary_queries(&hidden, &formatted, schema_tokens.len())?;
    let expected_text: Array3<f32> = npz.by_name("text_states")?;
    let expected_queries: Array3<f32> = npz.by_name("query_states")?;
    for (name, actual, expected) in [
        (
            "text_states",
            extracted.text_emb.view(),
            expected_text.index_axis(Axis(0), 0),
        ),
        (
            "query_states",
            queries.query_emb.view(),
            expected_queries.index_axis(Axis(0), 0),
        ),
    ] {
        ensure!(actual.shape() == expected.shape(), "{name} shape");
        let max_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        ensure!(max_error <= 1e-4, "{name} maximum error {max_error}");
    }
    Ok(())
}

#[test]
fn q0_relation_preflight_raw_api_and_adapter_swap_are_explicit_and_lossless() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let mut pipeline = BoundaryPipeline::from_dir(&bundle)?;
    let mixed = SchemaBuilder::new()
        .entities(vec!["person".to_owned(), "location".to_owned()])
        .classification(
            "sentiment",
            vec![
                "positive".to_owned(),
                "negative".to_owned(),
                "neutral".to_owned(),
            ],
        )
        .build();
    let text = "Ada Lovelace enjoyed her visit to London.";
    let before_load = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;

    let classification_only = SchemaBuilder::new()
        .classification(
            "sentiment",
            vec!["positive".to_owned(), "negative".to_owned()],
        )
        .build();
    ensure!(
        pipeline
            .extract("A genuinely useful release.", &classification_only, 0.5)?
            .entities
            .is_empty()
    );

    let empty_structure = SchemaSpec {
        structures: vec![StructureSpec {
            name: "record".to_owned(),
            fields: Vec::new(),
        }],
        ..SchemaSpec::default()
    };
    ensure!(
        pipeline
            .extract("fieldless structures are valid", &empty_structure, 0.5)?
            .structures
            .is_empty()
    );

    let invalid_relation = SchemaSpec {
        relations: vec![RelationSpec::new("related to").threshold(f32::NAN)],
        ..SchemaSpec::default()
    };
    let error = pipeline
        .extract("must not encode", &invalid_relation, 0.5)
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("relation threshold") && error.contains("[0,1]"),
        "{error}"
    );

    let adapter = tempdir()?;
    fs::copy(
        bundle.join("encoder.onnx"),
        adapter.path().join("encoder.onnx"),
    )?;
    pipeline.load_adapter(adapter.path())?;
    ensure!(pipeline.has_adapter());
    let after_first_load = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;
    ensure!(
        after_first_load == before_load,
        "first identical adapter load changed output"
    );

    pipeline.load_adapter(adapter.path())?;
    let after_second_load = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;
    ensure!(
        after_second_load == before_load,
        "second identical adapter load changed output"
    );

    let adapter_state = pipeline
        .adapter_config()
        .context("adapter state after second load")?
        .clone();
    let missing_adapter = tempdir()?;
    let error = pipeline
        .load_adapter(missing_adapter.path())
        .unwrap_err()
        .to_string();
    ensure!(error.contains("missing `encoder.onnx`"));
    ensure!(
        pipeline.adapter_config() == Some(&adapter_state),
        "failed adapter load changed adapter metadata"
    );
    let after_failed_load = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;
    ensure!(
        after_failed_load == after_second_load,
        "failed adapter load changed inference state"
    );

    pipeline.unload_adapter()?;
    ensure!(!pipeline.has_adapter());
    let after_unload = pipeline.extract_with_confidence_and_spans(text, &mixed, 0.5)?;
    ensure!(
        after_unload == before_load,
        "adapter unload did not restore base output"
    );

    let auto = AutoPipeline::from_dir(&bundle)?;
    let error = auto.infer_raw(&[], &[]).unwrap_err().to_string();
    ensure!(error.contains("legacy span-head output") && error.contains("unsupported"));

    // Both public paths name the same completed implementation.
    let _: Option<PublicBoundaryPipeline> = None;
    Ok(())
}
