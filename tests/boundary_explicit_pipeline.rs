use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use gliner2_rs::{
    boundary::{BoundaryPipeline, ExplicitSpanScores, preprocessing::WordSplitter},
    pipeline::{AutoPipeline, SpanPipeline},
};
use serde_json::Value;

const CASE_IDS: [&str; 3] = [
    "english_lines_whitespace",
    "unicode_multiline_whitespace",
    "unicode_compact_char",
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
    ];
    let missing: Vec<_> = required
        .iter()
        .map(|name| bundle.join(name))
        .filter(|path| !path.is_file())
        .collect();
    if missing.is_empty() {
        return Ok(Some(bundle));
    }
    let message = format!("missing public explicit-span artifacts: {missing:?}");
    if strict_boundary() {
        return Err(anyhow!(message));
    }
    eprintln!("SKIP: {message}");
    Ok(None)
}

fn fixture() -> Result<Value> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/public-explicit-vectors.json");
    let bytes = fs::read(&path)?;
    ensure!(
        bytes.len() <= 15_000,
        "public explicit fixture is {} bytes, exceeds 15,000",
        bytes.len()
    );
    serde_json::from_slice(&bytes).with_context(|| path.display().to_string())
}

fn labels(case: &Value) -> Result<Vec<String>> {
    case["labels"]
        .as_array()
        .context("fixture labels")?
        .iter()
        .map(|label| label.as_str().context("fixture label").map(str::to_owned))
        .collect()
}

fn spans(case: &Value) -> Result<Vec<[usize; 2]>> {
    case["spans"]
        .as_array()
        .context("fixture spans")?
        .iter()
        .map(|span| {
            let bounds = span["requested_utf8"]
                .as_array()
                .context("requested UTF-8 span")?;
            ensure!(bounds.len() == 2, "requested span width");
            Ok([
                bounds[0].as_u64().context("requested start")? as usize,
                bounds[1].as_u64().context("requested end")? as usize,
            ])
        })
        .collect()
}

fn close(actual: f32, expected: f32, atol: f32, rtol: f32) -> bool {
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

fn assert_case(pipeline: &mut BoundaryPipeline, case: &Value) -> Result<Vec<ExplicitSpanScores>> {
    let splitter = match case["splitter"].as_str().context("fixture splitter")? {
        "whitespace" => WordSplitter::Whitespace,
        "char" => WordSplitter::Char,
        other => return Err(anyhow!("unknown fixture splitter {other:?}")),
    };
    pipeline.set_word_splitter(splitter)?;
    let text = case["original_text"].as_str().context("fixture text")?;
    let labels = labels(case)?;
    let spans = spans(case)?;
    let actual = pipeline.score_explicit_spans(text, &labels, &spans)?;
    ensure!(actual.len() == labels.len(), "group count");

    let expected_logits = case["logits"].as_array().context("fixture logits")?;
    let expected_probabilities = case["probabilities"]
        .as_array()
        .context("fixture probabilities")?;
    let span_specs = case["spans"].as_array().context("fixture span specs")?;
    for (query, group) in actual.iter().enumerate() {
        ensure!(group.label == labels[query], "label order at query {query}");
        ensure!(
            group.spans.len() == spans.len(),
            "span count at query {query}"
        );
        let logits = expected_logits[query].as_array().context("query logits")?;
        let probabilities = expected_probabilities[query]
            .as_array()
            .context("query probabilities")?;
        ensure!(logits.len() == spans.len() && probabilities.len() == spans.len());
        for (candidate, score) in group.spans.iter().enumerate() {
            let [start, end] = spans[candidate];
            let expected_text = span_specs[candidate]["text"]
                .as_str()
                .context("fixture span text")?;
            ensure!(
                (score.start, score.end, score.text.as_str()) == (start, end, expected_text),
                "query {query} candidate {candidate}: source span/order mismatch"
            );
            ensure!(text.get(start..end) == Some(expected_text));
            ensure!(score.logit.is_finite(), "non-finite raw logit");
            ensure!(
                score.confidence.is_finite() && (0.0..=1.0).contains(&score.confidence),
                "invalid calibrated confidence"
            );
            let expected_logit = logits[candidate].as_f64().context("fixture logit")? as f32;
            let expected_probability = probabilities[candidate]
                .as_f64()
                .context("fixture probability")? as f32;
            ensure!(
                close(score.logit, expected_logit, 1e-4, 1e-3),
                "query {query} candidate {candidate}: raw logit {} != {expected_logit}",
                score.logit
            );
            ensure!(
                (score.confidence - expected_probability).abs() <= 1e-3,
                "query {query} candidate {candidate}: confidence {} != {expected_probability}",
                score.confidence
            );
            let recalibrated = 1.0 / (1.0 + (-score.logit).exp());
            ensure!(
                (score.confidence - recalibrated).abs() <= f32::EPSILON * 2.0,
                "confidence was not calibrated from the returned raw logit"
            );
        }
    }
    Ok(actual)
}

#[test]
fn compact_fixture_has_pinned_source_provenance_and_closed_case_set() -> Result<()> {
    let fixture = fixture()?;
    ensure!(fixture["format_version"] == 1);
    ensure!(
        fixture["oracle"]
            == "untouched BoundaryHead.score_explicit_spans after upstream preprocessing and encoding"
    );
    let provenance = &fixture["provenance"];
    ensure!(
        provenance["gliner2_commit"] == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
            && provenance["model_id"] == "fastino/gliner2.5-base-v1"
            && provenance["hf_revision"] == "78cea040597df251eedefa9d7ee2a756af39fe64"
            && provenance["onnx_used_as_reference"] == false
            && provenance["python_executable"] == "scripts/export/env/.venv/bin/python"
            && provenance["generator_sha256"]
                == "caa0e3c89f165a9fc0178594dd3b4ea87b445f34a872f684629aa460c1f7e64f"
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
        .context("source hashes")?;
    ensure!(source_hashes.len() == expected_hashes.len());
    for (name, expected) in expected_hashes {
        ensure!(source_hashes.get(name).and_then(Value::as_str) == Some(expected));
    }

    let cases = fixture["cases"].as_array().context("fixture cases")?;
    let actual_ids: Vec<_> = cases
        .iter()
        .map(|case| case["case_id"].as_str().context("case ID"))
        .collect::<Result<_>>()?;
    ensure!(actual_ids == CASE_IDS, "fixture case IDs must fail closed");
    ensure!(cases.iter().any(|case| case["splitter"] == "whitespace"));
    ensure!(cases.iter().any(|case| case["splitter"] == "char"));
    Ok(())
}

#[test]
fn public_explicit_api_matches_independent_upstream_oracle_and_preserves_axes() -> Result<()> {
    let Some(bundle) = bundle()? else {
        return Ok(());
    };
    let fixture = fixture()?;
    let cases = fixture["cases"].as_array().context("fixture cases")?;
    let mut pipeline = BoundaryPipeline::from_dir(&bundle)?;
    for case in cases {
        assert_case(&mut pipeline, case)
            .with_context(|| case["case_id"].as_str().unwrap_or("unknown").to_owned())?;
    }

    pipeline.set_word_splitter(WordSplitter::Whitespace)?;
    let text = "Alice met Bob.";
    let duplicate_labels = vec!["person".into(), "person".into(), "organization".into()];
    let duplicate_spans = [[10, 13], [0, 5], [10, 13]];
    let duplicated = pipeline.score_explicit_spans(text, &duplicate_labels, &duplicate_spans)?;
    ensure!(
        duplicated
            .iter()
            .map(|group| group.label.as_str())
            .collect::<Vec<_>>()
            == ["person", "person", "organization"]
    );
    for group in &duplicated {
        ensure!(
            group
                .spans
                .iter()
                .map(|span| (span.start, span.end, span.text.as_str()))
                .collect::<Vec<_>>()
                == [(10, 13, "Bob"), (0, 5, "Alice"), (10, 13, "Bob")]
        );
        ensure!(group.spans.iter().all(|span| span.logit.is_finite()));
    }

    ensure!(
        pipeline
            .score_explicit_spans(text, &[], &[[0, 5]])?
            .is_empty()
    );
    let empty_spans = pipeline.score_explicit_spans(text, &duplicate_labels, &[])?;
    ensure!(empty_spans.len() == duplicate_labels.len());
    ensure!(empty_spans.iter().all(|group| group.spans.is_empty()));

    let invalid_utf8 = pipeline
        .score_explicit_spans("café", &[], &[[0, 4]])
        .unwrap_err()
        .to_string();
    ensure!(invalid_utf8.contains("UTF-8"), "{invalid_utf8}");
    let partial = pipeline
        .score_explicit_spans("café", &["word".into()], &[[1, 5]])
        .unwrap_err()
        .to_string();
    ensure!(partial.contains("not aligned"), "{partial}");

    let long_text = vec!["word"; 4_100].join(" ");
    let last_start = long_text.rfind("word").context("last word")?;
    let truncated = pipeline
        .score_explicit_spans(
            &long_text,
            &["word".into()],
            &[[last_start, last_start + 4]],
        )
        .unwrap_err()
        .to_string();
    ensure!(truncated.contains("retained"), "{truncated}");

    let auto = AutoPipeline::Boundary(Box::new(pipeline));
    let forwarded = auto.score_explicit_spans(text, &["person".into()], &[[0, 5]])?;
    ensure!(forwarded.len() == 1 && forwarded[0].spans[0].text == "Alice");
    Ok(())
}

#[test]
fn span_architecture_rejects_explicit_scoring_without_boundary_emulation() -> Result<()> {
    let span_onnx = root().join("onnx/gliner2-base-v1");
    let span_model = root().join("models/gliner2-base-v1");
    let missing: Vec<_> = [
        span_model.join("config.json"),
        span_model.join("tokenizer.json"),
        span_onnx.join("encoder.onnx"),
        span_onnx.join("extractor_padded.onnx"),
    ]
    .into_iter()
    .filter(|path| !path.is_file())
    .collect();
    if !missing.is_empty() {
        ensure!(
            env::var("GLINER2_REQUIRE_MODELS").as_deref() != Ok("1"),
            "missing required v2 explicit-unsupported artifacts: {missing:?}"
        );
        eprintln!("SKIP: missing v2 explicit-unsupported artifacts: {missing:?}");
        return Ok(());
    }

    let span = AutoPipeline::Span(Box::new(SpanPipeline::new(
        &span_model,
        span_onnx.join("encoder.onnx"),
        span_onnx.join("extractor_padded.onnx"),
    )?));
    let error = span
        .score_explicit_spans("Alice", &["person".into()], &[[0, 5]])
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("unsupported for span models") && error.contains("boundary architecture"),
        "{error}"
    );
    Ok(())
}
