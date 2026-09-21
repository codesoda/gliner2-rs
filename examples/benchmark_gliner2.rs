//! Reproducible CPU entity-extraction benchmark for GLiNER2 base and GLiNER2.5 base.
//!
//! This is an explicit, opt-in benchmark. It never downloads models. See
//! `docs/BENCHMARK-gliner2.5.md` for the quiet-machine procedure and report
//! interpretation.

use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    hint::black_box,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use gliner2_rs::{
    boundary::preprocessing::{BoundaryPreprocessingPolicy, WordSplitter},
    config::ModelConfig,
    entities::build_entities_schema_tokens,
    preprocessing::PreprocessingPolicy,
    schema::{Segment, format_input_with_mapping},
    tokenizer::RuntimeTokenizer,
};
use gliner2_rs::{
    entities::EntityMatches,
    pipeline::{AutoPipeline, SpanPipeline},
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const WORD_COUNTS: [usize; 3] = [50, 500, 3_000];
const DEFAULT_LABELS: [&str; 4] = ["person", "organization", "location", "date"];
const SESSION_INTRA_OP_THREADS: usize = 4;
const V2_GRAPHS: [&str; 3] = ["encoder.onnx", "extractor_padded.onnx", "classifier.onnx"];
const V25_GRAPHS: [&str; 7] = [
    "encoder.onnx",
    "classifier.onnx",
    "boundary_marginals.onnx",
    "boundary_scorer.onnx",
    "boundary_explicit_scorer.onnx",
    "boundary_records.onnx",
    "boundary_relations.onnx",
];
const TEXT_WORDS: [&str; 24] = [
    "Alice",
    "from",
    "Acme",
    "Corporation",
    "visited",
    "Paris",
    "on",
    "Monday",
    ".",
    "Bob",
    "joined",
    "Globex",
    "in",
    "Berlin",
    "on",
    "Tuesday",
    ".",
    "Carol",
    "met",
    "Dana",
    "at",
    "Initech",
    "in",
    "London",
];

#[derive(Debug, Clone, PartialEq)]
struct Config {
    model: String,
    v2_model_dir: PathBuf,
    v2_onnx_dir: PathBuf,
    v25_bundle_dir: PathBuf,
    output: PathBuf,
    warmups: usize,
    repetitions: usize,
    threshold: f32,
    labels: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self {
            model: "both".to_owned(),
            v2_model_dir: root.join("models/gliner2-base-v1"),
            v2_onnx_dir: root.join("onnx/gliner2-base-v1"),
            v25_bundle_dir: root.join("onnx/gliner2.5-base-v1"),
            output: root.join("benchmark-gliner2.json"),
            warmups: 3,
            repetitions: 10,
            threshold: 0.5,
            labels: DEFAULT_LABELS
                .iter()
                .map(|label| (*label).to_owned())
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq)]
enum CliAction {
    Run(Config),
    Help,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    benchmark: &'static str,
    generated_at_unix_seconds: u64,
    configuration: ReportConfig,
    texts: Vec<TextIdentity>,
    provenance: Provenance,
    models: Vec<ModelReport>,
}

#[derive(Serialize)]
struct ReportConfig {
    word_counts: [usize; 3],
    labels: Vec<String>,
    threshold: f32,
    warmups: usize,
    repetitions: usize,
    session_intra_op_threads: usize,
    graph_optimization: &'static str,
    timing_clock: &'static str,
    timing_scope: &'static str,
    execution_order: Vec<&'static str>,
    process_isolation: &'static str,
    execution_provider: &'static str,
    session_inter_op_threads: Option<usize>,
    session_inter_op_policy: &'static str,
    load_timing_scope: &'static str,
    resource_measurement: &'static str,
}

#[derive(Serialize)]
struct TextIdentity {
    word_count: usize,
    utf8_bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct Provenance {
    crate_version: &'static str,
    git_commit: Option<String>,
    git_worktree_dirty: Option<bool>,
    rustc_verbose_version: Option<String>,
    cargo_profile: &'static str,
    target_os: &'static str,
    target_arch: &'static str,
    cpu_model: Option<String>,
    logical_cpus: Option<usize>,
    physical_cores: Option<usize>,
    power_mode: Option<String>,
    power_mode_source: &'static str,
    memory_bytes: Option<u64>,
    ort_crate_version: Option<String>,
    native_onnx_runtime: String,
    environment: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ModelReport {
    name: &'static str,
    architecture: &'static str,
    metadata_dir: PathBuf,
    graph_dir: PathBuf,
    graphs: Vec<GraphIdentity>,
    source: Value,
    peak_process_rss: Option<Value>,
    load_time_ns: u64,
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
struct GraphIdentity {
    name: String,
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct CaseReport {
    word_count: usize,
    warmup_iterations: usize,
    per_run_ns: Vec<u64>,
    median_ns: u64,
    p90_ns: u64,
    p95_ns: u64,
    min_ns: u64,
    max_ns: u64,
    submitted_words_per_second_at_median: f64,
    tokenization: Option<TokenizationReport>,
    label_group_count: usize,
    entity_count: usize,
    output_sha256: String,
}

#[derive(Serialize)]
struct TokenizationReport {
    tokenizer_sha256: String,
    retained_text_words: usize,
    text_only_subwords: usize,
    schema_plus_text_subwords: usize,
    max_original_words: Option<usize>,
    scope: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutputIntegrity {
    label_group_count: usize,
    entity_count: usize,
    sha256: String,
}

fn main() -> Result<()> {
    let action = parse_args(env::args().skip(1))?;
    let config = match action {
        CliAction::Help => {
            print_help();
            return Ok(());
        }
        CliAction::Run(config) => config,
    };

    let texts: Vec<String> = WORD_COUNTS.into_iter().map(generate_text).collect();
    for (&word_count, text) in WORD_COUNTS.iter().zip(&texts) {
        ensure!(
            text.split_whitespace().count() == word_count,
            "internal text generator produced the wrong word count"
        );
    }

    ensure!(
        !config.output.exists() && !config.output.is_symlink(),
        "report already exists: {}; choose a new output",
        config.output.display()
    );
    // Snapshot the small manifest before loading; verify all bytes after timings.
    // Never benchmark while another process is modifying model files.
    let manifest_bytes = if config.model != "v2" {
        let bytes = fs::read(config.v25_bundle_dir.join("export_manifest.json"))?;
        check_v25_identity(&serde_json::from_slice(&bytes)?)?;
        Some(bytes)
    } else {
        None
    };
    eprintln!(
        "benchmarking {} ({} warmups, {} measured repetitions)",
        config.model, config.warmups, config.repetitions
    );
    let mut models = Vec::new();
    if config.model != "v25" {
        models.push(benchmark_v2(&config, &texts)?);
    }
    if config.model != "v2" {
        models.push(benchmark_v25(&config, &texts)?);
    }

    // Provenance reads and token counts are outside ALL timing measurements.
    // Load timings are warm-filesystem/non-disk-cold, even in a fresh process.
    for model in &mut models {
        let boundary = model.architecture == "boundary";
        model.graphs = graph_identities(
            &model.graph_dir,
            if boundary { &V25_GRAPHS } else { &V2_GRAPHS },
        )?;
        model.source = if boundary {
            verify_v25_source(
                &model.graph_dir,
                manifest_bytes
                    .as_deref()
                    .context("missing manifest snapshot")?,
                &model.graphs,
            )?
        } else {
            verify_v2_source(&model.metadata_dir, &model.graphs)?
        };
        add_token_counts(model, &texts, &config.labels)?;
    }

    let text_identities = WORD_COUNTS
        .iter()
        .zip(&texts)
        .map(|(&word_count, text)| TextIdentity {
            word_count,
            utf8_bytes: text.len(),
            sha256: sha256_bytes(text.as_bytes()),
        })
        .collect();

    let report = Report {
        schema_version: 2,
        benchmark: "cpu-end-to-end-entity-extraction",
        generated_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_secs(),
        configuration: ReportConfig {
            word_counts: WORD_COUNTS,
            labels: config.labels.clone(),
            threshold: config.threshold,
            warmups: config.warmups,
            repetitions: config.repetitions,
            session_intra_op_threads: SESSION_INTRA_OP_THREADS,
            graph_optimization: "Level3",
            timing_clock: "std::time::Instant",
            timing_scope: "one public extract_entities call; output checks and graph hashing excluded",
            execution_order: models.iter().map(|model| model.name).collect(),
            process_isolation: if models.len() == 1 {
                "one selected model in this process"
            } else {
                "shared process; non-authoritative runtime-init/RSS comparison"
            },
            execution_provider: "CPUExecutionProvider (library default; no accelerator provider registered)",
            session_inter_op_threads: None,
            session_inter_op_policy: "not explicitly configured; ONNX Runtime default (not measured)",
            load_timing_scope: "constructor including runtime/session initialization; warm filesystem/non-disk-cold; no cache eviction",
            resource_measurement: "peak_process_rss is null in direct runs; authoritative shell wrapper measures whole-process high-water RSS with /usr/bin/time, including load, inference, checks and hashing",
        },
        texts: text_identities,
        provenance: collect_provenance(),
        models,
    };

    write_json_atomic(&config.output, &report)?;
    eprintln!("wrote {}", config.output.display());
    Ok(())
}

fn write_json_atomic(path: &Path, report: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    // Same-directory temporary + no-clobber publication: never expose partial JSON.
    let mut output = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut output, report)
        .with_context(|| format!("failed to serialize report {}", path.display()))?;
    output.write_all(b"\n")?;
    output.as_file().sync_all()?;
    output.persist_noclobber(path)?;
    Ok(())
}

fn benchmark_v2(config: &Config, texts: &[String]) -> Result<ModelReport> {
    let encoder = required_file(&config.v2_onnx_dir, "encoder.onnx")?;
    let extractor = required_file(&config.v2_onnx_dir, "extractor_padded.onnx")?;
    let classifier = required_file(&config.v2_onnx_dir, "classifier.onnx")?;

    let started = Instant::now();
    let pipeline = SpanPipeline::new(&config.v2_model_dir, encoder, extractor)
        .with_context(|| {
            format!(
                "failed to load v2 metadata from {} and graphs from {}",
                config.v2_model_dir.display(),
                config.v2_onnx_dir.display()
            )
        })?
        .with_classifier(classifier)
        .context("failed to load v2 classifier")?;
    let load_time_ns = duration_ns(started.elapsed())?;
    let pipeline = AutoPipeline::Span(Box::new(pipeline));
    let cases = benchmark_cases("gliner2-base-v1", &pipeline, config, texts)?;
    drop(pipeline);

    Ok(ModelReport {
        name: "gliner2-base-v1",
        architecture: "span",
        metadata_dir: canonical_or_original(&config.v2_model_dir),
        graph_dir: canonical_or_original(&config.v2_onnx_dir),
        graphs: Vec::new(),
        source: Value::Null,
        peak_process_rss: None,
        load_time_ns,
        cases,
    })
}

fn benchmark_v25(config: &Config, texts: &[String]) -> Result<ModelReport> {
    let started = Instant::now();
    let pipeline = AutoPipeline::from_dir(&config.v25_bundle_dir).with_context(|| {
        format!(
            "failed to load complete GLiNER2.5 bundle {}",
            config.v25_bundle_dir.display()
        )
    })?;
    let load_time_ns = duration_ns(started.elapsed())?;
    let cases = benchmark_cases("gliner2.5-base-v1", &pipeline, config, texts)?;
    drop(pipeline);

    Ok(ModelReport {
        name: "gliner2.5-base-v1",
        architecture: "boundary",
        metadata_dir: canonical_or_original(&config.v25_bundle_dir),
        graph_dir: canonical_or_original(&config.v25_bundle_dir),
        graphs: Vec::new(),
        source: Value::Null,
        peak_process_rss: None,
        load_time_ns,
        cases,
    })
}

fn benchmark_cases(
    model_name: &str,
    pipeline: &AutoPipeline,
    config: &Config,
    texts: &[String],
) -> Result<Vec<CaseReport>> {
    WORD_COUNTS
        .iter()
        .zip(texts)
        .map(|(&word_count, text)| {
            eprintln!("  {model_name}: {word_count} words");
            let mut expected: Option<OutputIntegrity> = None;

            for iteration in 0..config.warmups {
                let (_, integrity) = run_once(pipeline, text, &config.labels, config.threshold)
                    .with_context(|| {
                        format!(
                            "{model_name} failed during {word_count}-word warmup {}",
                            iteration + 1
                        )
                    })?;
                check_integrity(model_name, word_count, &mut expected, integrity)?;
            }

            let mut timings = Vec::with_capacity(config.repetitions);
            for iteration in 0..config.repetitions {
                let (elapsed, integrity) =
                    run_once(pipeline, text, &config.labels, config.threshold).with_context(
                        || {
                            format!(
                                "{model_name} failed during {word_count}-word measured run {}",
                                iteration + 1
                            )
                        },
                    )?;
                check_integrity(model_name, word_count, &mut expected, integrity)?;
                timings.push(elapsed);
            }

            let integrity = expected.context("benchmark configured without any runs")?;
            let (median_ns, p95_ns) = timing_statistics(&timings)?;
            ensure!(
                median_ns > 0,
                "zero median duration cannot yield throughput"
            );
            Ok(CaseReport {
                word_count,
                warmup_iterations: config.warmups,
                p90_ns: nearest_rank(&timings, 90)?,
                min_ns: *timings.iter().min().context("no timings")?,
                max_ns: *timings.iter().max().context("no timings")?,
                submitted_words_per_second_at_median: word_count as f64 * 1e9 / median_ns as f64,
                tokenization: None,
                per_run_ns: timings,
                median_ns,
                p95_ns,
                label_group_count: integrity.label_group_count,
                entity_count: integrity.entity_count,
                output_sha256: integrity.sha256,
            })
        })
        .collect()
}

fn run_once(
    pipeline: &AutoPipeline,
    text: &str,
    labels: &[String],
    threshold: f32,
) -> Result<(u64, OutputIntegrity)> {
    let started = Instant::now();
    let output =
        pipeline.extract_entities(black_box(text), black_box(labels), black_box(threshold))?;
    let elapsed = duration_ns(started.elapsed())?;

    // Consume the complete nested result before deriving explicit count/hash
    // checks. This keeps every model output observable in optimized builds.
    let output = black_box(output);
    let integrity = inspect_output(&output, labels)?;
    black_box(&integrity);
    Ok((elapsed, integrity))
}

fn inspect_output(output: &[EntityMatches], labels: &[String]) -> Result<OutputIntegrity> {
    ensure!(
        output.len() == labels.len(),
        "entity result has {} label groups; expected {}",
        output.len(),
        labels.len()
    );

    let mut hasher = Sha256::new();
    let mut entity_count = 0usize;
    for (group, expected_label) in output.iter().zip(labels) {
        ensure!(
            group.label == *expected_label,
            "entity result label {:?} does not match requested label {:?}",
            group.label,
            expected_label
        );
        hash_sized_bytes(&mut hasher, group.label.as_bytes())?;
        hash_u64(&mut hasher, group.spans.len())?;
        entity_count = entity_count
            .checked_add(group.spans.len())
            .context("entity count overflow")?;
        for span in &group.spans {
            hash_u64(&mut hasher, span.start)?;
            hash_u64(&mut hasher, span.end)?;
            hash_sized_bytes(&mut hasher, span.text.as_bytes())?;
            hasher.update(span.score.to_bits().to_be_bytes());
        }
    }

    Ok(OutputIntegrity {
        label_group_count: output.len(),
        entity_count,
        sha256: hex_digest(hasher.finalize()),
    })
}

fn check_integrity(
    model_name: &str,
    word_count: usize,
    expected: &mut Option<OutputIntegrity>,
    actual: OutputIntegrity,
) -> Result<()> {
    if let Some(expected) = expected {
        ensure!(
            *expected == actual,
            "{model_name} produced inconsistent output across {word_count}-word runs: expected {expected:?}, got {actual:?}"
        );
    } else {
        *expected = Some(actual);
    }
    Ok(())
}

fn timing_statistics(values: &[u64]) -> Result<(u64, u64)> {
    ensure!(!values.is_empty(), "cannot summarize zero timing samples");
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        sorted[middle - 1] + (sorted[middle] - sorted[middle - 1]) / 2
    } else {
        sorted[middle]
    };
    let p95_rank = sorted.len().saturating_mul(95).div_ceil(100);
    let p95 = sorted[p95_rank.saturating_sub(1)];
    Ok((median, p95))
}

fn nearest_rank(values: &[u64], percentile: usize) -> Result<u64> {
    ensure!(
        !values.is_empty() && (1..=100).contains(&percentile),
        "invalid percentile samples/rank"
    );
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Ok(sorted[sorted.len().saturating_mul(percentile).div_ceil(100) - 1])
}

fn generate_text(word_count: usize) -> String {
    assert!(word_count > 0, "benchmark text must contain words");
    let mut words = Vec::with_capacity(word_count);
    for index in 0..word_count - 1 {
        words.push(TEXT_WORDS[index % TEXT_WORDS.len()]);
    }
    words.push(".");
    words.join(" ")
}

fn parse_args<I, S>(args: I) -> Result<CliAction>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut config = Config::default();
    let mut labels = Vec::new();
    let mut explicit_labels = false;
    let mut args = args.into_iter().map(Into::into);

    while let Some(argument) = args.next() {
        let (flag, inline_value) = match argument.split_once('=') {
            Some((flag, value)) => (flag, Some(value.to_owned())),
            None => (argument.as_str(), None),
        };
        if flag == "--help" || flag == "-h" {
            ensure!(inline_value.is_none(), "{flag} does not accept a value");
            return Ok(CliAction::Help);
        }
        if !matches!(
            flag,
            "--model"
                | "--v2-model-dir"
                | "--v2-onnx-dir"
                | "--v25-bundle-dir"
                | "--output"
                | "--warmups"
                | "--repetitions"
                | "--threshold"
                | "--label"
        ) {
            if flag.starts_with('-') {
                bail!("unrecognized argument {flag:?}; use --help for usage");
            }
            bail!("unexpected positional argument {flag:?}");
        }

        let value = match inline_value {
            Some(value) => value,
            None => args
                .next()
                .ok_or_else(|| anyhow!("{flag} requires a value"))?,
        };
        match flag {
            "--model" => {
                ensure!(
                    matches!(value.as_str(), "v2" | "v25" | "both"),
                    "--model must be v2, v25, or both"
                );
                config.model = value;
            }
            "--v2-model-dir" => config.v2_model_dir = nonempty_path(flag, value)?,
            "--v2-onnx-dir" => config.v2_onnx_dir = nonempty_path(flag, value)?,
            "--v25-bundle-dir" => config.v25_bundle_dir = nonempty_path(flag, value)?,
            "--output" => config.output = nonempty_path(flag, value)?,
            "--warmups" => config.warmups = parse_usize(flag, &value)?,
            "--repetitions" => config.repetitions = parse_usize(flag, &value)?,
            "--threshold" => {
                config.threshold = value.parse::<f32>().with_context(|| {
                    format!("invalid {flag} value {value:?}: expected a number")
                })?;
            }
            "--label" => {
                ensure!(!value.trim().is_empty(), "--label must not be empty");
                explicit_labels = true;
                labels.push(value);
            }
            _ => unreachable!("recognized option was not handled"),
        }
    }

    ensure!(config.warmups > 0, "--warmups must be at least 1");
    ensure!(config.repetitions > 0, "--repetitions must be at least 1");
    ensure!(
        config.threshold.is_finite() && (0.0..=1.0).contains(&config.threshold),
        "--threshold must be finite and between 0 and 1 inclusive"
    );
    if explicit_labels {
        config.labels = labels;
    }
    ensure!(
        !config.labels.is_empty(),
        "at least one --label is required"
    );
    Ok(CliAction::Run(config))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid {flag} value {value:?}: expected a non-negative integer"))
}

fn nonempty_path(flag: &str, value: String) -> Result<PathBuf> {
    ensure!(!value.is_empty(), "{flag} path must not be empty");
    Ok(PathBuf::from(value))
}

fn print_help() {
    println!(
        "\
CPU entity-extraction benchmark for GLiNER2 base vs GLiNER2.5 base

Usage:
  cargo run --release --example benchmark_gliner2 -- [OPTIONS]

Options:
  --model v2|v25|both    models to run [both; use shell wrapper for isolated runs]
  --v2-model-dir PATH    v2 tokenizer/config directory [models/gliner2-base-v1]
  --v2-onnx-dir PATH     v2 ONNX graph directory [onnx/gliner2-base-v1]
  --v25-bundle-dir PATH  complete GLiNER2.5 bundle [onnx/gliner2.5-base-v1]
  --output PATH          machine-readable JSON report [benchmark-gliner2.json]
  --warmups N            warmups per model and word count [3; minimum 1]
  --repetitions N        measured runs per model and word count [10; minimum 1]
  --threshold FLOAT      shared entity threshold [0.5; range 0..=1]
  --label LABEL          shared label; repeat to replace the four defaults
  -h, --help             show this help

Word counts are fixed at exactly 50, 500, and 3000 whitespace-delimited words."
    );
}

fn required_file(directory: &Path, name: &str) -> Result<PathBuf> {
    let path = directory.join(name);
    ensure!(
        path.is_file(),
        "required graph {name} is missing at {}",
        path.display()
    );
    Ok(path)
}

fn graph_identities(directory: &Path, names: &[&str]) -> Result<Vec<GraphIdentity>> {
    names
        .iter()
        .map(|&name| {
            let path = required_file(directory, name)?;
            let metadata = fs::metadata(&path)
                .with_context(|| format!("failed to stat graph {}", path.display()))?;
            ensure!(metadata.len() > 0, "graph {} is empty", path.display());
            Ok(GraphIdentity {
                name: name.to_owned(),
                path: canonical_or_original(&path),
                bytes: metadata.len(),
                sha256: sha256_file(&path)?,
            })
        })
        .collect()
}

fn verify_identity(identity: &GraphIdentity, expected: &Value) -> Result<()> {
    ensure!(
        expected["bytes"].as_u64() == Some(identity.bytes) && identity.bytes > 0,
        "size mismatch for {}",
        identity.path.display()
    );
    ensure!(
        expected["sha256"].as_str() == Some(identity.sha256.as_str()),
        "SHA-256 mismatch for {}",
        identity.path.display()
    );
    Ok(())
}

fn verify_v2_source(metadata_dir: &Path, graphs: &[GraphIdentity]) -> Result<Value> {
    let pins_bytes = include_bytes!("../docs/checkpoints/v2-metadata-pins.json");
    let pins: Value = serde_json::from_slice(pins_bytes)?;
    let base = &pins["models"]["gliner2-base-v1"];
    ensure!(
        base["source_repository"] == "fastino/gliner2-base-v1"
            && base["source_revision"] == "79c3a777abc572b4767922f3916cf63fb5754df2",
        "unexpected compiled v2 base source pin"
    );
    let metadata = graph_identities(
        metadata_dir,
        &["config.json", "tokenizer.json", "tokenizer_config.json"],
    )?;
    for file in &metadata {
        verify_identity(file, &base["source_metadata"][&file.name])?;
    }
    for graph in graphs {
        verify_identity(graph, &base["hosted_onnx_files"][&graph.name])?;
    }
    Ok(json!({
        "hf_model": base["source_repository"],
        "hf_revision": base["source_revision"],
        "hosted_onnx": pins["hosted_onnx"],
        "pin_document_sha256": sha256_bytes(pins_bytes),
        "metadata_files": metadata,
        "verification": "metadata and loaded graphs match compiled repository base pins"
    }))
}

fn check_v25_identity(manifest: &Value) -> Result<()> {
    ensure!(
        manifest["hf_model"] == "fastino/gliner2.5-base-v1"
            && manifest["hf_revision"] == "78cea040597df251eedefa9d7ee2a756af39fe64",
        "manifest must identify the pinned GLiNER2.5 base checkpoint"
    );
    ensure!(
        manifest["architecture"] == "boundary"
            && manifest["architecture_version"] == 1
            && manifest["manifest_version"] == 1,
        "unsupported boundary manifest architecture/version"
    );
    ensure!(
        manifest["gliner2_commit"] == "d7c727458bf6929bc9ef5ee04e13c3f717a7c455",
        "wrong upstream GLiNER2 source commit"
    );
    match manifest["status"].as_str() {
        Some("exported-unvalidated") => ensure!(
            manifest["release_ready"] == false,
            "unvalidated bundle must not claim release_ready"
        ),
        Some("validated") => ensure!(
            manifest["release_ready"] == true,
            "validated bundle must claim release_ready (full validation follows timings)"
        ),
        _ => bail!("unsupported manifest status"),
    }
    Ok(())
}

fn safe_manifest_file(root: &Path, relative: &str) -> Result<PathBuf> {
    ensure!(
        !relative.is_empty()
            && !relative.contains(['\\', ':'])
            && relative
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "unsafe manifest path {relative:?}"
    );
    let path = root.join(relative).canonicalize()?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "manifest file escapes bundle: {relative}"
    );
    Ok(path)
}

fn verify_v25_source(directory: &Path, snapshot: &[u8], graphs: &[GraphIdentity]) -> Result<Value> {
    let root = directory.canonicalize()?;
    ensure!(
        fs::read(root.join("export_manifest.json"))? == snapshot,
        "manifest changed during benchmark; rerun against an immutable bundle"
    );
    let manifest: Value = serde_json::from_slice(snapshot)?;
    check_v25_identity(&manifest)?;
    let files = manifest["files"]
        .as_object()
        .context("manifest files must be an object")?;
    ensure!(
        !files.contains_key("export_manifest.json"),
        "manifest cannot authenticate itself"
    );
    for required in gliner2_rs::bundle::BOUNDARY_REQUIRED_FILES {
        ensure!(
            files.contains_key(*required),
            "manifest missing required file {required}"
        );
    }
    // Development validation intentionally accepts exported-unvalidated status,
    // but never promotes it or equates checksum verification with model parity.
    for (name, expected) in files {
        let path = safe_manifest_file(&root, name)?;
        if let Some(graph) = graphs.iter().find(|graph| graph.name == *name) {
            // Bind the hashes actually emitted in this report to the manifest.
            ensure!(graph.path == path, "loaded graph path changed: {name}");
            verify_identity(graph, expected)?;
        } else {
            let identity = GraphIdentity {
                name: name.clone(),
                bytes: fs::metadata(&path)?.len(),
                sha256: sha256_file(&path)?,
                path,
            };
            verify_identity(&identity, expected)?;
        }
    }
    let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
    ensure!(
        config["architecture"] == "boundary"
            && config["architecture_version"] == 1
            && config["model_name"] == "microsoft/deberta-v3-base",
        "authenticated config is not the expected boundary base model"
    );
    let validated = manifest["status"] == "validated";
    if validated {
        gliner2_rs::validate_bundle(&root).context("release-ready bundle validation failed")?;
    }
    ensure!(
        fs::read(root.join("export_manifest.json"))? == snapshot,
        "manifest changed during provenance verification"
    );
    Ok(json!({
        "hf_model": manifest["hf_model"], "hf_revision": manifest["hf_revision"],
        "gliner2_commit": manifest["gliner2_commit"],
        "manifest_sha256": sha256_bytes(snapshot),
        "status": manifest["status"], "release_ready": manifest["release_ready"],
        "source_file_sha256": manifest["source_file_sha256"],
        "authenticated_files": files,
        "verification": if validated { "validate_bundle passed after timings" }
            else { "base identity and all manifest file sizes/hashes checked; exported-unvalidated, NOT release-ready; no parity claim" }
    }))
}

fn add_token_counts(model: &mut ModelReport, texts: &[String], labels: &[String]) -> Result<()> {
    let tokenizer = RuntimeTokenizer::from_dir(&model.metadata_dir)?;
    let tokenizer_sha256 = sha256_file(&model.metadata_dir.join("tokenizer.json"))?;
    let config = ModelConfig::from_dir(&model.metadata_dir)?;
    let schema = vec![build_entities_schema_tokens(labels, None)];
    for (case, text) in model.cases.iter_mut().zip(texts) {
        let tokens = if model.architecture == "boundary" {
            BoundaryPreprocessingPolicy::from_optional_max_len(
                config.max_len,
                WordSplitter::Whitespace,
            )?
            .prepare(text, &[])
            .text_tokens
        } else {
            PreprocessingPolicy::new(config.max_len)
                .tokenize(text, true)
                .into_iter()
                .map(|span| span.token)
                .collect()
        };
        let formatted = format_input_with_mapping(&tokenizer, &schema, &tokens)?;
        case.tokenization = Some(TokenizationReport {
            tokenizer_sha256: tokenizer_sha256.clone(),
            retained_text_words: tokens.len(),
            text_only_subwords: formatted
                .mapped_indices
                .iter()
                .filter(|mapping| mapping.segment == Segment::Text)
                .count(),
            schema_plus_text_subwords: formatted.input_ids.len(),
            max_original_words: config.max_len,
            scope: "pipeline preprocessing + per-word RuntimeTokenizer.tokenize_token(add_special_tokens=false); text_only excludes schema/separators; schema_plus_text includes all formatted encoder IDs; computed outside timings, not raw whole-string tokenization",
        });
    }
    Ok(())
}

fn canonical_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn duration_ns(duration: Duration) -> Result<u64> {
    u64::try_from(duration.as_nanos()).context("duration exceeds the report's u64 nanosecond range")
}

fn collect_provenance() -> Provenance {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut environment = BTreeMap::new();
    for name in [
        "OMP_NUM_THREADS",
        "ORT_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "RUSTFLAGS",
    ] {
        if let Ok(value) = env::var(name) {
            environment.insert(name.to_owned(), value);
        }
    }

    Provenance {
        crate_version: env!("CARGO_PKG_VERSION"),
        git_commit: command_output_in(root, "git", &["rev-parse", "HEAD"]),
        git_worktree_dirty: git_worktree_dirty(root),
        rustc_verbose_version: command_output_in(root, "rustc", &["--version", "--verbose"]),
        cargo_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        cpu_model: cpu_model(),
        logical_cpus: std::thread::available_parallelism().ok().map(usize::from),
        physical_cores: physical_cores(),
        power_mode: env::var("BENCHMARK_POWER_MODE")
            .ok()
            .filter(|value| !value.trim().is_empty()),
        power_mode_source: "operator-declared BENCHMARK_POWER_MODE; null means not recorded, not inferred",
        memory_bytes: memory_bytes(),
        ort_crate_version: locked_package_version(root.join("Cargo.lock"), "ort"),
        native_onnx_runtime: ort::info().to_owned(),
        environment,
    }
}

fn command_output_in(directory: &Path, program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn git_worktree_dirty(directory: &Path) -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(directory)
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}

fn cpu_model() -> Option<String> {
    if env::consts::OS == "macos" {
        return command_output_in(
            Path::new("/"),
            "sysctl",
            &["-n", "machdep.cpu.brand_string"],
        )
        .or_else(|| command_output_in(Path::new("/"), "sysctl", &["-n", "hw.model"]));
    }
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        matches!(key.trim(), "model name" | "Hardware")
            .then(|| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn physical_cores() -> Option<usize> {
    if env::consts::OS == "macos" {
        return command_output_in(Path::new("/"), "sysctl", &["-n", "hw.physicalcpu"])?
            .parse()
            .ok();
    }
    let output = command_output_in(Path::new("/"), "lscpu", &["-p=SOCKET,CORE"])?;
    let cores: std::collections::BTreeSet<_> = output
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .collect();
    (!cores.is_empty()).then_some(cores.len())
}

fn memory_bytes() -> Option<u64> {
    if env::consts::OS == "macos" {
        return command_output_in(Path::new("/"), "sysctl", &["-n", "hw.memsize"])?
            .parse()
            .ok();
    }
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let kilobytes = meminfo.lines().find_map(|line| {
        let value = line.strip_prefix("MemTotal:")?.trim();
        value.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    kilobytes.checked_mul(1024)
}

fn locked_package_version(path: PathBuf, package: &str) -> Option<String> {
    let lock = fs::read_to_string(path).ok()?;
    let mut current_name: Option<&str> = None;
    let mut current_version: Option<&str> = None;
    for line in lock.lines().chain(["[[package]]"]) {
        if line == "[[package]]" {
            if current_name == Some(package) {
                return current_version.map(ToOwned::to_owned);
            }
            current_name = None;
            current_version = None;
        } else if let Some(value) = line.strip_prefix("name = \"") {
            current_name = value.strip_suffix('"');
        } else if let Some(value) = line.strip_prefix("version = \"") {
            current_version = value.strip_suffix('"');
        }
    }
    None
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open graph for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("failed while hashing graph {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn hash_u64(hasher: &mut Sha256, value: usize) -> Result<()> {
    let value = u64::try_from(value).context("output index does not fit u64")?;
    hasher.update(value.to_be_bytes());
    Ok(())
}

fn hash_sized_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    hash_u64(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for &byte in bytes.as_ref() {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_warmups_and_model_selection() {
        let CliAction::Run(defaults) = parse_args(Vec::<String>::new()).unwrap() else {
            panic!("run expected")
        };
        assert_eq!(defaults.warmups, 3);
        assert_eq!(defaults.model, "both");
        for model in ["v2", "v25", "both"] {
            let CliAction::Run(config) = parse_args(["--model", model]).unwrap() else {
                panic!("run expected")
            };
            assert_eq!(config.model, model);
        }
        assert!(parse_args(["--model", "small"]).is_err());
        assert!(parse_args(["--model"]).is_err());
    }

    #[test]
    fn percentile_rank_is_reproducible() {
        let values: Vec<u64> = (1..=20).rev().collect();
        assert_eq!(nearest_rank(&values, 90).unwrap(), 18);
        assert_eq!(nearest_rank(&values, 95).unwrap(), 19);
        assert_eq!(nearest_rank(&[7], 90).unwrap(), 7);
        assert!(nearest_rank(&[], 90).is_err());
        assert!(nearest_rank(&values, 0).is_err());
        assert!(nearest_rank(&values, 101).is_err());
    }

    #[test]
    fn manifest_identity_distinguishes_development_from_release() {
        let mut manifest = json!({
            "hf_model": "fastino/gliner2.5-base-v1",
            "hf_revision": "78cea040597df251eedefa9d7ee2a756af39fe64",
            "gliner2_commit": "d7c727458bf6929bc9ef5ee04e13c3f717a7c455",
            "manifest_version": 1, "architecture": "boundary", "architecture_version": 1,
            "status": "exported-unvalidated", "release_ready": false
        });
        check_v25_identity(&manifest).unwrap();
        manifest["release_ready"] = json!(true);
        assert!(check_v25_identity(&manifest).is_err());
        manifest["status"] = json!("validated");
        check_v25_identity(&manifest).unwrap();
        manifest["hf_model"] = json!("fastino/gliner2.5-small-v1");
        assert!(check_v25_identity(&manifest).is_err());
        manifest["hf_model"] = json!("fastino/gliner2.5-base-v1");
        manifest["hf_revision"] = json!("main");
        assert!(check_v25_identity(&manifest).is_err());
    }

    #[test]
    fn identity_and_paths_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        fs::write(root.join("sample"), b"abc").unwrap();
        let identity = graph_identities(&root, &["sample"]).unwrap().remove(0);
        verify_identity(
            &identity,
            &json!({"bytes": 3, "sha256": sha256_bytes(b"abc")}),
        )
        .unwrap();
        assert!(verify_identity(&identity, &json!({"bytes": 3, "sha256": "wrong"})).is_err());
        assert!(
            verify_identity(
                &identity,
                &json!({"bytes": 4, "sha256": sha256_bytes(b"abc")})
            )
            .is_err()
        );
        assert_eq!(
            safe_manifest_file(&root, "sample").unwrap(),
            root.join("sample")
        );
        for name in ["../sample", "/sample", "./sample", "", "a//b", "C:/sample"] {
            assert!(safe_manifest_file(&root, name).is_err());
        }
        #[cfg(unix)]
        {
            let outside = tempfile::NamedTempFile::new().unwrap();
            std::os::unix::fs::symlink(outside.path(), root.join("escape")).unwrap();
            assert!(safe_manifest_file(&root, "escape").is_err());
        }
    }

    #[test]
    fn report_publication_is_complete_and_no_clobber() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/report.json");
        write_json_atomic(&path, &json!({"complete": true})).unwrap();
        assert!(write_json_atomic(&path, &json!({"complete": false})).is_err());
        let actual: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(actual, json!({"complete": true}));
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn generated_text_has_exact_requested_sizes_and_is_stable() {
        for word_count in WORD_COUNTS {
            let first = generate_text(word_count);
            let second = generate_text(word_count);
            assert_eq!(first.split_whitespace().count(), word_count);
            assert_eq!(first, second);
            assert_eq!(
                sha256_bytes(first.as_bytes()),
                sha256_bytes(second.as_bytes())
            );
        }
    }

    #[test]
    fn timing_statistics_use_average_median_and_nearest_rank_p95() {
        assert_eq!(timing_statistics(&[9, 1, 3]).unwrap(), (3, 9));
        assert_eq!(timing_statistics(&[40, 10, 30, 20]).unwrap(), (25, 40));
        let one_to_twenty: Vec<u64> = (1..=20).collect();
        assert_eq!(timing_statistics(&one_to_twenty).unwrap(), (10, 19));
        assert!(timing_statistics(&[]).is_err());
    }

    #[test]
    fn parser_accepts_split_layout_and_repeated_labels() {
        let action = parse_args([
            "--v2-model-dir",
            "/metadata",
            "--v2-onnx-dir=/graphs",
            "--v25-bundle-dir",
            "/bundle",
            "--output",
            "/report.json",
            "--warmups",
            "3",
            "--repetitions=7",
            "--threshold",
            "0.25",
            "--label",
            "person",
            "--label=organization",
        ])
        .unwrap();
        let CliAction::Run(config) = action else {
            panic!("expected run action");
        };
        assert_eq!(config.v2_model_dir, Path::new("/metadata"));
        assert_eq!(config.v2_onnx_dir, Path::new("/graphs"));
        assert_eq!(config.v25_bundle_dir, Path::new("/bundle"));
        assert_eq!(config.warmups, 3);
        assert_eq!(config.repetitions, 7);
        assert_eq!(config.threshold, 0.25);
        assert_eq!(config.labels, ["person", "organization"]);
    }

    #[test]
    fn parser_rejects_invalid_parameters_clearly() {
        let cases = [
            (vec!["--warmups", "0"], "--warmups must be at least 1"),
            (
                vec!["--repetitions", "0"],
                "--repetitions must be at least 1",
            ),
            (
                vec!["--threshold", "NaN"],
                "--threshold must be finite and between 0 and 1 inclusive",
            ),
            (
                vec!["--threshold", "1.1"],
                "--threshold must be finite and between 0 and 1 inclusive",
            ),
            (vec!["--label", ""], "--label must not be empty"),
            (vec!["--warmups"], "--warmups requires a value"),
            (vec!["--unknown", "x"], "unrecognized argument"),
            (vec!["positional"], "unexpected positional argument"),
        ];
        for (arguments, expected) in cases {
            let error = parse_args(arguments).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected {error:?} to contain {expected:?}"
            );
        }
    }

    #[test]
    fn sha256_matches_standard_vectors_and_streaming() {
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut streamed = Sha256::new();
        streamed.update(b"a");
        streamed.update(b"b");
        streamed.update(b"c");
        assert_eq!(
            hex_digest(streamed.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let million_as = vec![b'a'; 1_000_000];
        let mut chunked = Sha256::new();
        for chunk in million_as.chunks(17) {
            chunked.update(chunk);
        }
        assert_eq!(
            hex_digest(chunked.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn lockfile_package_version_parser_finds_ort() {
        let version = locked_package_version(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"),
            "ort",
        );
        assert_eq!(version.as_deref(), Some("2.0.0-rc.13"));
    }
}
