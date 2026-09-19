use std::fs;

use gliner2_rs::{
    Extractor,
    pipeline::{AutoPipeline, BoundaryPipeline, Gliner2Pipeline, SpanPipeline},
};
use tempfile::tempdir;

#[test]
fn public_aliases_compile_to_expected_high_level_types() {
    fn root_alias(_: Option<Extractor>) {}
    fn auto(_: Option<AutoPipeline>) {}
    fn legacy(_: Option<Gliner2Pipeline>) {}
    fn span(_: Option<SpanPipeline>) {}

    root_alias(None);
    auto(None);
    legacy(None);
    span(None);
}

#[test]
fn span_bundle_reports_missing_files_before_loading_sessions() {
    let dir = tempdir().unwrap();
    let error = AutoPipeline::from_dir(dir.path())
        .err()
        .expect("incomplete span bundle must fail")
        .to_string();
    assert!(error.contains("tokenizer.json"));
}

#[test]
fn boundary_bundle_reports_missing_files_without_span_fallback() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architecture":"boundary","architecture_version":1}"#,
    )
    .unwrap();

    let direct = BoundaryPipeline::from_dir(dir.path())
        .err()
        .expect("incomplete boundary bundle must fail")
        .to_string();
    assert!(direct.contains("boundary bundle missing required `tokenizer.json`"));

    let auto = AutoPipeline::from_dir(dir.path())
        .err()
        .expect("boundary dispatch must load only boundary artifacts")
        .to_string();
    assert!(auto.contains("boundary bundle missing required `tokenizer.json`"));
    assert!(!auto.contains("extractor_padded.onnx"));
}
