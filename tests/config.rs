use std::fs;

use gliner2_rs::config::{Architecture, ModelConfig};
use tempfile::tempdir;

#[test]
fn missing_and_legacy_configs_default_to_uncapped_span() {
    let missing = tempdir().unwrap();
    assert_eq!(
        ModelConfig::from_dir(missing.path()).unwrap(),
        ModelConfig::default()
    );

    let legacy = tempdir().unwrap();
    fs::write(
        legacy.path().join("config.json"),
        r#"{"model_type":"extractor","max_width":8}"#,
    )
    .unwrap();
    assert_eq!(
        ModelConfig::from_dir(legacy.path()).unwrap(),
        ModelConfig::default()
    );
}

#[test]
fn boundary_defaults_version_and_word_cap() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architecture":"boundary","boundary_head":{"candidate_pool":"shared"}}"#,
    )
    .unwrap();

    let config = ModelConfig::from_dir(dir.path()).unwrap();
    assert_eq!(config.architecture, Architecture::Boundary);
    assert_eq!(config.architecture_version, Some(1));
    assert_eq!(config.max_len, Some(4096));
}

#[test]
fn explicit_span_cap_is_preserved() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"architecture":"span","architecture_version":1,"max_len":12}"#,
    )
    .unwrap();

    let config = ModelConfig::from_dir(dir.path()).unwrap();
    assert_eq!(config.architecture, Architecture::Span);
    assert_eq!(config.max_len, Some(12));
}

#[test]
fn malformed_and_unknown_explicit_values_are_errors() {
    let cases = [
        ("{", "malformed model config"),
        (r#"{"architecture":"future"}"#, "unsupported architecture"),
        (
            r#"{"architecture":"boundary","architecture_version":2}"#,
            "unsupported architecture_version",
        ),
        (r#"{"architecture":"span","max_len":0}"#, "positive integer"),
        (
            r#"{"architecture":"boundary","boundary_head":[]}"#,
            "malformed `boundary_head`",
        ),
        (
            r#"{"architecture":"boundary","boundary_head":{"candidate_pool":"per_query"}}"#,
            "unsupported boundary candidate_pool",
        ),
    ];

    for (contents, expected) in cases {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), contents).unwrap();
        let error = ModelConfig::from_dir(dir.path()).unwrap_err().to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}
