use std::fs;

use anyhow::Result;
use gliner2_rs::boundary::config::BoundaryRuntimeConfig;
use serde_json::{Value, json};
use tempfile::tempdir;

fn load(head: Value) -> Result<BoundaryRuntimeConfig> {
    let dir = tempdir()?;
    let config = json!({
        "architecture": "boundary",
        "architecture_version": 1,
        "boundary_head": head,
    });
    fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
    BoundaryRuntimeConfig::from_dir(dir.path())
}

#[test]
fn optional_runtime_settings_default_without_changing_existing_temperatures() -> Result<()> {
    let runtime = load(json!({
        "pair_temperature": 0.75,
        "classification_temperature": 1.25,
    }))?;

    assert_eq!(runtime.pair_temperature, 0.75);
    assert_eq!(runtime.classification_temperature, 1.25);
    assert_eq!(runtime.record_temperature, 1.0);
    assert_eq!(runtime.relation_temperature, 1.0);
    assert_eq!(runtime.relation_proposals.heads_per_relation, 32);
    assert_eq!(runtime.relation_proposals.tails_per_relation, 32);
    assert_eq!(runtime.relation_proposals.pair_cap, 64);
    assert_eq!(runtime.relation_proposals.argument_threshold, 0.2);
    Ok(())
}

#[test]
fn record_temperature_accepts_positive_f32_values() -> Result<()> {
    let runtime = load(json!({"record_temperature": 0.375}))?;
    assert_eq!(runtime.record_temperature, 0.375);
    Ok(())
}

#[test]
fn record_temperature_rejects_invalid_numbers_and_types() {
    for (case, value) in [
        ("zero", json!(0.0)),
        ("negative", json!(-0.5)),
        ("wrong type", json!("1.0")),
        ("f32 underflow", json!(1e-300)),
        ("f32 overflow", json!(1e39)),
    ] {
        let error = load(json!({"record_temperature": value}))
            .unwrap_err()
            .to_string();
        assert!(error.contains("record_temperature"), "{case}: {error}");
    }
}

#[test]
fn relation_settings_accept_valid_values() -> Result<()> {
    let runtime = load(json!({
        "relation_temperature": 0.375,
        "relation_heads_per_type": 7,
        "relation_tails_per_type": 9,
        "relation_pair_cap": 11,
        "relation_argument_proposal_threshold": 0.75
    }))?;
    assert_eq!(runtime.relation_temperature, 0.375);
    assert_eq!(runtime.relation_proposals.heads_per_relation, 7);
    assert_eq!(runtime.relation_proposals.tails_per_relation, 9);
    assert_eq!(runtime.relation_proposals.pair_cap, 11);
    assert_eq!(runtime.relation_proposals.argument_threshold, 0.75);
    Ok(())
}

#[test]
fn relation_settings_reject_invalid_numbers_and_types() {
    for (name, value) in [
        ("relation_temperature", json!(0.0)),
        ("relation_temperature", json!(-0.5)),
        ("relation_temperature", json!("1.0")),
        ("relation_temperature", json!(1e-300)),
        ("relation_temperature", json!(1e39)),
        ("relation_heads_per_type", json!(0)),
        ("relation_tails_per_type", json!(0)),
        ("relation_pair_cap", json!(0)),
        ("relation_argument_proposal_threshold", json!(-0.1)),
        ("relation_argument_proposal_threshold", json!(1.1)),
        ("relation_argument_proposal_threshold", json!("0.2")),
    ] {
        let error = load(json!({name: value})).unwrap_err().to_string();
        assert!(error.contains(name), "{name}: {error}");
    }
}

#[test]
fn relation_cartesian_capacity_overflow_is_rejected_before_loading_models() {
    let error = load(json!({
        "relation_heads_per_type": usize::MAX,
        "relation_tails_per_type": 2,
    }))
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("relation_heads_per_type") && error.contains("overflows"),
        "{error}"
    );
}

#[test]
fn incompatible_record_graph_dimensions_are_rejected() {
    for (name, actual, expected) in [
        ("record_dim", json!(64), "128"),
        ("record_instance_queries", json!(16), "32"),
    ] {
        let error = load(json!({name: actual})).unwrap_err().to_string();
        assert!(error.contains(name), "{name}: {error}");
        assert!(error.contains(expected), "{name}: {error}");
        assert!(
            error.contains("unsupported boundary graph setting"),
            "{name}: {error}"
        );
    }
}

#[test]
fn pinned_checkpoint_record_settings_are_supported() -> Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for checkpoint in ["small", "base", "multi"] {
        let dir = tempdir()?;
        fs::copy(
            root.join(format!("docs/checkpoints/{checkpoint}-config.json")),
            dir.path().join("config.json"),
        )?;
        let runtime = BoundaryRuntimeConfig::from_dir(dir.path())?;
        assert_eq!(runtime.record_temperature, 1.0, "{checkpoint}");
        assert_eq!(runtime.relation_temperature, 1.0, "{checkpoint}");
        assert_eq!(
            runtime.relation_proposals.heads_per_relation, 32,
            "{checkpoint}"
        );
        assert_eq!(
            runtime.relation_proposals.tails_per_relation, 32,
            "{checkpoint}"
        );
        assert_eq!(runtime.relation_proposals.pair_cap, 64, "{checkpoint}");
        assert_eq!(
            runtime.relation_proposals.argument_threshold, 0.2,
            "{checkpoint}"
        );
    }
    Ok(())
}
