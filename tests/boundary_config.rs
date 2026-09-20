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
fn record_temperature_defaults_without_changing_existing_temperatures() -> Result<()> {
    let runtime = load(json!({
        "pair_temperature": 0.75,
        "classification_temperature": 1.25,
    }))?;

    assert_eq!(runtime.pair_temperature, 0.75);
    assert_eq!(runtime.classification_temperature, 1.25);
    assert_eq!(runtime.record_temperature, 1.0);
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
    let cases = [
        ("zero", json!(0.0)),
        ("negative", json!(-0.5)),
        ("wrong type", json!("1.0")),
        ("f32 underflow", json!(1e-300)),
        ("f32 overflow", json!(1e39)),
    ];

    for (case, value) in cases {
        let error = load(json!({"record_temperature": value}))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("record_temperature")
                && (error.contains("positive") || error.contains("number")),
            "{case}: {error}"
        );
    }
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
    }
    Ok(())
}
