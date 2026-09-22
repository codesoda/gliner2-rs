use std::{fs, path::Path};

use gliner2_rs::bundle::{BOUNDARY_GRAPH_FILES, BOUNDARY_REQUIRED_FILES, validate_bundle};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn fixture() -> TempDir {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    for relative in BOUNDARY_REQUIRED_FILES {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = if *relative == "config.json" {
            br#"{"architecture":"boundary","architecture_version":1,"model_name":"microsoft/deberta-v3-base"}"#.to_vec()
        } else {
            format!("test content for {relative}\n").into_bytes()
        };
        fs::write(path, bytes).unwrap();
    }
    write_manifest(root, base_manifest(root));
    temp
}

fn base_manifest(root: &Path) -> Value {
    let files = BOUNDARY_REQUIRED_FILES
        .iter()
        .map(|relative| {
            let bytes = fs::read(root.join(relative)).unwrap();
            (
                (*relative).to_string(),
                json!({"bytes": bytes.len(), "sha256": sha256(&bytes)}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let graphs = BOUNDARY_GRAPH_FILES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                json!({
                    "inputs": {
                        "input": {"dtype": "FLOAT", "shape": ["dynamic", 1]}
                    },
                    "outputs": {
                        "output": {"dtype": "FLOAT", "shape": ["dynamic", 1]}
                    }
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    json!({
        "manifest_version": 1,
        "architecture": "boundary",
        "architecture_version": 1,
        "status": "validated",
        "release_ready": true,
        "hf_model": "fastino/gliner2.5-base-v1",
        "hf_revision": "78cea040597df251eedefa9d7ee2a756af39fe64",
        "gliner2_commit": "d7c727458bf6929bc9ef5ee04e13c3f717a7c455",
        "opset": 17,
        "precision": "fp32",
        "ort_crate_version": "2.0.0-rc.13",
        "native_onnx_runtime": "1.28.0",
        "validation_onnxruntime": "1.20.1",
        "dependencies": {
            "gliner2": "2.0.0",
            "torch": "2.8.0",
            "transformers": "4.57.6",
            "onnx": "1.17.0",
            "onnxruntime": "1.20.1",
            "numpy": "2.2.6",
            "peft": "0.17.1",
            "sentencepiece": "0.2.1"
        },
        "source_file_sha256": {
            "gliner2/model.py": "0000000000000000000000000000000000000000000000000000000000000000"
        },
        "files": files,
        "graphs": graphs,
        "validation": {
            "report": "validation.json",
            "sha256": "1111111111111111111111111111111111111111111111111111111111111111"
        }
    })
}

fn read_manifest(root: &Path) -> Value {
    serde_json::from_slice(&fs::read(root.join("export_manifest.json")).unwrap()).unwrap()
}

fn write_manifest(root: &Path, manifest: Value) {
    fs::write(
        root.join("export_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

fn replace_authenticated_file(root: &Path, relative: &str, bytes: &[u8]) {
    fs::write(root.join(relative), bytes).unwrap();
    let mut manifest = read_manifest(root);
    manifest["files"][relative] = json!({
        "bytes": bytes.len(),
        "sha256": sha256(bytes),
    });
    write_manifest(root, manifest);
}

fn error(root: &Path) -> String {
    format!("{:#}", validate_bundle(root).unwrap_err())
}

#[test]
fn validates_exporter_shaped_release_ready_bundle() {
    let temp = fixture();
    let validated = validate_bundle(temp.path()).unwrap();
    assert_eq!(validated.root, temp.path().canonicalize().unwrap());
    assert_eq!(validated.manifest.hf_model, "fastino/gliner2.5-base-v1");
    assert_eq!(
        validated.manifest.files.len(),
        BOUNDARY_REQUIRED_FILES.len()
    );
    let encoder = &validated.manifest.graphs["encoder.onnx"];
    assert_eq!(encoder.inputs["input"].dtype, "FLOAT");
    assert_eq!(
        encoder.inputs["input"].shape,
        vec![json!("dynamic"), json!(1)]
    );
}

#[test]
fn rejects_legacy_or_malformed_graph_signatures() {
    let cases = [
        (
            json!([{"name": "input", "dtype": "FLOAT", "shape": [1]}]),
            "invalid type",
        ),
        (
            json!({"": {"dtype": "FLOAT", "shape": [1]}}),
            "invalid inputs tensor name",
        ),
        (
            json!({"input": {"dtype": "float32", "shape": [1]}}),
            "invalid ONNX dtype",
        ),
        (
            json!({"input": {"dtype": "FLOAT", "shape": [-1]}}),
            "invalid shape dimension",
        ),
        (
            json!({"input": {"dtype": "FLOAT", "shape": [true]}}),
            "invalid shape dimension",
        ),
        (
            json!({"input": {"dtype": "FLOAT", "shape": [" "]}}),
            "invalid shape dimension",
        ),
    ];

    for (inputs, message) in cases {
        let temp = fixture();
        let mut manifest = read_manifest(temp.path());
        manifest["graphs"]["encoder.onnx"]["inputs"] = inputs;
        write_manifest(temp.path(), manifest);
        assert!(error(temp.path()).contains(message), "expected {message}");
    }
}

#[test]
fn rejects_duplicate_json_object_keys() {
    let temp = fixture();
    let path = temp.path().join("export_manifest.json");
    let text = fs::read_to_string(&path).unwrap();
    let duplicate = text.replacen(
        "\"manifest_version\": 1,",
        "\"manifest_version\": 1,\n  \"manifest_version\": 1,",
        1,
    );
    assert_ne!(duplicate, text);
    fs::write(path, duplicate).unwrap();
    assert!(error(temp.path()).contains("duplicate JSON object key `manifest_version`"));
}

#[test]
fn rejects_missing_manifest_and_required_files() {
    let empty = tempfile::tempdir().unwrap();
    assert!(error(empty.path()).contains("export_manifest.json"));

    let temp = fixture();
    fs::remove_file(temp.path().join("encoder.onnx")).unwrap();
    assert!(error(temp.path()).contains("encoder.onnx"));

    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["files"].as_object_mut().unwrap().remove("NOTICE");
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("missing required file entry `NOTICE`"));

    let temp = fixture();
    fs::write(temp.path().join("unlisted.bin"), b"not authenticated").unwrap();
    assert!(error(temp.path()).contains("unlisted file `unlisted.bin`"));
}

#[test]
fn rejects_bad_sizes_and_hashes() {
    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["files"]["LICENSE"]["bytes"] = json!(9999);
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("size mismatch for `LICENSE`"));

    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["files"]["LICENSE"]["sha256"] =
        json!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("SHA-256 mismatch for `LICENSE`"));

    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["files"]["LICENSE"]["sha256"] = json!("ABC");
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("64 lowercase hexadecimal"));
}

#[test]
fn rejects_partial_or_unpromoted_status() {
    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["status"] = json!("exported-unvalidated");
    manifest["release_ready"] = json!(false);
    manifest["validation"] = Value::Null;
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("status is not `validated`"));

    let temp = fixture();
    let mut manifest = read_manifest(temp.path());
    manifest["release_ready"] = json!(false);
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("not release-ready"));
}

#[test]
fn rejects_wrong_manifest_architecture_version_and_pins() {
    let cases = [
        ("architecture", json!("span"), "bundle architecture"),
        ("architecture_version", json!(2), "architecture_version"),
        ("manifest_version", json!(2), "manifest_version"),
        (
            "hf_model",
            json!("fastino/not-a-supported-model"),
            "unsupported boundary model",
        ),
        (
            "hf_revision",
            json!("0000000000000000000000000000000000000000"),
            "wrong immutable revision",
        ),
        (
            "gliner2_commit",
            json!("0000000000000000000000000000000000000000"),
            "source commit",
        ),
        ("dependencies", json!({"torch": "0.0.0"}), "dependency pins"),
    ];

    for (field, replacement, message) in cases {
        let temp = fixture();
        let mut manifest = read_manifest(temp.path());
        manifest[field] = replacement;
        write_manifest(temp.path(), manifest);
        assert!(error(temp.path()).contains(message), "field {field}");
    }
}

#[test]
fn rejects_wrong_authenticated_config_identity() {
    let cases = [
        (
            br#"{"architecture":"span","architecture_version":1,"model_name":"microsoft/deberta-v3-base"}"#.as_slice(),
            "wrong model architecture",
        ),
        (
            br#"{"architecture":"boundary","architecture_version":2,"model_name":"microsoft/deberta-v3-base"}"#.as_slice(),
            "wrong architecture_version",
        ),
        (
            br#"{"architecture":"boundary","architecture_version":1,"model_name":"microsoft/deberta-v3-xsmall"}"#.as_slice(),
            "wrong encoder model",
        ),
    ];

    for (config, message) in cases {
        let temp = fixture();
        replace_authenticated_file(temp.path(), "config.json", config);
        assert!(error(temp.path()).contains(message));
    }
}

#[test]
fn rejects_unsafe_manifest_paths() {
    for unsafe_path in [
        "../outside",
        "/tmp/outside",
        "nested/../../outside",
        "nested\\outside",
        "C:/outside",
        "nested//file",
        "./file",
    ] {
        let temp = fixture();
        let mut manifest = read_manifest(temp.path());
        manifest["files"][unsafe_path] = json!({
            "bytes": 1,
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
        });
        write_manifest(temp.path(), manifest);
        let message = error(temp.path());
        assert!(
            message.contains("manifest path") || message.contains("Windows"),
            "path {unsafe_path:?} produced: {message}"
        );
    }
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_that_escape_bundle_root() {
    use std::os::unix::fs::symlink;

    let temp = fixture();
    let outside = tempfile::NamedTempFile::new().unwrap();
    fs::write(outside.path(), b"outside").unwrap();
    symlink(outside.path(), temp.path().join("escape.bin")).unwrap();

    let mut manifest = read_manifest(temp.path());
    manifest["files"]["escape.bin"] = json!({
        "bytes": 7,
        "sha256": sha256(b"outside")
    });
    write_manifest(temp.path(), manifest);
    assert!(error(temp.path()).contains("escapes bundle directory"));
}
