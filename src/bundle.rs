//! Validation for published GLiNER2.5 boundary bundles.
//!
//! Bundle manifests are an untrusted download boundary. Validation is deliberately
//! separate from model loading and performs no network access.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, anyhow, ensure};
use serde::{
    Deserialize, Deserializer,
    de::{Error as _, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use crate::Result;

const MANIFEST_FILE: &str = "export_manifest.json";
const MAX_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
const GLINER2_COMMIT: &str = "d7c727458bf6929bc9ef5ee04e13c3f717a7c455";
const ONNX_TENSOR_DTYPES: &[&str] = &[
    "FLOAT",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "INT32",
    "INT64",
    "STRING",
    "BOOL",
    "FLOAT16",
    "DOUBLE",
    "UINT32",
    "UINT64",
    "COMPLEX64",
    "COMPLEX128",
    "BFLOAT16",
    "FLOAT8E4M3FN",
    "FLOAT8E4M3FNUZ",
    "FLOAT8E5M2",
    "FLOAT8E5M2FNUZ",
    "UINT4",
    "INT4",
    "FLOAT4E2M1",
];
const EXPECTED_EXPORT_DEPENDENCIES: &[(&str, &str)] = &[
    ("gliner2", "2.0.0"),
    ("torch", "2.8.0"),
    ("transformers", "4.57.6"),
    ("onnx", "1.17.0"),
    ("onnxruntime", "1.20.1"),
    ("numpy", "2.2.6"),
    ("peft", "0.17.1"),
    ("sentencepiece", "0.2.1"),
];

/// Files every published boundary bundle must contain and authenticate.
pub const BOUNDARY_REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "encoder_config/config.json",
    "SOURCE_MODEL_CARD.md",
    "LICENSE",
    "NOTICE",
    "encoder.onnx",
    "classifier.onnx",
    "boundary_marginals.onnx",
    "boundary_scorer.onnx",
    "boundary_explicit_scorer.onnx",
    "boundary_records.onnx",
    "boundary_relations.onnx",
];

/// The exact graph set supported by boundary architecture version 1.
pub const BOUNDARY_GRAPH_FILES: &[&str] = &[
    "encoder.onnx",
    "classifier.onnx",
    "boundary_marginals.onnx",
    "boundary_scorer.onnx",
    "boundary_explicit_scorer.onnx",
    "boundary_records.onnx",
    "boundary_relations.onnx",
];

/// Immutable source identity accepted for a GLiNER2.5 boundary model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryModelPin {
    pub hf_model: &'static str,
    pub hf_revision: &'static str,
    pub encoder_model: &'static str,
}

pub const BOUNDARY_MODEL_PINS: &[BoundaryModelPin] = &[
    BoundaryModelPin {
        hf_model: "fastino/gliner2.5-small-v1",
        hf_revision: "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
        encoder_model: "microsoft/deberta-v3-xsmall",
    },
    BoundaryModelPin {
        hf_model: "fastino/gliner2.5-base-v1",
        hf_revision: "78cea040597df251eedefa9d7ee2a756af39fe64",
        encoder_model: "microsoft/deberta-v3-base",
    },
    BoundaryModelPin {
        hf_model: "fastino/gliner2.5-multi-v1",
        hf_revision: "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        encoder_model: "microsoft/mdeberta-v3-base",
    },
];

/// Publication state recorded in a boundary manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BundleStatus {
    ExportedUnvalidated,
    Validated,
}

/// Authenticated size and digest for one bundle-relative file.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BundleFileMetadata {
    pub bytes: u64,
    pub sha256: String,
}

/// One typed ONNX input or output signature.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TensorSignature {
    pub dtype: String,
    pub shape: Vec<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Named input and output signatures for one ONNX graph.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GraphMetadata {
    pub inputs: BTreeMap<String, TensorSignature>,
    pub outputs: BTreeMap<String, TensorSignature>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Version 1 boundary bundle manifest.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BundleManifest {
    pub manifest_version: u64,
    pub architecture: String,
    pub architecture_version: u64,
    pub status: BundleStatus,
    pub release_ready: bool,
    pub hf_model: String,
    pub hf_revision: String,
    pub gliner2_commit: String,
    pub opset: u64,
    pub precision: String,
    pub ort_crate_version: String,
    pub native_onnx_runtime: String,
    pub validation_onnxruntime: String,
    pub dependencies: BTreeMap<String, String>,
    pub source_file_sha256: BTreeMap<String, String>,
    pub files: BTreeMap<String, BundleFileMetadata>,
    pub graphs: BTreeMap<String, GraphMetadata>,
    pub validation: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A complete boundary bundle whose manifest, identity, sizes and checksums
/// have passed validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedBundle {
    /// Canonical bundle directory, used to prevent symlink escapes.
    pub root: PathBuf,
    pub manifest: BundleManifest,
}

/// Validate a complete, release-ready GLiNER2.5 boundary bundle.
///
/// This parses `export_manifest.json` with a bounded allocation, checks all
/// fixed architecture/runtime/source pins, rejects unsafe relative paths and
/// escaping symlinks, then streams every listed file through SHA-256.
/// Exported-but-unvalidated development bundles are intentionally rejected.
pub fn validate_bundle(path: impl AsRef<Path>) -> Result<ValidatedBundle> {
    let requested_root = path.as_ref();
    let root = requested_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve bundle directory {}",
            requested_root.display()
        )
    })?;
    ensure!(
        root.is_dir(),
        "bundle path is not a directory: {}",
        requested_root.display()
    );

    let manifest_path = resolve_existing_file(&root, Path::new(MANIFEST_FILE))
        .context("boundary bundle is missing a safe export_manifest.json")?;
    let manifest = read_manifest(&manifest_path)?;
    let pin = validate_manifest_metadata(&manifest)?;

    for required in BOUNDARY_REQUIRED_FILES {
        ensure!(
            manifest.files.contains_key(*required),
            "boundary manifest is partial: missing required file entry `{required}`"
        );
    }
    ensure!(
        !manifest.files.contains_key(MANIFEST_FILE),
        "boundary manifest must not authenticate itself"
    );

    // Resolve the entire untrusted path set before reading any potentially large
    // graph. A late unsafe entry must fail before checksum work begins.
    let mut resolved_files = BTreeMap::new();
    for (relative, expected) in &manifest.files {
        validate_relative_path(relative)?;
        ensure!(
            expected.bytes > 0,
            "manifest file `{relative}` has a non-positive byte size"
        );
        validate_sha256(&expected.sha256)
            .with_context(|| format!("invalid SHA-256 for manifest file `{relative}`"))?;
        let resolved = resolve_existing_file(&root, Path::new(relative))
            .with_context(|| format!("missing or unsafe manifest file `{relative}`"))?;
        resolved_files.insert(relative.as_str(), resolved);
    }
    validate_bundle_contents(&root, manifest.files.keys().map(String::as_str))?;

    for (relative, expected) in &manifest.files {
        let resolved = &resolved_files[relative.as_str()];
        let (actual_bytes, actual_sha256) = hash_file(resolved)
            .with_context(|| format!("failed to checksum manifest file `{relative}`"))?;
        ensure!(
            actual_bytes == expected.bytes,
            "size mismatch for `{relative}`: expected {}, got {actual_bytes}",
            expected.bytes
        );
        ensure!(
            actual_sha256 == expected.sha256,
            "SHA-256 mismatch for `{relative}`: expected {}, got {actual_sha256}",
            expected.sha256
        );
    }

    validate_config_identity(&resolved_files["config.json"], pin)?;

    Ok(ValidatedBundle { root, manifest })
}

fn read_manifest(path: &Path) -> Result<BundleManifest> {
    let file = File::open(path)
        .with_context(|| format!("failed to open boundary manifest at {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read boundary manifest at {}", path.display()))?;
    ensure!(
        !bytes.is_empty(),
        "boundary manifest is empty at {}",
        path.display()
    );
    ensure!(
        u64::try_from(bytes.len())? <= MAX_MANIFEST_BYTES,
        "boundary manifest exceeds the {} byte limit at {}",
        MAX_MANIFEST_BYTES,
        path.display()
    );

    let unique: UniqueJsonValue = serde_json::from_slice(&bytes)
        .with_context(|| format!("malformed boundary manifest at {}", path.display()))?;
    serde_json::from_value(unique.0)
        .with_context(|| format!("malformed boundary manifest at {}", path.display()))
}

fn validate_manifest_metadata(manifest: &BundleManifest) -> Result<&'static BoundaryModelPin> {
    ensure!(
        manifest.manifest_version == 1,
        "unsupported boundary manifest_version {}; expected 1",
        manifest.manifest_version
    );
    ensure!(
        manifest.architecture == "boundary",
        "unsupported bundle architecture `{}`; expected `boundary`",
        manifest.architecture
    );
    ensure!(
        manifest.architecture_version == 1,
        "unsupported boundary architecture_version {}; expected 1",
        manifest.architecture_version
    );
    ensure!(
        manifest.status == BundleStatus::Validated,
        "boundary bundle status is not `validated`"
    );
    ensure!(
        manifest.release_ready,
        "boundary bundle is not release-ready"
    );
    ensure!(
        manifest
            .validation
            .as_ref()
            .and_then(Value::as_object)
            .is_some_and(|value| !value.is_empty()),
        "validated boundary bundle has no validation evidence object"
    );

    ensure!(
        manifest.gliner2_commit == GLINER2_COMMIT,
        "unsupported GLiNER2 source commit `{}`",
        manifest.gliner2_commit
    );
    let pin = BOUNDARY_MODEL_PINS
        .iter()
        .find(|pin| pin.hf_model == manifest.hf_model)
        .ok_or_else(|| anyhow!("unsupported boundary model `{}`", manifest.hf_model))?;
    ensure!(
        manifest.hf_revision == pin.hf_revision,
        "wrong immutable revision `{}` for {}; expected {}",
        manifest.hf_revision,
        pin.hf_model,
        pin.hf_revision
    );

    ensure!(
        manifest.opset == 17,
        "unsupported ONNX opset {}; expected 17",
        manifest.opset
    );
    ensure!(
        manifest.precision == "fp32",
        "unsupported precision `{}`; expected `fp32`",
        manifest.precision
    );
    ensure!(
        manifest.ort_crate_version == "2.0.0-rc.13",
        "unsupported ort crate version `{}`; expected `2.0.0-rc.13`",
        manifest.ort_crate_version
    );
    ensure!(
        manifest.native_onnx_runtime == "1.28.0",
        "unsupported native ONNX Runtime `{}`; expected `1.28.0`",
        manifest.native_onnx_runtime
    );
    ensure!(
        manifest.validation_onnxruntime == "1.20.1",
        "unsupported validation onnxruntime `{}`; expected `1.20.1`",
        manifest.validation_onnxruntime
    );
    ensure!(
        manifest.dependencies.len() == EXPECTED_EXPORT_DEPENDENCIES.len()
            && EXPECTED_EXPORT_DEPENDENCIES
                .iter()
                .all(
                    |(name, version)| manifest.dependencies.get(*name).map(String::as_str)
                        == Some(*version)
                ),
        "boundary manifest export dependency pins do not match the supported environment"
    );
    ensure!(
        !manifest.source_file_sha256.is_empty(),
        "boundary manifest has no source file hashes"
    );
    for (source, digest) in &manifest.source_file_sha256 {
        validate_relative_path(source)
            .with_context(|| format!("invalid source path `{source}`"))?;
        validate_sha256(digest).with_context(|| format!("invalid source hash for `{source}`"))?;
    }

    ensure!(
        manifest.graphs.len() == BOUNDARY_GRAPH_FILES.len()
            && BOUNDARY_GRAPH_FILES
                .iter()
                .all(|name| manifest.graphs.contains_key(*name)),
        "boundary manifest graph set does not match the seven required graphs"
    );
    for (name, graph) in &manifest.graphs {
        validate_tensor_table(name, "inputs", &graph.inputs)?;
        validate_tensor_table(name, "outputs", &graph.outputs)?;
    }

    Ok(pin)
}

fn validate_tensor_table(
    graph_name: &str,
    axis: &str,
    tensors: &BTreeMap<String, TensorSignature>,
) -> Result<()> {
    ensure!(
        !tensors.is_empty(),
        "graph `{graph_name}` has no {axis} signatures"
    );
    for (tensor_name, signature) in tensors {
        ensure!(
            valid_abi_name(tensor_name),
            "graph `{graph_name}` has an invalid {axis} tensor name"
        );
        ensure!(
            ONNX_TENSOR_DTYPES.contains(&signature.dtype.as_str()),
            "graph `{graph_name}` tensor `{tensor_name}` has invalid ONNX dtype `{}`",
            signature.dtype
        );
        for dimension in &signature.shape {
            let valid = match dimension {
                Value::Null => true,
                Value::Number(value) => value.as_u64().is_some(),
                Value::String(value) => valid_abi_name(value),
                _ => false,
            };
            ensure!(
                valid,
                "graph `{graph_name}` tensor `{tensor_name}` has an invalid shape dimension"
            );
        }
    }
    Ok(())
}

fn valid_abi_name(value: &str) -> bool {
    !value.is_empty()
        && value.trim() == value
        && !value
            .chars()
            .any(|character| character <= '\u{1f}' || character == '\u{7f}')
}

fn validate_config_identity(path: &Path, pin: &BoundaryModelPin) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open authenticated config at {}", path.display()))?;
    let unique: UniqueJsonValue = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("malformed authenticated config at {}", path.display()))?;
    let value = unique.0;
    let object = value.as_object().ok_or_else(|| {
        anyhow!(
            "authenticated config at {} is not an object",
            path.display()
        )
    })?;

    ensure!(
        object.get("architecture").and_then(Value::as_str) == Some("boundary"),
        "authenticated config has the wrong model architecture"
    );
    ensure!(
        object.get("architecture_version").and_then(Value::as_u64) == Some(1),
        "authenticated config has the wrong architecture_version"
    );
    ensure!(
        object.get("model_name").and_then(Value::as_str) == Some(pin.encoder_model),
        "authenticated config has the wrong encoder model for {}",
        pin.hf_model
    );
    Ok(())
}

fn validate_relative_path(relative: &str) -> Result<()> {
    ensure!(!relative.is_empty(), "manifest path is empty");
    ensure!(
        !relative.contains('\\'),
        "manifest path uses a Windows separator"
    );
    ensure!(!relative.starts_with('/'), "manifest path is absolute");
    ensure!(
        !(relative.as_bytes().get(1) == Some(&b':')
            && relative.as_bytes()[0].is_ascii_alphabetic()),
        "manifest path has a Windows drive prefix"
    );
    ensure!(
        relative
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != ".."),
        "manifest path is not a normalized relative path"
    );
    Ok(())
}

fn validate_bundle_contents<'a>(
    root: &Path,
    manifest_files: impl Iterator<Item = &'a str>,
) -> Result<()> {
    let mut allowed = manifest_files
        .map(|relative| root.join(relative))
        .collect::<BTreeSet<_>>();
    allowed.insert(root.join(MANIFEST_FILE));

    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory).with_context(|| {
            format!("failed to inspect bundle directory {}", directory.display())
        })? {
            let entry = entry.with_context(|| {
                format!("failed to inspect bundle directory {}", directory.display())
            })?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect bundle entry {}", path.display()))?;
            if file_type.is_dir() {
                directories.push(path);
            } else {
                ensure!(
                    allowed.contains(&path),
                    "bundle contains unlisted file `{}`",
                    path.strip_prefix(root).unwrap_or(&path).display()
                );
            }
        }
    }
    Ok(())
}

fn resolve_existing_file(root: &Path, relative: &Path) -> Result<PathBuf> {
    let resolved = root.join(relative).canonicalize().with_context(|| {
        format!(
            "failed to resolve {} below {}",
            relative.display(),
            root.display()
        )
    })?;
    ensure!(
        resolved.starts_with(root),
        "resolved path {} escapes bundle directory {}",
        resolved.display(),
        root.display()
    );
    ensure!(
        resolved.is_file(),
        "resolved path is not a regular file: {}",
        resolved.display()
    );
    Ok(resolved)
}

fn validate_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "expected 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read)?)
            .ok_or_else(|| anyhow!("file size overflow while hashing {}", path.display()))?;
        hasher.update(&buffer[..read]);
    }

    Ok((bytes, lowercase_hex(&hasher.finalize())))
}

struct UniqueJsonValue(Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut values: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut sequence = Vec::new();
        while let Some(value) = values.next_element::<UniqueJsonValue>()? {
            sequence.push(value.0);
        }
        Ok(UniqueJsonValue(Value::Array(sequence)))
    }

    fn visit_map<A>(self, mut values: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut object = Map::new();
        while let Some(key) = values.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key `{key}`"
                )));
            }
            let value = values.next_value::<UniqueJsonValue>()?;
            object.insert(key, value.0);
        }
        Ok(UniqueJsonValue(Value::Object(object)))
    }
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}
