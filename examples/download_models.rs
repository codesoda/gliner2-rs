//! Manually download verified GLiNER2 model bundles from Hugging Face.
//!
//! This is an explicit operation; crate build and inference never download files.
//!
//! ```bash
//! cargo run --example download_models -- --model base
//! cargo run --example download_models -- --model 2.5-small --dest ./onnx --revision <immutable-hosted-commit>
//! cargo run --example download_models -- --model all --revision <immutable-hosted-commit>
//! ```
//! `all` downloads all five bundles and can require several gigabytes.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use gliner2_rs::bundle::validate_bundle;
use hf_hub::{HFClientSync, HFRepositorySync, repository::RepoTypeModel, split_id};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

mod common;
use common::repo_root;

const V2_PINS_JSON: &str = include_str!("../docs/checkpoints/v2-metadata-pins.json");
const ALL_SELECTORS: &str = "base|large|2.5-small|2.5-base|2.5-multi|all";

#[derive(Debug, Deserialize)]
struct V2Pins {
    schema_version: u64,
    hash_algorithm: String,
    hosted_onnx: HostedOnnx,
    models: BTreeMap<String, V2ModelPin>,
}

#[derive(Debug, Deserialize)]
struct HostedOnnx {
    repository: String,
    verified_revision: String,
}

#[derive(Debug, Deserialize)]
struct V2ModelPin {
    selector: String,
    source_repository: String,
    source_revision: String,
    source_metadata: BTreeMap<String, PinnedFile>,
    hosted_onnx_files: BTreeMap<String, PinnedFile>,
    observed_identity: V2ObservedIdentity,
}

#[derive(Debug, Deserialize)]
struct PinnedFile {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct V2ObservedIdentity {
    model_name: String,
    max_width: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    V2,
    Boundary,
}

#[derive(Debug, Clone, Copy)]
struct ModelSpec {
    selector: &'static str,
    bundle_name: &'static str,
    kind: ModelKind,
    boundary_source_model: Option<&'static str>,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        selector: "base",
        bundle_name: "gliner2-base-v1",
        kind: ModelKind::V2,
        boundary_source_model: None,
    },
    ModelSpec {
        selector: "large",
        bundle_name: "gliner2-large-v1",
        kind: ModelKind::V2,
        boundary_source_model: None,
    },
    ModelSpec {
        selector: "2.5-small",
        bundle_name: "gliner2.5-small-v1",
        kind: ModelKind::Boundary,
        boundary_source_model: Some("fastino/gliner2.5-small-v1"),
    },
    ModelSpec {
        selector: "2.5-base",
        bundle_name: "gliner2.5-base-v1",
        kind: ModelKind::Boundary,
        boundary_source_model: Some("fastino/gliner2.5-base-v1"),
    },
    ModelSpec {
        selector: "2.5-multi",
        bundle_name: "gliner2.5-multi-v1",
        kind: ModelKind::Boundary,
        boundary_source_model: Some("fastino/gliner2.5-multi-v1"),
    },
];

fn main() -> Result<()> {
    let pins: V2Pins =
        serde_json::from_str(V2_PINS_JSON).context("failed to parse embedded v2 metadata pins")?;
    validate_v2_pins(&pins)?;

    let (model, dest, revision) = parse_args(&pins)?;
    let selected = select_models(&model)?;
    if model == "all" {
        println!("Downloading all five bundles; this can require several gigabytes.");
    }

    prepare_destination(&dest)?;
    let client = HFClientSync::new()?;
    let (hosted_owner, hosted_name) = split_id(&pins.hosted_onnx.repository);
    let hosted_repo = client.model(hosted_owner, hosted_name);

    for spec in selected {
        println!(
            "Downloading {} from {} at {} ...",
            spec.bundle_name, pins.hosted_onnx.repository, revision
        );
        let staging_root = unique_sibling(&dest, "download");
        std::fs::create_dir(&staging_root).with_context(|| {
            format!(
                "failed to create private staging directory {}",
                staging_root.display()
            )
        })?;
        let staged_bundle = staging_root.join(spec.bundle_name);
        let download_result = (|| -> Result<()> {
            match spec.kind {
                ModelKind::Boundary => {
                    hosted_repo
                        .snapshot_download()
                        .revision(revision.clone())
                        .allow_patterns(vec![
                            format!("{}/*", spec.bundle_name),
                            format!("{}/**", spec.bundle_name),
                        ])
                        .local_dir(staging_root.clone())
                        .send()
                        .with_context(|| {
                            format!(
                                "failed to download {} from {} at {}",
                                spec.bundle_name, pins.hosted_onnx.repository, revision
                            )
                        })?;
                    let validated = validate_bundle(&staged_bundle).with_context(|| {
                        format!(
                            "downloaded boundary bundle {} failed validation",
                            staged_bundle.display()
                        )
                    })?;
                    ensure!(
                        Some(validated.manifest.hf_model.as_str()) == spec.boundary_source_model,
                        "downloaded {} contains manifest identity {}",
                        spec.bundle_name,
                        validated.manifest.hf_model
                    );
                }
                ModelKind::V2 => {
                    let pin = pins.models.get(spec.bundle_name).ok_or_else(|| {
                        anyhow::anyhow!("embedded pins are missing {}", spec.bundle_name)
                    })?;
                    std::fs::create_dir(&staged_bundle)
                        .with_context(|| format!("failed to create {}", staged_bundle.display()))?;
                    download_hosted_v2_files(
                        &hosted_repo,
                        &staging_root,
                        spec.bundle_name,
                        &revision,
                        &pin.hosted_onnx_files,
                    )?;
                    validate_pinned_files(&staged_bundle, &pin.hosted_onnx_files)
                        .context("hosted v2 ONNX validation failed")?;
                    download_v2_metadata(&client, &staged_bundle, pin)?;
                    validate_pinned_files(&staged_bundle, &pin.source_metadata)
                        .context("v2 source metadata validation failed")?;
                    validate_exact_v2_contents(&staged_bundle, pin)?;
                    validate_v2_config(&staged_bundle.join("config.json"), &pin.observed_identity)?;
                }
            }
            install_bundle(&staged_bundle, &dest.join(spec.bundle_name))
        })();
        let _ = std::fs::remove_dir_all(&staging_root);
        download_result?;
        println!("  verified {}", dest.join(spec.bundle_name).display());
    }

    println!("Done. Verified models are under {}", dest.display());
    Ok(())
}

fn parse_args(pins: &V2Pins) -> Result<(String, PathBuf, String)> {
    let mut model = "all".to_string();
    let mut dest = repo_root().join("onnx");
    let mut revision = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = next_arg(&mut args, "--model")?,
            "--dest" => dest = next_arg(&mut args, "--dest")?.into(),
            "--revision" => revision = Some(next_arg(&mut args, "--revision")?),
            "-h" | "--help" => {
                println!(
                    "Usage: download_models [--model {ALL_SELECTORS}] [--dest DIR] [--revision IMMUTABLE_COMMIT]\n\n`all` downloads all five bundles. --revision is required for boundary/all until a validated bundle revision is published; base/large default to the verified v2 revision. Boundary downloads are accepted only when their manifest is validated and release-ready."
                );
                std::process::exit(0);
            }
            other => {
                bail!("unrecognized argument `{other}`; expected --model, --dest or --revision")
            }
        }
    }
    let revision = match revision {
        Some(revision) => revision,
        None => {
            ensure!(
                select_models(&model)?
                    .iter()
                    .all(|spec| spec.kind == ModelKind::V2),
                "boundary downloads require --revision with the immutable published commit containing validated bundles"
            );
            pins.hosted_onnx.verified_revision.clone()
        }
    };
    validate_commit(&revision).context("--revision must be an immutable 40-character commit")?;
    Ok((model, dest, revision))
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
}

fn select_models(model: &str) -> Result<Vec<&'static ModelSpec>> {
    if model == "all" {
        return Ok(MODELS.iter().collect());
    }
    MODELS
        .iter()
        .find(|spec| model == spec.selector || model == spec.bundle_name)
        .map(|spec| vec![spec])
        .ok_or_else(|| {
            anyhow::anyhow!("unknown --model value `{model}` (expected {ALL_SELECTORS})")
        })
}

fn validate_v2_pins(pins: &V2Pins) -> Result<()> {
    ensure!(
        pins.schema_version == 1,
        "unsupported v2 pin schema version"
    );
    ensure!(
        pins.hash_algorithm == "sha256",
        "unsupported v2 pin hash algorithm"
    );
    ensure!(
        pins.hosted_onnx.repository == "codesoda/gliner2-onnx",
        "unexpected hosted ONNX repository in v2 pins"
    );
    validate_commit(&pins.hosted_onnx.verified_revision)
        .context("v2 hosted ONNX revision is not immutable")?;

    for spec in MODELS.iter().filter(|spec| spec.kind == ModelKind::V2) {
        let pin = pins
            .models
            .get(spec.bundle_name)
            .ok_or_else(|| anyhow::anyhow!("v2 pins are missing model `{}`", spec.bundle_name))?;
        ensure!(
            pin.selector == spec.selector,
            "v2 selector mismatch for {}",
            spec.bundle_name
        );
        ensure!(
            pin.source_repository == format!("fastino/{}", spec.bundle_name),
            "unexpected source repository for {}",
            spec.bundle_name
        );
        validate_commit(&pin.source_revision).with_context(|| {
            format!("source revision for {} is not immutable", spec.bundle_name)
        })?;
        ensure!(
            pin.source_metadata.len() == 3
                && ["config.json", "tokenizer.json", "tokenizer_config.json"]
                    .iter()
                    .all(|name| pin.source_metadata.contains_key(*name)),
            "source metadata pin set is incomplete for {}",
            spec.bundle_name
        );
        ensure!(
            ["encoder.onnx", "extractor_padded.onnx", "classifier.onnx"]
                .iter()
                .all(|name| pin.hosted_onnx_files.contains_key(*name)),
            "hosted ONNX pin set is incomplete for {}",
            spec.bundle_name
        );
        for (name, file) in pin.source_metadata.iter().chain(&pin.hosted_onnx_files) {
            validate_pinned_file(name, file)?;
        }
    }
    Ok(())
}

fn download_hosted_v2_files(
    hosted_repo: &HFRepositorySync<RepoTypeModel>,
    staging_root: &Path,
    bundle_name: &str,
    revision: &str,
    files: &BTreeMap<String, PinnedFile>,
) -> Result<()> {
    for filename in files.keys() {
        let remote_name = format!("{bundle_name}/{filename}");
        hosted_repo
            .download_file()
            .filename(remote_name.clone())
            .revision(revision.to_owned())
            .local_dir(staging_root.to_path_buf())
            .send()
            .with_context(|| format!("failed to download hosted v2 file {remote_name}"))?;
    }
    Ok(())
}

fn download_v2_metadata(client: &HFClientSync, bundle_dir: &Path, pin: &V2ModelPin) -> Result<()> {
    let (owner, name) = split_id(&pin.source_repository);
    let source_repo = client.model(owner, name);
    for filename in pin.source_metadata.keys() {
        source_repo
            .download_file()
            .filename(filename.clone())
            .revision(pin.source_revision.clone())
            .local_dir(bundle_dir.to_path_buf())
            .send()
            .with_context(|| {
                format!(
                    "failed to download {filename} from {} at {}",
                    pin.source_repository, pin.source_revision
                )
            })?;
    }
    Ok(())
}

fn validate_pinned_files(root: &Path, files: &BTreeMap<String, PinnedFile>) -> Result<()> {
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve bundle directory {}", root.display()))?;
    for (name, expected) in files {
        validate_pinned_file(name, expected)?;
        let path = root.join(name);
        let resolved = path
            .canonicalize()
            .with_context(|| format!("missing required file {}", path.display()))?;
        ensure!(
            resolved.starts_with(&canonical_root) && resolved.is_file(),
            "pinned file escapes its bundle directory: {}",
            path.display()
        );
        let (bytes, sha256) = hash_file(&resolved)?;
        ensure!(
            bytes == expected.bytes,
            "size mismatch for {}: expected {}, got {bytes}",
            path.display(),
            expected.bytes
        );
        ensure!(
            sha256 == expected.sha256,
            "SHA-256 mismatch for {}: expected {}, got {sha256}",
            path.display(),
            expected.sha256
        );
    }
    Ok(())
}

fn validate_exact_v2_contents(root: &Path, pin: &V2ModelPin) -> Result<()> {
    let expected = pin
        .hosted_onnx_files
        .keys()
        .chain(pin.source_metadata.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut actual = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("failed to inspect v2 bundle {}", root.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        ensure!(
            file_type.is_file() || file_type.is_symlink(),
            "v2 bundle contains unexpected non-file entry {}",
            entry.path().display()
        );
        actual.insert(entry.file_name().to_string_lossy().into_owned());
    }
    ensure!(
        actual == expected,
        "v2 bundle contains files outside the authenticated pin set"
    );
    Ok(())
}

fn validate_pinned_file(name: &str, file: &PinnedFile) -> Result<()> {
    ensure!(
        !name.is_empty()
            && !name.contains('/')
            && !name.contains('\\')
            && name != "."
            && name != "..",
        "unsafe pinned filename `{name}`"
    );
    ensure!(file.bytes > 0, "pinned file `{name}` has zero bytes");
    validate_sha256(&file.sha256).with_context(|| format!("invalid pinned hash for `{name}`"))
}

fn validate_v2_config(path: &Path, expected: &V2ObservedIdentity) -> Result<()> {
    let value: Value = serde_json::from_reader(BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    ))
    .with_context(|| format!("malformed v2 config at {}", path.display()))?;
    ensure!(
        value.get("model_name").and_then(Value::as_str) == Some(&expected.model_name),
        "v2 config model identity mismatch at {}",
        path.display()
    );
    ensure!(
        value.get("max_width").and_then(Value::as_u64) == Some(expected.max_width),
        "v2 config max_width mismatch at {}",
        path.display()
    );
    if let Some(architecture) = value.get("architecture") {
        ensure!(
            architecture.as_str() == Some("span"),
            "v2 config has non-span architecture at {}",
            path.display()
        );
    }
    Ok(())
}

fn validate_commit(value: &str) -> Result<()> {
    ensure!(
        value.len() == 40
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "expected 40 lowercase hexadecimal characters"
    );
    Ok(())
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
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
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
            .ok_or_else(|| anyhow::anyhow!("file size overflow for {}", path.display()))?;
        hasher.update(&buffer[..read]);
    }
    Ok((bytes, lowercase_hex(&hasher.finalize())))
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

fn unique_sibling(parent: &Path, purpose: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    parent.join(format!(".gliner2-{purpose}-{}-{nonce}", std::process::id()))
}

fn install_bundle(staged: &Path, destination: &Path) -> Result<()> {
    let staged_metadata = std::fs::symlink_metadata(staged)
        .with_context(|| format!("staged bundle is missing: {}", staged.display()))?;
    ensure!(
        staged_metadata.file_type().is_dir() && !staged_metadata.file_type().is_symlink(),
        "staged bundle is not a real directory: {}",
        staged.display()
    );
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("bundle destination has no parent"))?;
    let backup = unique_sibling(parent, "backup");
    let destination_metadata = match std::fs::symlink_metadata(destination) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("failed to inspect bundle destination"),
    };
    if let Some(metadata) = &destination_metadata {
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "refusing to replace non-directory or symlink destination {}",
            destination.display()
        );
        std::fs::rename(destination, &backup).with_context(|| {
            format!(
                "failed to move existing bundle {} aside",
                destination.display()
            )
        })?;
    }

    if let Err(error) = std::fs::rename(staged, destination) {
        if destination_metadata.is_some()
            && let Err(restore_error) = std::fs::rename(&backup, destination)
        {
            return Err(anyhow::anyhow!(
                "failed to install {} ({error}) and restore its previous bundle from {} ({restore_error})",
                destination.display(),
                backup.display()
            ));
        }
        return Err(error).with_context(|| {
            format!(
                "failed to install verified bundle at {}",
                destination.display()
            )
        });
    }

    if destination_metadata.is_some() {
        std::fs::remove_dir_all(&backup)
            .with_context(|| format!("failed to remove replaced bundle at {}", backup.display()))?;
    }
    Ok(())
}

fn prepare_destination(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
                "destination must be a real directory: {}",
                path.display()
            );
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir_all(path)
            .with_context(|| format!("failed to create destination {}", path.display())),
        Err(error) => {
            Err(error).with_context(|| format!("failed to inspect destination {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_pins_and_selectors_are_model_free() {
        let pins: V2Pins = serde_json::from_str(V2_PINS_JSON).unwrap();
        validate_v2_pins(&pins).unwrap();
        assert_eq!(select_models("all").unwrap().len(), 5);
        assert_eq!(
            select_models("gliner2.5-multi-v1").unwrap()[0].selector,
            "2.5-multi"
        );
        assert!(select_models("small").is_err());
    }

    #[test]
    fn install_replaces_bundle_only_after_staging() {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("bundle");
        let staged = temp.path().join("staged");
        std::fs::create_dir_all(&destination).unwrap();
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(destination.join("old"), b"old").unwrap();
        std::fs::write(staged.join("new"), b"new").unwrap();

        install_bundle(&staged, &destination).unwrap();
        assert!(!destination.join("old").exists());
        assert_eq!(std::fs::read(destination.join("new")).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn install_refuses_symlink_destination() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let actual = temp.path().join("actual");
        let destination = temp.path().join("bundle");
        let staged = temp.path().join("staged");
        std::fs::create_dir(&actual).unwrap();
        std::fs::create_dir(&staged).unwrap();
        std::fs::write(actual.join("old"), b"old").unwrap();
        symlink(&actual, &destination).unwrap();

        let error = install_bundle(&staged, &destination)
            .unwrap_err()
            .to_string();
        assert!(error.contains("symlink destination"));
        assert_eq!(std::fs::read(actual.join("old")).unwrap(), b"old");
    }
}
