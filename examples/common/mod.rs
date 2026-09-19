use std::{env, path::PathBuf};

/// Resolve model/ONNX paths, with optional `--model <onnx_dir>` override.
/// Defaults to the base ONNX bundle (`onnx/gliner2-base-v1`).
#[allow(dead_code)] // Shared by examples that use different subsets of these paths.
pub struct ModelPaths {
    pub model_dir: PathBuf,
    pub onnx_dir: PathBuf,
    pub encoder: PathBuf,
    pub extractor: PathBuf,
    pub classifier: Option<PathBuf>,
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[allow(dead_code)] // `download_models` includes this shared module but needs only `repo_root`.
pub fn model_paths_from_args(default_onnx_rel: &str) -> ModelPaths {
    let root = repo_root();
    let default_onnx = root.join(default_onnx_rel);
    let default_name = default_onnx
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("gliner2-base-v1")
        .to_string();

    let mut override_path: Option<PathBuf> = None;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        if arg == "--model" {
            if let Some(val) = args.next() {
                override_path = Some(PathBuf::from(val));
            }
        } else if let Some(rest) = arg.strip_prefix("--model=") {
            override_path = Some(PathBuf::from(rest));
        }
    }

    let onnx_dir = override_path
        .map(|p| if p.is_relative() { root.join(p) } else { p })
        .unwrap_or(default_onnx);

    // Best-effort model_dir: if override provided, assume sibling under /models/<name>.
    let model_dir = onnx_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(|name| root.join("models").join(name))
        .unwrap_or_else(|| root.join("models").join(default_name));

    let encoder = onnx_dir.join("encoder.onnx");
    let extractor = onnx_dir.join("extractor_padded.onnx");
    let classifier_path = onnx_dir.join("classifier.onnx");
    let classifier = if classifier_path.exists() {
        Some(classifier_path)
    } else {
        None
    };

    ModelPaths {
        model_dir,
        onnx_dir,
        encoder,
        extractor,
        classifier,
    }
}
