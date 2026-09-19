use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use gliner2_rs::Result;

/// Artifact root for integration tests. Defaults to the crate root.
pub fn model_root() -> PathBuf {
    env::var_os("GLINER2_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

/// Return false to skip in ordinary no-model CI, but fail when strict artifact
/// mode is requested with `GLINER2_REQUIRE_MODELS=1`.
pub fn artifacts_available(paths: &[&Path]) -> Result<bool> {
    let missing: Vec<_> = paths.iter().filter(|path| !path.exists()).collect();
    if missing.is_empty() {
        return Ok(true);
    }

    let message = format!(
        "missing required model artifacts: {}",
        missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if env::var("GLINER2_REQUIRE_MODELS").as_deref() == Ok("1") {
        return Err(anyhow!(message));
    }

    eprintln!("SKIP: {message}");
    Ok(false)
}
