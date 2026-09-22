use std::{fs, path::Path};

use anyhow::{Context, anyhow};
use serde_json::{Map, Value};

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Span,
    Boundary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub architecture_version: Option<u64>,
    pub max_len: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::Span,
            architecture_version: None,
            max_len: None,
        }
    }
}

impl ModelConfig {
    /// Load `config.json` from a model directory.
    ///
    /// Legacy span bundles may omit the file or the `architecture` key. Boundary
    /// bundles currently support architecture version 1 and default to 4096
    /// words when `max_len` is absent.
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        if !path.exists() {
            return Ok(Self::default());
        }

        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read model config at {}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("malformed model config at {}", path.display()))?;
        let object = value.as_object().ok_or_else(|| {
            anyhow!(
                "malformed model config at {}: expected a JSON object",
                path.display()
            )
        })?;

        Self::from_object(object, &path)
    }

    fn from_object(object: &Map<String, Value>, path: &Path) -> Result<Self> {
        let architecture = match object.get("architecture") {
            None => Architecture::Span,
            Some(Value::String(value)) if value == "span" => Architecture::Span,
            Some(Value::String(value)) if value == "boundary" => Architecture::Boundary,
            Some(Value::String(value)) => {
                return Err(anyhow!(
                    "unsupported architecture `{value}` in {}",
                    path.display()
                ));
            }
            Some(_) => {
                return Err(anyhow!(
                    "malformed `architecture` in {}: expected `span` or `boundary`",
                    path.display()
                ));
            }
        };

        let explicit_version = parse_optional_u64(object, "architecture_version", path)?;
        if let Some(version) = explicit_version
            && version != 1
        {
            return Err(anyhow!(
                "unsupported architecture_version {version} in {}; only version 1 is supported",
                path.display()
            ));
        }

        let max_len =
            match parse_optional_u64(object, "max_len", path)? {
                Some(0) => {
                    return Err(anyhow!(
                        "malformed `max_len` in {}: expected a positive integer",
                        path.display()
                    ));
                }
                Some(value) => Some(usize::try_from(value).with_context(|| {
                    format!("`max_len` in {} does not fit usize", path.display())
                })?),
                None if architecture == Architecture::Boundary => Some(4096),
                None => None,
            };

        let boundary_head = match object.get("boundary_head") {
            Some(Value::Object(head)) => Some(head),
            Some(_) if architecture == Architecture::Boundary => {
                return Err(anyhow!(
                    "malformed `boundary_head` in {}: expected an object",
                    path.display()
                ));
            }
            _ => None,
        };
        if architecture == Architecture::Boundary
            && let Some(candidate_pool) = boundary_head.and_then(|head| head.get("candidate_pool"))
        {
            match candidate_pool {
                Value::String(value) if value == "shared" => {}
                Value::String(value) => {
                    return Err(anyhow!(
                        "unsupported boundary candidate_pool `{value}` in {}; expected `shared`",
                        path.display()
                    ));
                }
                _ => {
                    return Err(anyhow!(
                        "malformed boundary `candidate_pool` in {}: expected a string",
                        path.display()
                    ));
                }
            }
        }

        Ok(Self {
            architecture,
            architecture_version: match architecture {
                Architecture::Boundary => Some(explicit_version.unwrap_or(1)),
                Architecture::Span => explicit_version,
            },
            max_len,
        })
    }
}

fn parse_optional_u64(
    object: &Map<String, Value>,
    field: &str,
    path: &Path,
) -> Result<Option<u64>> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::Number(value)) => value.as_u64().map(Some).ok_or_else(|| {
            anyhow!(
                "malformed `{field}` in {}: expected a non-negative integer",
                path.display()
            )
        }),
        Some(_) => Err(anyhow!(
            "malformed `{field}` in {}: expected an integer",
            path.display()
        )),
    }
}
