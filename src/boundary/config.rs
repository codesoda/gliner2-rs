use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Map, Value};

use super::{decode::OverlapPolicy, pool::PoolConfig, relation_pairs::RelationProposalConfig};
use crate::config::{Architecture, ModelConfig};

/// Runtime settings which are not baked into the boundary ONNX graphs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryRuntimeConfig {
    pub max_len: usize,
    pub pair_temperature: f32,
    pub classification_temperature: f32,
    pub record_temperature: f32,
    pub relation_temperature: f32,
    pub abstention_threshold: f32,
    pub overlap_policy: OverlapPolicy,
    pub pool: PoolConfig,
    pub relation_proposals: RelationProposalConfig,
}

impl BoundaryRuntimeConfig {
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        let model = ModelConfig::from_dir(bundle)?;
        ensure!(
            model.architecture == Architecture::Boundary,
            "BoundaryPipeline requires `architecture: boundary` in {}",
            bundle.join("config.json").display()
        );
        ensure!(
            model.architecture_version == Some(1),
            "boundary architecture_version must be 1"
        );

        let path = bundle.join("config.json");
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read boundary config at {}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("malformed boundary config at {}", path.display()))?;
        let object = value.as_object().ok_or_else(|| {
            anyhow!(
                "malformed boundary config at {}: expected a JSON object",
                path.display()
            )
        })?;
        Self::from_object(object, model.max_len.unwrap_or(4096), &path)
    }

    fn from_object(object: &Map<String, Value>, max_len: usize, path: &Path) -> Result<Self> {
        if let Some(token_pooling) = object.get("token_pooling") {
            ensure_string_eq(token_pooling, "token_pooling", "first", path)?;
        }

        let head = match object.get("boundary_head") {
            None => None,
            Some(Value::Object(head)) => Some(head),
            Some(_) => {
                return Err(anyhow!(
                    "malformed `boundary_head` in {}: expected an object",
                    path.display()
                ));
            }
        };

        // These settings change the exported marginal/scorer graph. Accepting a
        // different value would load valid ONNX while silently applying the
        // wrong architecture contract.
        for (name, expected) in [
            ("candidate_pool", Value::String("shared".to_owned())),
            ("boundary_dim", Value::from(128)),
            ("pair_dim", Value::from(128)),
            ("record_dim", Value::from(128)),
            ("record_instance_queries", Value::from(32)),
            ("boundary_attention_layers", Value::from(2)),
            ("boundary_attention_heads", Value::from(4)),
            ("boundary_attention_window", Value::from(128)),
            ("boundary_refinement_layers", Value::from(1)),
            ("candidate_attention_layers", Value::from(0)),
            ("candidate_attention_heads", Value::from(4)),
            ("query_attention_layers", Value::from(0)),
            ("enable_span_content", Value::Bool(true)),
            ("content_dim", Value::from(64)),
            ("content_soft_max_pool", Value::Bool(false)),
            ("use_inside_evidence", Value::Bool(true)),
            ("query_conditioned_inside_weight", Value::Bool(true)),
            ("reranker_endpoint_compat", Value::Bool(true)),
            ("bidirectional_proposals", Value::Bool(true)),
            ("enable_abstention", Value::Bool(true)),
            ("enable_count_head", Value::Bool(true)),
            ("enable_records", Value::Bool(true)),
            ("enable_relations", Value::Bool(true)),
            ("enable_rotary_endpoints", Value::Bool(true)),
            ("endpoint_difference_features", Value::Bool(true)),
            ("multihead_pair_compat_heads", Value::from(8)),
            ("directional_relation_states", Value::Bool(true)),
            ("relation_biaffine_content", Value::Bool(true)),
            ("adaptive_threshold", Value::Bool(false)),
        ] {
            if let Some(actual) = head.and_then(|head| head.get(name)) {
                ensure!(
                    actual == &expected,
                    "unsupported boundary graph setting `{name}`={} in {}; expected {}",
                    actual,
                    path.display(),
                    expected
                );
            }
        }

        exact_f64(head, "boundary_ffn_multiplier", 2.0, path)?;
        exact_f64(head, "rotary_base", 10_000.0, path)?;

        let pair_temperature = positive_f32(head, "pair_temperature", 1.0, path)?;
        let classification_temperature =
            positive_f32(head, "classification_temperature", 1.0, path)?;
        let record_temperature = positive_f32(head, "record_temperature", 1.0, path)?;
        let relation_temperature = positive_f32(head, "relation_temperature", 1.0, path)?;
        let abstention_threshold = probability(head, "abstention_threshold", 0.5, path)?;
        let boundary_top_k = positive_usize(head, "pool_boundary_top_k", 32, path)?;
        let capacity = positive_usize(head, "pool_size", 192, path)?;
        let min_per_query = positive_usize(head, "min_pool_per_query", 8, path)?;
        let heads_per_relation = positive_usize(head, "relation_heads_per_type", 32, path)?;
        let tails_per_relation = positive_usize(head, "relation_tails_per_type", 32, path)?;
        let pair_cap = positive_usize(head, "relation_pair_cap", 64, path)?;
        ensure!(
            heads_per_relation.checked_mul(tails_per_relation).is_some(),
            "boundary relation_heads_per_type * relation_tails_per_type overflows usize in {}",
            path.display()
        );
        let argument_threshold =
            probability(head, "relation_argument_proposal_threshold", 0.2, path)?;
        ensure!(
            min_per_query <= capacity,
            "boundary min_pool_per_query {min_per_query} exceeds pool_size {capacity} in {}",
            path.display()
        );
        let overlap = optional_string(head, "overlap_policy", "flat", path)?;
        let overlap_policy = overlap.parse().with_context(|| {
            format!(
                "invalid boundary overlap_policy {overlap:?} in {}",
                path.display()
            )
        })?;

        Ok(Self {
            max_len,
            pair_temperature,
            classification_temperature,
            record_temperature,
            relation_temperature,
            abstention_threshold,
            overlap_policy,
            pool: PoolConfig {
                boundary_top_k,
                capacity,
                min_per_query,
            },
            relation_proposals: RelationProposalConfig {
                heads_per_relation,
                tails_per_relation,
                pair_cap,
                argument_threshold,
            },
        })
    }
}

fn field<'a>(head: Option<&'a Map<String, Value>>, name: &str) -> Option<&'a Value> {
    head.and_then(|head| head.get(name))
}

fn positive_f32(
    head: Option<&Map<String, Value>>,
    name: &str,
    default: f32,
    path: &Path,
) -> Result<f32> {
    let Some(value) = field(head, name) else {
        return Ok(default);
    };
    let number = value.as_f64().ok_or_else(|| {
        anyhow!(
            "malformed boundary `{name}` in {}: expected a number",
            path.display()
        )
    })?;
    ensure!(
        number.is_finite() && number > 0.0 && number <= f64::from(f32::MAX),
        "boundary `{name}` in {} must be finite and positive",
        path.display()
    );
    let number = number as f32;
    ensure!(
        number.is_finite() && number > 0.0,
        "boundary `{name}` in {} must remain finite and positive when represented as f32",
        path.display()
    );
    Ok(number)
}

fn exact_f64(
    head: Option<&Map<String, Value>>,
    name: &str,
    expected: f64,
    path: &Path,
) -> Result<()> {
    let Some(value) = field(head, name) else {
        return Ok(());
    };
    let number = value.as_f64().ok_or_else(|| {
        anyhow!(
            "malformed boundary `{name}` in {}: expected a number",
            path.display()
        )
    })?;
    ensure!(
        number.is_finite() && number == expected,
        "unsupported boundary graph setting `{name}`={} in {}; expected {}",
        value,
        path.display(),
        expected
    );
    Ok(())
}

fn probability(
    head: Option<&Map<String, Value>>,
    name: &str,
    default: f32,
    path: &Path,
) -> Result<f32> {
    let Some(value) = field(head, name) else {
        return Ok(default);
    };
    let number = value.as_f64().ok_or_else(|| {
        anyhow!(
            "malformed boundary `{name}` in {}: expected a number",
            path.display()
        )
    })?;
    ensure!(
        number.is_finite() && (0.0..=1.0).contains(&number),
        "boundary `{name}` in {} must be a probability in [0,1]",
        path.display()
    );
    Ok(number as f32)
}

fn positive_usize(
    head: Option<&Map<String, Value>>,
    name: &str,
    default: usize,
    path: &Path,
) -> Result<usize> {
    let Some(value) = field(head, name) else {
        return Ok(default);
    };
    let number = value.as_u64().ok_or_else(|| {
        anyhow!(
            "malformed boundary `{name}` in {}: expected a positive integer",
            path.display()
        )
    })?;
    ensure!(
        number > 0,
        "boundary `{name}` in {} must be positive",
        path.display()
    );
    usize::try_from(number)
        .with_context(|| format!("boundary `{name}` in {} does not fit usize", path.display()))
}

fn optional_string(
    head: Option<&Map<String, Value>>,
    name: &str,
    default: &str,
    path: &Path,
) -> Result<String> {
    match field(head, name) {
        None => Ok(default.to_owned()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(anyhow!(
            "malformed boundary `{name}` in {}: expected a string",
            path.display()
        )),
    }
}

fn ensure_string_eq(value: &Value, name: &str, expected: &str, path: &Path) -> Result<()> {
    match value {
        Value::String(actual) if actual == expected => Ok(()),
        Value::String(actual) => Err(anyhow!(
            "unsupported boundary `{name}` {actual:?} in {}; expected {expected:?}",
            path.display()
        )),
        _ => Err(anyhow!(
            "malformed boundary `{name}` in {}: expected a string",
            path.display()
        )),
    }
}
