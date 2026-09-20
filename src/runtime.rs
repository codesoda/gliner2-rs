use std::collections::BTreeSet;
use std::fmt::Debug;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn};
use ort::session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel};
use ort::value::{PrimitiveTensorElementType, Tensor};

/// A directly-owned ONNX Runtime session with the crate's fixed CPU settings.
///
/// `ort` requires mutable session access for inference. The lock is per model,
/// so independent model sessions can still execute concurrently.
pub(crate) struct RuntimeSession {
    session: Mutex<Session>,
    model_name: &'static str,
}

impl RuntimeSession {
    pub(crate) fn load(
        model_path: impl AsRef<Path>,
        model_name: &'static str,
        expected_inputs: &[&str],
        expected_outputs: &[&str],
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let mut builder = Session::builder()
            .with_context(|| format!("failed to create {model_name} ONNX Runtime session"))?;
        builder = builder.with_intra_threads(4).map_err(|error| {
            anyhow!("failed to configure {model_name} intra-op threads: {error}")
        })?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                anyhow!("failed to configure {model_name} graph optimizations: {error}")
            })?;
        let session = builder.commit_from_file(model_path).with_context(|| {
            format!("failed to load {model_name} model {}", model_path.display())
        })?;

        let actual_inputs: Vec<_> = session
            .inputs()
            .iter()
            .map(|outlet| outlet.name())
            .collect();
        let actual_outputs: Vec<_> = session
            .outputs()
            .iter()
            .map(|outlet| outlet.name())
            .collect();
        validate_signature(
            model_name,
            &actual_inputs,
            &actual_outputs,
            expected_inputs,
            expected_outputs,
        )?;

        Ok(Self {
            session: Mutex::new(session),
            model_name,
        })
    }

    pub(crate) fn run<'i, 'v: 'i, const N: usize, T>(
        &self,
        inputs: impl Into<SessionInputs<'i, 'v, N>>,
        decode: impl FnOnce(&SessionOutputs<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut session = self.session.lock().map_err(|error| {
            anyhow!(
                "{} ONNX Runtime session lock is poisoned: {error}",
                self.model_name
            )
        })?;
        let outputs = session
            .run(inputs)
            .with_context(|| format!("{} ONNX Runtime inference failed", self.model_name))?;
        decode(&outputs)
            .with_context(|| format!("failed to decode {} ONNX outputs", self.model_name))
    }
}

/// Copy an ndarray 0.16 array into an ORT-owned tensor.
///
/// Iteration follows logical row-major index order, rather than the backing
/// allocation order, so transposed and otherwise non-standard owned arrays are
/// represented correctly. This bridge deliberately avoids exposing ORT's
/// ndarray 0.17 dependency in the crate's public API.
pub(crate) fn tensor<T, S, D>(array: &ArrayBase<S, D>) -> Result<Tensor<T>>
where
    T: PrimitiveTensorElementType + Copy + Debug + 'static,
    S: Data<Elem = T>,
    D: Dimension,
{
    let (shape, values) = tensor_parts(array);
    Tensor::from_array((shape, values)).context("failed to create ONNX input tensor")
}

fn tensor_parts<T, S, D>(array: &ArrayBase<S, D>) -> (Vec<usize>, Vec<T>)
where
    T: Copy,
    S: Data<Elem = T>,
    D: Dimension,
{
    (array.shape().to_vec(), array.iter().copied().collect())
}

/// Copy a borrowed ORT output into an owned ndarray 0.16 array.
pub(crate) fn extract<T, D>(outputs: &SessionOutputs<'_>, name: &str) -> Result<Array<T, D>>
where
    T: PrimitiveTensorElementType + Clone,
    D: Dimension,
{
    let value = outputs
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    let (shape, values) = value.try_extract_tensor::<T>()?;
    let dimensions: Vec<usize> = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension)
                .with_context(|| format!("{name} has invalid dimension {dimension}"))
        })
        .collect::<Result<_>>()?;
    Array::from_shape_vec(IxDyn(&dimensions), values.to_vec())
        .with_context(|| format!("invalid {name} output buffer for shape {dimensions:?}"))?
        .into_dimensionality::<D>()
        .map_err(|error| anyhow!("unexpected {name} shape: {error}"))
}

/// Native ONNX Runtime build information, for private tests and diagnostics.
#[cfg(test)]
pub(crate) fn native_runtime_build_info() -> &'static str {
    ort::info()
}

fn validate_signature(
    model_name: &str,
    actual_inputs: &[&str],
    actual_outputs: &[&str],
    expected_inputs: &[&str],
    expected_outputs: &[&str],
) -> Result<()> {
    let actual_inputs: BTreeSet<_> = actual_inputs.iter().copied().collect();
    let actual_outputs: BTreeSet<_> = actual_outputs.iter().copied().collect();
    let expected_inputs: BTreeSet<_> = expected_inputs.iter().copied().collect();
    let expected_outputs: BTreeSet<_> = expected_outputs.iter().copied().collect();

    let missing_inputs: Vec<_> = expected_inputs
        .difference(&actual_inputs)
        .copied()
        .collect();
    let extra_inputs: Vec<_> = actual_inputs
        .difference(&expected_inputs)
        .copied()
        .collect();
    if !missing_inputs.is_empty() || !extra_inputs.is_empty() {
        bail!(
            "{model_name} model input mismatch: missing {missing_inputs:?}, extra {extra_inputs:?}"
        );
    }

    let missing_outputs: Vec<_> = expected_outputs
        .difference(&actual_outputs)
        .copied()
        .collect();
    if !missing_outputs.is_empty() {
        bail!("{model_name} model outputs missing {missing_outputs:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::{native_runtime_build_info, tensor_parts, validate_signature};

    #[test]
    fn native_runtime_is_1_28() {
        let info = native_runtime_build_info();
        eprintln!("{info}");
        assert!(
            info.contains("rel-1.28") || info.contains("1.28"),
            "expected ONNX Runtime 1.28 build information, got: {info}"
        );
    }

    #[test]
    fn bridge_preserves_transposed_logical_order() {
        let transposed = arr2(&[[1_i64, 2, 3], [4, 5, 6]]).reversed_axes();
        assert!(!transposed.is_standard_layout());
        let (shape, values) = tensor_parts(&transposed);
        assert_eq!(shape, [3, 2]);
        assert_eq!(values, [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn rejects_missing_input() {
        let error =
            validate_signature("test", &["a"], &["out"], &["a", "b"], &["out"]).unwrap_err();
        assert!(error.to_string().contains("missing [\"b\"]"));
    }

    #[test]
    fn rejects_extra_input() {
        let error =
            validate_signature("test", &["a", "b"], &["out"], &["a"], &["out"]).unwrap_err();
        assert!(error.to_string().contains("extra [\"b\"]"));
    }

    #[test]
    fn rejects_missing_output() {
        let error = validate_signature("test", &["a"], &["other"], &["a"], &["out"]).unwrap_err();
        assert!(error.to_string().contains("outputs missing [\"out\"]"));
    }

    #[test]
    fn tolerates_extra_output() {
        validate_signature("test", &["a"], &["out", "debug"], &["a"], &["out"]).unwrap();
    }
}
