//! Explicit ONNX Runtime construction options and effective-runtime reports.
//!
//! Every session a pipeline opens is built from one [`RuntimeOptions`] value.
//! The defaults reproduce the crate's historical fixed settings (CPU, four
//! intra-op threads, full graph optimization), so existing constructors keep
//! their numerical baseline. Options are validated once, before any model file
//! is opened, and never read from the environment or from global state.
//!
//! Only the CPU execution provider is supported by this build. Other providers
//! can be requested so that a caller gets a clear, contextual rejection instead
//! of a silent fallback to CPU.

use std::{error::Error, fmt};

/// Execution provider requested for all sessions of one pipeline instance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ExecutionProvider {
    /// ONNX Runtime CPU provider. The only provider supported by this crate.
    #[default]
    Cpu,
    /// Apple CoreML. Requested for diagnostics only; rejected at validation.
    CoreMl,
    /// NVIDIA CUDA. Requested for diagnostics only; rejected at validation.
    Cuda,
}

impl ExecutionProvider {
    /// Providers this build can actually execute on.
    pub const SUPPORTED: &'static [ExecutionProvider] = &[ExecutionProvider::Cpu];

    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::Cuda => "cuda",
        }
    }
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// ONNX Runtime graph optimization level.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum OptimizationLevel {
    Disable,
    Basic,
    Extended,
    #[default]
    All,
}

impl OptimizationLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disable => "disable",
            Self::Basic => "basic",
            Self::Extended => "extended",
            Self::All => "all",
        }
    }
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Typed session construction options.
///
/// Construct with [`RuntimeOptions::default`] and adjust with the `with_*`
/// methods. Thread counts of zero and unsupported providers are rejected by
/// [`RuntimeOptions::validate`], which every loader calls first.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RuntimeOptions {
    provider: ExecutionProvider,
    intra_threads: usize,
    inter_threads: Option<usize>,
    optimization_level: OptimizationLevel,
}

/// Historical fixed intra-op thread count used by every legacy constructor.
pub const DEFAULT_INTRA_THREADS: usize = 4;

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            provider: ExecutionProvider::Cpu,
            intra_threads: DEFAULT_INTRA_THREADS,
            inter_threads: None,
            optimization_level: OptimizationLevel::All,
        }
    }
}

impl RuntimeOptions {
    pub fn with_provider(mut self, provider: ExecutionProvider) -> Self {
        self.provider = provider;
        self
    }

    /// Threads used inside one operator. Must be positive.
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Threads used across independent operators. `None` keeps the ONNX
    /// Runtime default (sequential execution). Must be positive when set.
    pub fn with_inter_threads(mut self, threads: Option<usize>) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    pub const fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    pub const fn intra_threads(&self) -> usize {
        self.intra_threads
    }

    pub const fn inter_threads(&self) -> Option<usize> {
        self.inter_threads
    }

    pub const fn optimization_level(&self) -> OptimizationLevel {
        self.optimization_level
    }

    /// Reject options this build cannot honour, before any file is opened.
    pub fn validate(&self) -> Result<(), RuntimeOptionsError> {
        if !self.provider.is_supported() {
            return Err(RuntimeOptionsError::UnsupportedProvider(self.provider));
        }
        if self.intra_threads == 0 {
            return Err(RuntimeOptionsError::ZeroThreads("intra"));
        }
        if self.inter_threads == Some(0) {
            return Err(RuntimeOptionsError::ZeroThreads("inter"));
        }
        Ok(())
    }
}

/// Validation failures for [`RuntimeOptions`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeOptionsError {
    UnsupportedProvider(ExecutionProvider),
    ZeroThreads(&'static str),
}

impl fmt::Display for RuntimeOptionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProvider(provider) => write!(
                formatter,
                "execution provider {provider} is not supported by this gliner2-rs build; supported providers: {}",
                ExecutionProvider::SUPPORTED
                    .iter()
                    .map(|provider| provider.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ZeroThreads(kind) => {
                write!(formatter, "{kind}-op thread count must be positive")
            }
        }
    }
}

impl Error for RuntimeOptionsError {}

/// What a loaded pipeline actually runs with.
///
/// `provider` is the provider registered on every session. With the CPU
/// provider there is no node-level fallback to report; when other providers
/// become supported this report must distinguish registration from placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeReport {
    pub provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: Option<usize>,
    pub optimization_level: OptimizationLevel,
    /// Native ONNX Runtime build string, as reported by the linked library.
    pub native_runtime: String,
    /// Model names of the sessions this instance opened, in load order.
    pub sessions: Vec<&'static str>,
}

impl RuntimeReport {
    /// The options that produced this report.
    pub fn into_options(self) -> RuntimeOptions {
        RuntimeOptions {
            provider: self.provider,
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads,
            optimization_level: self.optimization_level,
        }
    }

    pub(crate) fn new(options: RuntimeOptions, sessions: Vec<&'static str>) -> Self {
        Self {
            provider: options.provider,
            intra_threads: options.intra_threads,
            inter_threads: options.inter_threads,
            optimization_level: options.optimization_level,
            native_runtime: ort::info().to_owned(),
            sessions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_legacy_fixed_settings() {
        let options = RuntimeOptions::default();
        assert_eq!(options.provider(), ExecutionProvider::Cpu);
        assert_eq!(options.intra_threads(), 4);
        assert_eq!(options.inter_threads(), None);
        assert_eq!(options.optimization_level(), OptimizationLevel::All);
        options.validate().unwrap();
    }

    #[test]
    fn rejects_zero_threads() {
        let error = RuntimeOptions::default()
            .with_intra_threads(0)
            .validate()
            .unwrap_err();
        assert_eq!(error, RuntimeOptionsError::ZeroThreads("intra"));
        let error = RuntimeOptions::default()
            .with_inter_threads(Some(0))
            .validate()
            .unwrap_err();
        assert_eq!(error, RuntimeOptionsError::ZeroThreads("inter"));
        RuntimeOptions::default()
            .with_inter_threads(Some(2))
            .validate()
            .unwrap();
    }

    #[test]
    fn rejects_unsupported_providers_with_supported_list() {
        for provider in [ExecutionProvider::CoreMl, ExecutionProvider::Cuda] {
            let error = RuntimeOptions::default()
                .with_provider(provider)
                .validate()
                .unwrap_err();
            assert_eq!(error, RuntimeOptionsError::UnsupportedProvider(provider));
            let message = error.to_string();
            assert!(message.contains(provider.as_str()), "{message}");
            assert!(message.contains("supported providers: cpu"), "{message}");
        }
        assert_eq!(ExecutionProvider::SUPPORTED, [ExecutionProvider::Cpu]);
    }

    #[test]
    fn report_copies_options() {
        let options = RuntimeOptions::default()
            .with_intra_threads(2)
            .with_optimization_level(OptimizationLevel::Basic);
        let report = RuntimeReport::new(options, vec!["encoder", "classifier"]);
        assert_eq!(report.intra_threads, 2);
        assert_eq!(report.optimization_level, OptimizationLevel::Basic);
        assert_eq!(report.sessions, ["encoder", "classifier"]);
        assert!(!report.native_runtime.is_empty());
    }
}
