use regex::{Regex, RegexBuilder};

// TODO(regex): consider supporting full Python `re` parity (look-around, more flags) by making the
// regex engine pluggable (e.g. optional `fancy-regex` feature). Keep `regex` as the default since
// it's fast and guaranteed linear-time (no catastrophic backtracking / ReDoS).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexMode {
    Full,
    Partial,
}

#[derive(Debug, Clone)]
pub struct RegexValidator {
    pub pattern: String,
    pub mode: RegexMode,
    pub exclude: bool,
    pub case_insensitive: bool,
    compiled: Regex,
}

impl RegexValidator {
    pub fn new(pattern: impl Into<String>) -> Self {
        let pattern = pattern.into();
        let compiled = RegexBuilder::new(&pattern)
            .case_insensitive(true)
            .build()
            .unwrap_or_else(|e| panic!("invalid regex '{pattern}': {e}"));
        Self {
            pattern,
            mode: RegexMode::Full,
            exclude: false,
            case_insensitive: true,
            compiled,
        }
    }

    pub fn mode(mut self, mode: RegexMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn exclude(mut self, exclude: bool) -> Self {
        self.exclude = exclude;
        self
    }

    pub fn case_insensitive(mut self, case_insensitive: bool) -> Self {
        if self.case_insensitive != case_insensitive {
            self.case_insensitive = case_insensitive;
            self.compiled = RegexBuilder::new(&self.pattern)
                .case_insensitive(case_insensitive)
                .build()
                .unwrap_or_else(|e| panic!("invalid regex '{}': {e}", self.pattern));
        }
        self
    }

    pub fn validate(&self, text: &str) -> bool {
        let matched = match self.mode {
            RegexMode::Full => self
                .compiled
                .find_iter(text)
                .any(|m| m.start() == 0 && m.end() == text.len()),
            RegexMode::Partial => self.compiled.is_match(text),
        };

        if self.exclude { !matched } else { matched }
    }
}

impl PartialEq for RegexValidator {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.mode == other.mode
            && self.exclude == other.exclude
            && self.case_insensitive == other.case_insensitive
    }
}

impl Eq for RegexValidator {}
