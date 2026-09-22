use std::env;

use anyhow::Context;

use crate::Result;

#[derive(Debug, Clone)]
pub struct ApiExtractor {
    pub base_url: String,
    pub api_key: String,
}

impl ApiExtractor {
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("PIONEER_API_KEY")
            .context("missing PIONEER_API_KEY environment variable (see tutorial #7)")?;
        let base_url = env::var("GLINER2_API_BASE_URL")
            .unwrap_or_else(|_| "https://gliner.pioneer.ai".to_string());

        Ok(Self { base_url, api_key })
    }
}
