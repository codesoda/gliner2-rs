use std::time::Instant;

use gliner2_rs::{Result, api::ApiExtractor};

fn main() -> Result<()> {
    // Mirrors tutorial/7-api.md "Getting Started".
    //
    // NOTE: This only loads credentials/config (no network calls yet).
    // TODO(tutorial-7): implement an HTTP client + auth + response parsing for actual API usage.
    let start = Instant::now();
    let extractor = ApiExtractor::from_env()?;
    println!("api extractor init took: {:.2?}", start.elapsed());
    println!("base_url: {}", extractor.base_url);
    println!("-----------------");
    Ok(())
}
