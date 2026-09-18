//! Fetch the ONNX model exports from Hugging Face into ./onnx.
//!
//! This is a manual, opt-in step — nothing else in this crate calls it
//! automatically. Run it explicitly when you need the models:
//!
//! ```bash
//! cargo run --example download_models -- --model all
//! cargo run --example download_models -- --model base
//! ```

use anyhow::Result;
use hf_hub::HFClientSync;

mod common;
use common::repo_root;

const REPO_ID_OWNER: &str = "codesoda";
const REPO_ID_NAME: &str = "gliner2-onnx";

fn model_prefix(name: &str) -> Result<&'static str> {
    match name {
        "base" => Ok("gliner2-base-v1"),
        "large" => Ok("gliner2-large-v1"),
        other => anyhow::bail!("unknown --model value '{other}' (expected base|large|all)"),
    }
}

fn main() -> Result<()> {
    let mut model = "all".to_string();
    let mut dest = repo_root().join("onnx");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = args.next().ok_or_else(|| anyhow::anyhow!("--model needs a value"))?;
            }
            "--dest" => {
                dest = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--dest needs a value"))?
                    .into();
            }
            other => anyhow::bail!("unrecognized argument: {other}"),
        }
    }

    let prefixes: Vec<&str> = if model == "all" {
        vec!["gliner2-base-v1", "gliner2-large-v1"]
    } else {
        vec![model_prefix(&model)?]
    };

    std::fs::create_dir_all(&dest)?;

    let client = HFClientSync::new()?;
    let repo = client.model(REPO_ID_OWNER, REPO_ID_NAME);

    for prefix in prefixes {
        println!("Downloading {prefix} from {REPO_ID_OWNER}/{REPO_ID_NAME} ...");
        let pattern = format!("{prefix}/*");
        let out = repo
            .snapshot_download()
            .allow_patterns(vec![pattern])
            .local_dir(dest.clone())
            .send()?;
        println!("  -> {}", out.display());
    }

    println!("Done. Models available under {}", dest.display());
    Ok(())
}
