use gliner2_rs::{Result, training::load_jsonl};

fn main() -> Result<()> {
    // Mirrors tutorial/8-train_data.md "Quick Start with JSONL".
    //
    // TODO: Implement JSONL parsing and mapping to InputExample records.
    let _examples = load_jsonl("train.jsonl")?;
    Ok(())
}

