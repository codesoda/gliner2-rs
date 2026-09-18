use std::{fs, path::Path};

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterConfig {
    pub adapter_dir: std::path::PathBuf,
    pub encoder_onnx: std::path::PathBuf,
    pub lora_r: Option<usize>,
}

pub fn read_lora_r(adapter_dir: &Path) -> Option<usize> {
    let config_path = adapter_dir.join("adapter_config.json");
    let contents = fs::read_to_string(config_path).ok()?;

    // Try the most explicit key first.
    if let Some(v) = extract_usize_json_field(&contents, "lora_r") {
        return Some(v);
    }

    None
}

fn extract_usize_json_field(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let idx = json.find(&needle)?;
    let after_key = &json[idx + needle.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();

    let digits: String = after_colon
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();

    if digits.is_empty() {
        return None;
    }
    digits.parse::<usize>().ok()
}
