use std::path::PathBuf;

use ndarray::{Array1, Array2};

use gliner2_rs::{Result, extractor::Extractor, spans::build_spans};

fn model_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn extractor_runs_on_dummy_inputs() -> Result<()> {
    let root = model_root();
    let onnx_path = root.join("onnx/gliner2-base-v1/extractor_padded.onnx");

    if !onnx_path.exists() {
        eprintln!("SKIP: missing {}", onnx_path.display());
        return Ok(());
    }

    let hidden = 768usize;
    let max_width = 8usize;
    let text_len = 6usize;
    // Must match export/export_extractor_padded.py.
    let max_fields = 64usize;

    let text_emb = Array2::<f32>::zeros((text_len, hidden));
    let schema_emb_padded = Array2::<f32>::zeros((1 + max_fields, hidden));
    let schema_mask = Array1::<bool>::from_elem(max_fields, true);
    let spans_idx = build_spans(text_len, max_width);

    let extractor = Extractor::new(&onnx_path)?;
    let out = extractor.infer(text_emb, schema_emb_padded, schema_mask, spans_idx)?;

    assert_eq!(out.count_logits.shape(), &[1, 20]);
    assert_eq!(out.span_scores.shape(), &[20, max_fields, text_len, max_width]);

    Ok(())
}
