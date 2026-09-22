use anyhow::Result;
use gliner2_rs::encoder::Encoder;
use gliner2_rs::extractor::Extractor;
use ndarray::{Array1, Array2};

mod common;
use common::model_paths_from_args;

fn build_spans(text_len: usize, max_width: usize) -> Array2<i64> {
    let num_spans = text_len * max_width;
    let mut spans = Array2::<i64>::zeros((num_spans, 2));
    let mut idx = 0;
    for start in 0..text_len {
        for width in 0..max_width {
            if start + width < text_len {
                spans[[idx, 0]] = start as i64;
                spans[[idx, 1]] = (start + width) as i64;
            } else {
                spans[[idx, 0]] = -1;
                spans[[idx, 1]] = -1;
            }
            idx += 1;
        }
    }
    spans
}

fn main() -> Result<()> {
    let paths = model_paths_from_args("../onnx/gliner2-base-v1");

    let encoder = Encoder::new(paths.encoder)?;
    let encoder_output = encoder.infer(Array2::ones((2, 8)), Array2::ones((2, 8)))?;
    println!(
        "encoder last_hidden_state shape: {:?}",
        encoder_output.shape()
    );

    let hidden_size = 768;
    let text_len = 6;
    let max_fields = 64;
    let extractor = Extractor::new(paths.extractor)?;
    let extractor_output = extractor.infer(
        Array2::zeros((text_len, hidden_size)),
        Array2::zeros((1 + max_fields, hidden_size)),
        Array1::from_elem(max_fields, true),
        build_spans(text_len, 8),
    )?;
    println!(
        "extractor count_logits shape: {:?}, span_scores shape: {:?}",
        extractor_output.count_logits.shape(),
        extractor_output.span_scores.shape()
    );

    Ok(())
}
