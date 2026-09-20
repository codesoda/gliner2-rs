use std::{sync::Arc, thread};

use gliner2_rs::{Result, encoder::Encoder, tokenizer::RuntimeTokenizer};

mod common;
use common::{artifacts_available, model_root};

#[test]
fn encoder_runs_on_real_tokens() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let onnx_path = root.join("onnx/gliner2-base-v1/encoder.onnx");

    if !artifacts_available(&[&model_dir, &onnx_path])? {
        return Ok(());
    }

    let tokenizer = RuntimeTokenizer::from_dir(&model_dir)?;
    let (input_ids, attention_mask) = tokenizer.encode_text("Hello world!")?;

    let len = input_ids.shape()[1];

    let encoder = Encoder::new(&onnx_path)?;
    let hidden = encoder.infer(input_ids, attention_mask)?;

    assert_eq!(hidden.shape(), &[1, len, 768]);
    Ok(())
}

#[test]
fn shared_encoder_serializes_concurrent_inference() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");
    let onnx_path = root.join("onnx/gliner2-base-v1/encoder.onnx");

    if !artifacts_available(&[&model_dir, &onnx_path])? {
        return Ok(());
    }

    let tokenizer = RuntimeTokenizer::from_dir(&model_dir)?;
    let (input_ids, attention_mask) = tokenizer.encode_text("Concurrent inference")?;
    let encoder = Arc::new(Encoder::new(&onnx_path)?);
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let encoder = Arc::clone(&encoder);
            let input_ids = input_ids.clone();
            let attention_mask = attention_mask.clone();
            thread::spawn(move || encoder.infer(input_ids, attention_mask))
        })
        .collect();
    let outputs: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("encoder inference thread panicked"))
        .collect::<Result<_>>()?;

    assert_eq!(outputs[0], outputs[1]);
    Ok(())
}

#[test]
fn convert_known_subwords() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");

    if !artifacts_available(&[&model_dir])? {
        return Ok(());
    }

    let tokenizer = RuntimeTokenizer::from_dir(&model_dir)?;

    let (ids, mask) = tokenizer.convert_subwords(["[P]", "[SEP_TEXT]"].iter().copied())?;
    assert_eq!(ids.shape(), &[1, 2]);
    assert_eq!(mask.sum(), 2);
    Ok(())
}

#[test]
fn special_token_ids_match_expected() -> Result<()> {
    let root = model_root();
    let model_dir = root.join("models/gliner2-base-v1");

    if !artifacts_available(&[&model_dir])? {
        return Ok(());
    }

    let tokenizer = RuntimeTokenizer::from_dir(&model_dir)?;

    // Explicit IDs pulled from special_tokens_map.json/vocab.
    let expected = [
        ("[SEP_STRUCT]", 128001),
        ("[SEP_TEXT]", 128002),
        ("[P]", 128003),
        ("[C]", 128004),
        ("[E]", 128005),
        ("[R]", 128006),
        ("[L]", 128007),
        ("[EXAMPLE]", 128008),
        ("[OUTPUT]", 128009),
        ("[DESCRIPTION]", 128010),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("[MASK]", 128000),
        ("[PAD]", 0),
        ("[UNK]", 3),
    ];

    for (tok, id) in expected {
        let actual = tokenizer.id_for(tok).expect("token missing");
        assert_eq!(actual, id, "token {} id mismatch", tok);
    }

    Ok(())
}
