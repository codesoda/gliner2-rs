use std::path::Path;

use ndarray::{Array2, Array3, Ix3};

use crate::Result;
use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &["input_ids", "attention_mask"];
const OUTPUTS: &[&str] = &["last_hidden_state"];

/// Thin wrapper around the encoder ONNX model.
pub struct Encoder {
    session: RuntimeSession,
}

impl Encoder {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load(model_path, "encoder", INPUTS, OUTPUTS)?,
        })
    }

    /// Run encoder with already-built tensors.
    pub fn infer(
        &self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
    ) -> Result<Array3<f32>> {
        let inputs = ort::inputs! {
            "input_ids" => tensor(&input_ids)?,
            "attention_mask" => tensor(&attention_mask)?,
        };
        self.session.run(inputs, |outputs| {
            extract::<f32, Ix3>(outputs, "last_hidden_state")
        })
    }
}
