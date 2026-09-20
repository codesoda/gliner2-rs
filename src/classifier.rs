use std::path::Path;

use ndarray::{Array1, Array2, Ix1};

use crate::Result;
use crate::runtime::{RuntimeSession, extract, tensor};

const INPUTS: &[&str] = &["cls_embeds"];
const OUTPUTS: &[&str] = &["logits"];

pub struct Classifier {
    session: RuntimeSession,
}

impl Classifier {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: RuntimeSession::load(model_path, "classifier", INPUTS, OUTPUTS)?,
        })
    }

    pub fn infer(&self, cls_embeds: Array2<f32>) -> Result<Array1<f32>> {
        let inputs = ort::inputs! {
            "cls_embeds" => tensor(&cls_embeds)?,
        };
        self.session
            .run(inputs, |outputs| extract::<f32, Ix1>(outputs, "logits"))
    }
}
