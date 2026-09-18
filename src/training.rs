use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use crate::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct InputExample {
    pub text: String,
    pub entities: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub output_dir: PathBuf,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub encoder_lr: f32,
    pub task_lr: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./output"),
            num_epochs: 1,
            batch_size: 8,
            encoder_lr: 1e-5,
            task_lr: 5e-4,
        }
    }
}

pub struct Trainer;

impl Trainer {
    pub fn train(_train_data: Vec<InputExample>, _config: TrainingConfig) -> Result<()> {
        todo!("Tutorial #9 training is not implemented in Rust yet.")
    }
}

pub fn load_jsonl(_path: impl AsRef<Path>) -> Result<Vec<InputExample>> {
    todo!("Tutorial #8 JSONL dataset loading is not implemented in Rust yet.")
}
