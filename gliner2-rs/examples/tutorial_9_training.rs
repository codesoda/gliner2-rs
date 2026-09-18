use std::collections::BTreeMap;

use gliner2_rs::{Result, training::{InputExample, Trainer, TrainingConfig}};

fn main() -> Result<()> {
    // Mirrors tutorial/9-training.md "Minimal Example".
    let examples = vec![
        InputExample {
            text: "John works at Google in California.".to_string(),
            entities: BTreeMap::from([
                ("person".to_string(), vec!["John".to_string()]),
                ("company".to_string(), vec!["Google".to_string()]),
                ("location".to_string(), vec!["California".to_string()]),
            ]),
        },
        InputExample {
            text: "Apple released iPhone 15.".to_string(),
            entities: BTreeMap::from([
                ("company".to_string(), vec!["Apple".to_string()]),
                ("product".to_string(), vec!["iPhone 15".to_string()]),
            ]),
        },
    ];

    let config = TrainingConfig {
        output_dir: "./output".into(),
        num_epochs: 10,
        batch_size: 8,
        encoder_lr: 1e-5,
        task_lr: 5e-4,
    };

    // TODO: Training is not implemented in Rust yet.
    Trainer::train(examples, config)?;
    Ok(())
}

