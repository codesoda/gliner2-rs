//! Complete classification distribution with explicit runtime options.
//!
//! ```bash
//! cargo run --release --example classification_scores -- onnx/gliner2.5-base-v1
//! ```
//!
//! Loads only the tokenizer, encoder and classifier, scores one text against
//! two tasks in one encoder pass, and prints every label's raw logit and
//! probability. Nothing is filtered by a threshold.
//!
//! Do not treat sigmoid outputs as a categorical distribution: with
//! `Activation::Sigmoid` each label is an independent probability and the
//! values do not sum to one. Use `Activation::Softmax` when you need one
//! distribution over the labels.

use std::env;

use anyhow::{Context, Result};
use gliner2_rs::{Activation, ClassificationPipeline, ClassificationRequest, RuntimeOptions};

fn main() -> Result<()> {
    let bundle = env::args()
        .nth(1)
        .context("usage: classification_scores <bundle-dir>")?;
    let options = RuntimeOptions::default().with_intra_threads(2);
    let pipeline = ClassificationPipeline::from_dir_with_options(&bundle, options)?;
    let report = pipeline.runtime_report();
    eprintln!(
        "provider={} intra={} inter={:?} optimization={} sessions={:?} runtime={}",
        report.provider,
        report.intra_threads,
        report.inter_threads,
        report.optimization_level,
        report.sessions,
        report.native_runtime
    );

    let text = "I love this phone, but please tell me how to turn off the flashlight.";
    let requests = [
        ClassificationRequest::new(
            "sentiment",
            ["positive", "negative", "neutral"]
                .map(str::to_owned)
                .to_vec(),
            Activation::Softmax,
        ),
        ClassificationRequest::new(
            "intent",
            ["praise", "complaint", "how-to question"]
                .map(str::to_owned)
                .to_vec(),
            Activation::Softmax,
        )
        .with_instruction("What does the user want?"),
    ];
    let scores = pipeline.score_classifications(text, &requests)?;
    for task in &scores {
        println!(
            "{} ({}; temperature {}; {} input tokens{})",
            task.task(),
            task.activation(),
            task.temperature(),
            task.usage().input_tokens,
            if task.usage().truncated() {
                ", truncated"
            } else {
                ""
            }
        );
        for ((label, logit), probability) in task
            .labels()
            .iter()
            .zip(task.raw_logits())
            .zip(task.probabilities())
        {
            println!("  {label:18} logit {logit:+9.4}  p {probability:.6}");
        }
        println!("  argmax -> {}", task.labels()[task.argmax()]);
    }
    Ok(())
}
