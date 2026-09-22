//! Examples mirrored from the `fastino/gliner2-large-v1` Hugging Face model card:
//! https://huggingface.co/fastino/gliner2-large-v1
//!
//! By default this example runs with the local base model (`models/gliner2-base-v1`).
//! To run it with the large model, download the HF repo into `models/gliner2-large-v1`
//! and export ONNX into `onnx/gliner2-large-v1` (encoder/extractor_padded/classifier).
//! You can also pass `--model onnx/gliner2-large-v1` (or any other ONNX bundle).

use std::{collections::BTreeMap, time::Instant};

use gliner2_rs::{
    Result,
    json::JsonSchema,
    pipeline::Gliner2Pipeline,
    schema_spec::{
        ClassificationOptions, FieldDtype, QuickClassificationTask, SchemaBuilder,
        StructureFieldSpec,
    },
};
mod common;
use common::{model_paths_from_args, repo_root};

fn main() -> Result<()> {
    // If an explicit --model is provided, honor it; otherwise prefer the large model if present.
    let mut paths = model_paths_from_args("onnx/gliner2-base-v1");
    let root = repo_root();
    let defaulted = !std::env::args().any(|a| a.starts_with("--model"));
    if defaulted {
        let large_onnx = root.join("onnx/gliner2-large-v1");
        let large_model = root.join("models/gliner2-large-v1");
        if large_onnx.exists() && large_model.exists() {
            paths = model_paths_from_args("onnx/gliner2-large-v1");
        }
    }

    let encoder_onnx = paths.onnx_dir.join("encoder.onnx");
    let extractor_onnx = paths.onnx_dir.join("extractor_padded.onnx");
    let classifier_onnx = paths.onnx_dir.join("classifier.onnx");

    let load_start = Instant::now();
    let pipeline = if classifier_onnx.exists() {
        Gliner2Pipeline::new(&paths.model_dir, &encoder_onnx, &extractor_onnx)?
            .with_classifier(&classifier_onnx)?
    } else {
        println!(
            "warning: classifier not found at {}; skipping classification examples",
            classifier_onnx.display()
        );
        Gliner2Pipeline::new(&paths.model_dir, &encoder_onnx, &extractor_onnx)?
    };
    println!(
        "model load took: {:.2?} (model_dir={}, onnx_dir={})",
        load_start.elapsed(),
        paths.model_dir.display(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // --- Entity Extraction ---
    let mut entity_desc: BTreeMap<String, String> = BTreeMap::new();
    entity_desc.insert(
        "medication".to_string(),
        "Names of drugs, medications, or pharmaceutical substances".to_string(),
    );
    entity_desc.insert(
        "dosage".to_string(),
        "Specific amounts like '400mg', '2 tablets', or '5ml'".to_string(),
    );
    entity_desc.insert(
        "symptom".to_string(),
        "Medical symptoms, conditions, or patient complaints".to_string(),
    );
    entity_desc.insert(
        "time".to_string(),
        "Time references like '2 PM', 'morning', or 'after lunch'".to_string(),
    );

    let text = "Patient received 400mg ibuprofen for severe headache at 2 PM.";
    let start = Instant::now();
    let entities = pipeline.extract_entities_text(text, entity_desc, 0.5, false, false)?;
    println!("text: {text}");
    println!("entities: {entities:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Text Classification ---
    if classifier_onnx.exists() {
        // Single-label classification.
        let tasks: BTreeMap<String, QuickClassificationTask> = BTreeMap::from([(
            "sentiment".to_string(),
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ]
            .into(),
        )]);

        let text = "This laptop has amazing performance but terrible battery life!";
        let start = Instant::now();
        let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
        println!("text: {text}");
        println!("sentiment: {out:#?}");
        println!("inference took: {:.2?}", start.elapsed());
        println!("-----------------");

        // Multi-label classification.
        let tasks: BTreeMap<String, QuickClassificationTask> = BTreeMap::from([(
            "aspects".to_string(),
            QuickClassificationTask::config(
                vec![
                    "camera".to_string(),
                    "performance".to_string(),
                    "battery".to_string(),
                    "display".to_string(),
                    "price".to_string(),
                ],
                ClassificationOptions {
                    multi_label: true,
                    cls_threshold: 0.4,
                    ..Default::default()
                },
            ),
        )]);

        let text = "Great camera quality, decent performance, but poor battery life.";
        let start = Instant::now();
        let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
        println!("text: {text}");
        println!("aspects: {out:#?}");
        println!("inference took: {:.2?}", start.elapsed());
        println!("-----------------");
    }

    // --- Structured Data Extraction ---
    let text = r#"
Transaction Report: Goldman Sachs processed a $2.5M equity trade for Tesla Inc.
on March 15, 2024. Commission: $1,250. Status: Completed.
"#;

    let schema = JsonSchema::new().structure(
        "transaction",
        vec![
            "broker::str::Financial institution or brokerage firm".to_string(),
            "amount::str::Transaction amount with currency".to_string(),
            "security::str::Stock, bond, or financial instrument".to_string(),
            "date::str::Transaction date".to_string(),
            "commission::str::Fees or commission charged".to_string(),
            "status::str::Transaction status".to_string(),
            "type::[equity|bond|option|future|forex]::str::Type of financial instrument"
                .to_string(),
        ],
    );

    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("extract_json: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multi-Task Schema Composition ---
    let contract_text = r#"
Service Agreement between TechCorp LLC and DataSystems Inc., effective January 1, 2024.
Monthly fee: $15,000. Contract term: 24 months with automatic renewal.
Termination clause: 30-day written notice required.
"#;

    let mut builder = SchemaBuilder::new().entities(vec![
        "company".to_string(),
        "date".to_string(),
        "duration".to_string(),
        "fee".to_string(),
    ]);

    if classifier_onnx.exists() {
        builder = builder.classification(
            "contract_type",
            vec![
                "service".to_string(),
                "employment".to_string(),
                "nda".to_string(),
                "partnership".to_string(),
            ],
        );
    }

    let schema = builder
        .structure("contract_terms")
        .field(StructureFieldSpec::new("parties").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("effective_date").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("monthly_fee").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("term_length").dtype(FieldDtype::Str))
        .field(
            StructureFieldSpec::new("renewal")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "automatic".to_string(),
                    "manual".to_string(),
                    "none".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("termination_notice").dtype(FieldDtype::Str))
        .finish()
        .build();

    let start = Instant::now();
    let out = pipeline.extract(contract_text, &schema, 0.5)?;
    println!("text: {contract_text}");
    println!("extract: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    Ok(())
}
