use std::time::Instant;

use gliner2_rs::{
    Result,
    schema_spec::{EntityOptions, EntitySpec, FieldDtype, SchemaBuilder},
};
mod common;
use common::{load_auto_pipeline, model_paths_from_args};

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2-base-v1");

    let load_start = Instant::now();
    let pipeline = load_auto_pipeline(&paths, false)?;
    println!(
        "model load took: {:.2?} (onnx={})",
        load_start.elapsed(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // Mirrors `tutorial/2-ner.md`.

    // --- Basic Entity Extraction ---
    let text = "Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023.";
    let labels = vec![
        "company".to_string(),
        "person".to_string(),
        "product".to_string(),
        "location".to_string(),
        "date".to_string(),
    ];

    let start = Instant::now();
    let out = pipeline.extract_entities_text(text, labels.clone(), 0.5, false, false)?;
    println!("text: {text}");
    println!("entities: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Using Schema Builder ---
    let schema = SchemaBuilder::new().entities(labels).build();
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("schema.entities + extract(): {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Entity Extraction with Descriptions ---
    let schema = SchemaBuilder::new()
        .entities(vec![
            (
                "drug".to_string(),
                "Pharmaceutical drugs, medications, or treatment names".to_string(),
            ),
            (
                "disease".to_string(),
                "Medical conditions, illnesses, or disorders".to_string(),
            ),
            (
                "symptom".to_string(),
                "Clinical symptoms or patient-reported symptoms".to_string(),
            ),
            (
                "dosage".to_string(),
                "Medication amounts like '50mg' or '2 tablets daily'".to_string(),
            ),
            (
                "organ".to_string(),
                "Body parts or organs mentioned in medical context".to_string(),
            ),
        ])
        .build();

    let medical_text = r#"
Patient was prescribed Metformin 500mg twice daily for Type 2 Diabetes. 
She reported fatigue and occasional dizziness. Liver function tests ordered.
"#;
    let start = Instant::now();
    let out = pipeline.extract(medical_text, &schema, 0.5)?;
    println!("medical_text: {medical_text}");
    println!("entities (descriptions): {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Single vs Multiple Entities ---
    // Multiple entities per type (default).
    let schema = SchemaBuilder::new()
        .entities_with_options(
            vec!["person".to_string(), "organization".to_string()],
            EntityOptions {
                dtype: FieldDtype::List,
                threshold: None,
            },
        )
        .build();
    let text = "Bill Gates and Steve Jobs founded Microsoft and Apple respectively.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("dtype=list: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Single entity per type.
    let schema = SchemaBuilder::new()
        .entities_with_options(
            vec!["company".to_string(), "ceo".to_string()],
            EntityOptions {
                dtype: FieldDtype::Str,
                threshold: None,
            },
        )
        .build();
    let text = "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("dtype=str: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Custom Thresholds ---
    // Global threshold (quick API).
    let text = "Contact John Doe at john.doe@email.com or call 555-1234.";
    let start = Instant::now();
    let out = pipeline.extract_entities_text(
        text,
        vec![
            "email".to_string(),
            "phone".to_string(),
            "address".to_string(),
        ],
        0.8,
        false,
        false,
    )?;
    println!("text: {text}");
    println!("threshold=0.8: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // With confidence scores.
    let text = "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino.";
    let start = Instant::now();
    let out = pipeline.extract_entities_text(
        text,
        vec![
            "company".to_string(),
            "person".to_string(),
            "product".to_string(),
        ],
        0.5,
        true,
        false,
    )?;
    println!("text: {text}");
    println!("include_confidence: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // With spans (character offsets).
    let start = Instant::now();
    let out = pipeline.extract_entities_text(
        text,
        vec!["company".to_string(), "person".to_string()],
        0.5,
        false,
        true,
    )?;
    println!("include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // With both confidence and spans.
    let start = Instant::now();
    let out = pipeline.extract_entities_text(
        text,
        vec!["company".to_string(), "product".to_string()],
        0.5,
        true,
        true,
    )?;
    println!("include_confidence+include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Per-Entity Thresholds ---
    let schema = SchemaBuilder::new()
        .entities(vec![
            EntitySpec::new("email")
                .description("Email addresses")
                .dtype(FieldDtype::List)
                .threshold(0.9),
            EntitySpec::new("phone")
                .description("Phone numbers including mobile and landline")
                .dtype(FieldDtype::List)
                .threshold(0.7),
            EntitySpec::new("name")
                .description("Person names")
                .dtype(FieldDtype::List)
                .threshold(0.5),
        ])
        .build();
    let contact_text = "Contact John Doe at john.doe@email.com or call 555-1234.";
    let start = Instant::now();
    let out = pipeline.extract(contact_text, &schema, 0.6)?;
    println!("contact_text: {contact_text}");
    println!("per-entity thresholds (default=0.6): {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Advanced Configuration ---
    // Mixed configuration.
    let schema = SchemaBuilder::new()
        .entities(vec![
            "date".to_string(),
            "time".to_string(),
            "currency".to_string(),
        ])
        .entities(vec![
            (
                "technical_term".to_string(),
                "Technical jargon or specialized terminology".to_string(),
            ),
            (
                "metric".to_string(),
                "Measurements, KPIs, or quantitative values".to_string(),
            ),
        ])
        .entities(vec![
            EntitySpec::new("competitor")
                .description("Competing companies or products")
                .dtype(FieldDtype::List)
                .threshold(0.7),
            EntitySpec::new("revenue")
                .description("Revenue figures or financial amounts")
                .dtype(FieldDtype::Str)
                .threshold(0.8),
        ])
        .build();

    let text =
        "In Q3 2024, Acme reported revenue of $3.2M and beat BetaCorp on key performance metrics.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("mixed_config: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Incremental entity addition.
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "location".to_string()])
        .entities(vec![(
            "company".to_string(),
            "Company or organization names".to_string(),
        )])
        .entities(vec![
            EntitySpec::new("financial_term")
                .description("Financial instruments, metrics, or terminology")
                .threshold(0.75),
        ])
        .build();

    let text = "Alice moved from Paris to London to join Acme and discuss P/E ratios.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("incremental_addition: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Domain-Specific Entities ---
    let legal_schema = SchemaBuilder::new()
        .entities(vec![
            (
                "party".to_string(),
                "Parties involved in legal proceedings (plaintiff, defendant, etc.)".to_string(),
            ),
            (
                "law_firm".to_string(),
                "Law firm or legal practice names".to_string(),
            ),
            (
                "court".to_string(),
                "Court names or judicial bodies".to_string(),
            ),
            (
                "statute".to_string(),
                "Legal statutes, laws, or regulations cited".to_string(),
            ),
            (
                "case".to_string(),
                "Legal case names or citations".to_string(),
            ),
            (
                "judge".to_string(),
                "Names of judges or magistrates".to_string(),
            ),
            (
                "legal_term".to_string(),
                "Legal terminology or concepts".to_string(),
            ),
        ])
        .build();

    let legal_text = r#"
In the case of Smith v. Jones, Judge Sarah Williams of the Superior Court 
ruled that the defendant violated Section 15.2 of the Consumer Protection Act.
The plaintiff was represented by Miller & Associates.
"#;
    let start = Instant::now();
    let out = pipeline.extract(legal_text, &legal_schema, 0.5)?;
    println!("legal_text: {legal_text}");
    println!("legal_entities: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let finance_schema = SchemaBuilder::new()
        .entities(vec![
            (
                "ticker".to_string(),
                "Stock ticker symbols (e.g., AAPL, GOOGL)".to_string(),
            ),
            (
                "financial_metric".to_string(),
                "Financial metrics like P/E ratio, market cap".to_string(),
            ),
            (
                "currency_amount".to_string(),
                "Monetary values with currency symbols".to_string(),
            ),
            (
                "percentage".to_string(),
                "Percentage values (e.g., 5.2%, -3%)".to_string(),
            ),
            (
                "financial_org".to_string(),
                "Banks, investment firms, financial institutions".to_string(),
            ),
            (
                "market_index".to_string(),
                "Stock market indices (S&P 500, NASDAQ, etc.)".to_string(),
            ),
        ])
        .build();

    let finance_text = r#"
AAPL rose 3.5% to $185.50 after beating earnings expectations. 
The company's P/E ratio of 28.5 attracted Goldman Sachs analysts. 
The NASDAQ composite gained 1.2% for the day.
"#;
    let start = Instant::now();
    let out = pipeline.extract(finance_text, &finance_schema, 0.5)?;
    println!("finance_text: {finance_text}");
    println!("finance_entities: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let science_schema = SchemaBuilder::new()
        .entities(vec![
            (
                "chemical".to_string(),
                "Chemical compounds or elements".to_string(),
            ),
            (
                "organism".to_string(),
                "Biological organisms, species names".to_string(),
            ),
            ("gene".to_string(), "Gene names or identifiers".to_string()),
            (
                "measurement".to_string(),
                "Scientific measurements with units".to_string(),
            ),
            (
                "research_method".to_string(),
                "Research techniques or methodologies".to_string(),
            ),
            (
                "institution".to_string(),
                "Universities or research institutions".to_string(),
            ),
        ])
        .build();

    let science_text = r#"
Researchers at MIT discovered that the BRCA1 gene mutation increases 
cancer risk by 70%. The study used CRISPR-Cas9 to modify DNA sequences
in Mus musculus specimens, measuring tumor growth in millimeters.
"#;
    let start = Instant::now();
    let out = pipeline.extract(science_text, &science_schema, 0.5)?;
    println!("science_text: {science_text}");
    println!("science_entities: {:#?}", out.entities);
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
