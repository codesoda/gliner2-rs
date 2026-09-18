use std::time::Instant;

use anyhow::anyhow;
use gliner2_rs::{
    Result, json::JsonSchema, pipeline::Gliner2Pipeline, schema_spec::{FieldDtype, SchemaBuilder, StructureFieldSpec}
};
mod common;
use common::model_paths_from_args;

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2-base-v1");
    let classifier_onnx = paths
        .classifier
        .as_ref()
        .ok_or_else(|| anyhow!("missing classifier.onnx in {}", paths.onnx_dir.display()))?;

    let load_start = Instant::now();
    let pipeline = Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?
        .with_classifier(classifier_onnx)?;
    println!(
        "model load took: {:.2?} (onnx={})",
        load_start.elapsed(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // Mirrors `tutorial/3-json_extraction.md`.

    // --- Quick API with extract_json ---
    // Basic Structure Extraction
    let text = "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.";
    let schema = JsonSchema::new().structure(
        "product",
        vec!["name::str".to_string(), "price".to_string(), "features".to_string()],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("product: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Contact Information
    let text = r#"
Contact: John Smith
Email: john@example.com
Phones: 555-1234, 555-5678
Address: 123 Main St, NYC
"#;
    let schema = JsonSchema::new().structure(
        "contact",
        vec![
            "name::str".to_string(),
            "email::str".to_string(),
            "phone::list".to_string(),
            "address".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("contact: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Field Types and Specifications ---
    // String vs List Fields
    let text = r#"
Tech Conference 2024 on June 15th in San Francisco. 
Topics include AI, Machine Learning, and Cloud Computing.
Registration fee: $299 for early bird tickets.
"#;
    let schema = JsonSchema::new().structure(
        "event",
        vec![
            "name::str::Event or conference name".to_string(),
            "date::str::Event date".to_string(),
            "location::str".to_string(),
            "topics::list::Conference topics".to_string(),
            "registration_fee::str".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("event: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Choice Fields (Classification within Structure)
    let text = r#"
Reservation at Le Bernardin for 4 people on March 15th at 7:30 PM. 
We'd prefer outdoor seating. Two guests are vegetarian and one is gluten-free.
"#;
    let schema = JsonSchema::new().structure(
        "reservation",
        vec![
            "restaurant::str::Restaurant name".to_string(),
            "date::str".to_string(),
            "time::str".to_string(),
            "party_size::[1|2|3|4|5|6+]::str::Number of guests".to_string(),
            "seating::[indoor|outdoor|bar]::str::Seating preference".to_string(),
            "dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("reservation: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multiple Instances ---
    // Multiple Transactions
    let text = r#"
Recent transactions:
- Jan 5: Starbucks $5.50 (food)
- Jan 5: Uber $23.00 (transport)  
- Jan 6: Amazon $156.99 (shopping)
"#;
    let schema = JsonSchema::new().structure(
        "transaction",
        vec![
            "date::str".to_string(),
            "merchant::str".to_string(),
            "amount::str".to_string(),
            "category::[food|transport|shopping|utilities]::str".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("transaction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Multiple Hotel Bookings
    let text = r#"
Alice Brown booked the Hilton Downtown from March 10 to March 12. She selected a double room 
for $340 total with breakfast and parking included.

Robert Taylor reserved The Grand Hotel, April 1 to April 5, suite at $1,200 total. 
Amenities include breakfast, wifi, gym, and spa access.
"#;
    let schema = JsonSchema::new().structure(
        "booking",
        vec![
            "guest::str::Guest name".to_string(),
            "hotel::str::Hotel name".to_string(),
            "check_in::str".to_string(),
            "check_out::str".to_string(),
            "room_type::[single|double|suite|deluxe]::str".to_string(),
            "total_price::str".to_string(),
            "amenities::[breakfast|wifi|parking|gym|spa]::list".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("booking: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Schema Builder (Multi-Task) ---
    // Multi-Task Extraction
    let schema = SchemaBuilder::new()
        .entities(vec![
            "person".to_string(),
            "company".to_string(),
            "location".to_string(),
        ])
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .structure("product")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("price").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("features").dtype(FieldDtype::List))
        .field(
            StructureFieldSpec::new("category")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "electronics".to_string(),
                    "software".to_string(),
                    "service".to_string(),
                ]),
        )
        .finish()
        .build();

    let text = "Apple CEO Tim Cook announced iPhone 15 for $999 with amazing new features. This is exciting!";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("multi_task: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Advanced Configuration
    let schema = SchemaBuilder::new()
        .classification("urgency", vec!["low".to_string(), "medium".to_string(), "high".to_string()])
        .structure("support_ticket")
        .field(StructureFieldSpec::new("ticket_id").dtype(FieldDtype::Str).threshold(0.9))
        .field(StructureFieldSpec::new("customer").dtype(FieldDtype::Str).description("Customer name"))
        .field(StructureFieldSpec::new("issue").dtype(FieldDtype::Str).description("Problem description"))
        .field(
            StructureFieldSpec::new("priority")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                    "urgent".to_string(),
                ]),
        )
        .field(
            StructureFieldSpec::new("tags")
                .dtype(FieldDtype::List)
                .choices(vec![
                    "bug".to_string(),
                    "feature".to_string(),
                    "support".to_string(),
                    "billing".to_string(),
                ]),
        )
        .finish()
        .build();

    let text = "Ticket #ABC-123: Jane Doe reports a billing issue with her invoice. This is urgent support. Tag: billing.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("support_ticket: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Examples ---
    // Financial Transaction Processing
    let text = r#"
Goldman Sachs processed a $2.5M equity trade for Tesla Inc. on March 15, 2024. 
Commission: $1,250. Status: Completed.
"#;
    let schema = JsonSchema::new().structure(
        "transaction",
        vec![
            "broker::str::Financial institution".to_string(),
            "amount::str::Transaction amount".to_string(),
            "security::str::Stock or financial instrument".to_string(),
            "date::str::Transaction date".to_string(),
            "commission::str::Fees charged".to_string(),
            "status::[pending|completed|failed]::str".to_string(),
            "type::[equity|bond|option|future]::str".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("financial_transaction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Medical Prescription Extraction
    let text = r#"
Patient: Sarah Johnson, 34, presented with chest pain.
Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily.
Follow-up scheduled for next Tuesday.
"#;
    let schema = JsonSchema::new()
        .structure(
            "patient",
            vec![
                "name::str::Patient full name".to_string(),
                "age::str::Patient age".to_string(),
                "symptoms::list::Reported symptoms".to_string(),
            ],
        )
        .structure(
            "prescription",
            vec![
                "medication::str::Drug name".to_string(),
                "dosage::str::Dosage amount".to_string(),
                "frequency::str::How often to take".to_string(),
            ],
        );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("medical_prescription: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // E-commerce Order Processing
    let text = r#"
Order #ORD-2024-001 for Alexandra Thompson
Items: Laptop Stand (2x $45.99), Wireless Mouse (1x $29.99), USB Hub (3x $35.50)
Subtotal: $228.46, Tax: $18.28, Total: $246.74
Status: Processing
"#;
    let schema = JsonSchema::new().structure(
        "order",
        vec![
            "order_id::str::Order number".to_string(),
            "customer::str::Customer name".to_string(),
            "items::list::Product names".to_string(),
            "quantities::list::Item quantities".to_string(),
            "unit_prices::list::Individual prices".to_string(),
            "subtotal::str".to_string(),
            "tax::str".to_string(),
            "total::str".to_string(),
            "status::[pending|processing|shipped|delivered]::str".to_string(),
        ],
    );
    let start = Instant::now();
    let out = pipeline.extract_json(text, &schema)?;
    println!("text: {text}");
    println!("ecommerce_order: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Confidence Scores and Character Positions ---
    let text = "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage.";
    let schema = JsonSchema::new().structure(
        "product",
        vec!["name::str".to_string(), "price".to_string(), "features".to_string()],
    );

    // With confidence
    let start = Instant::now();
    let out = pipeline.extract_json_with_confidence(text, &schema, 0.5)?;
    println!("include_confidence: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // With spans
    let start = Instant::now();
    let out = pipeline.extract_json_with_spans(text, &schema, 0.5)?;
    println!("include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // With both confidence and spans
    let start = Instant::now();
    let out = pipeline.extract_json_with_confidence_and_spans(text, &schema, 0.5)?;
    println!("include_confidence+include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
