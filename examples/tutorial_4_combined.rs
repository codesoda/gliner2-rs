use std::{collections::BTreeMap, time::Instant};

use anyhow::anyhow;
use gliner2_rs::{
    Result,
    pipeline::Gliner2Pipeline,
    schema_spec::{ClassificationOptions, EntitySpec, FieldDtype, SchemaBuilder, StructureFieldSpec},
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
    let pipeline =
        Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?.with_classifier(classifier_onnx)?;
    println!(
        "model load took: {:.2?} (onnx={})",
        load_start.elapsed(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // Mirrors `tutorial/4-combined.md`.

    // --- Entities + Classification ---
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "product".to_string(), "company".to_string()])
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .classification(
            "category",
            vec!["review".to_string(), "news".to_string(), "opinion".to_string()],
        )
        .build();

    let text = "Tim Cook announced that Apple's new iPhone is exceeding sales expectations.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Entities + Structures ---
    let schema = SchemaBuilder::new()
        .entities(BTreeMap::from([
            (
                "person".to_string(),
                "Names of people mentioned".to_string(),
            ),
            (
                "date".to_string(),
                "Dates and time references".to_string(),
            ),
        ]))
        .structure("appointment")
        .field(StructureFieldSpec::new("patient").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("doctor").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("date"))
        .field(StructureFieldSpec::new("time"))
        .field(
            StructureFieldSpec::new("type")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "checkup".to_string(),
                    "followup".to_string(),
                    "consultation".to_string(),
                ]),
        )
        .finish()
        .build();

    let text = r#"
Dr. Sarah Johnson confirmed the appointment with John Smith for 
March 15th at 2:30 PM. This will be a follow-up consultation 
regarding his previous visit on February 1st.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Classification + Structures ---
    let schema = SchemaBuilder::new()
        .classification(
            "email_type",
            vec![
                "order_confirmation".to_string(),
                "shipping_update".to_string(),
                "promotional".to_string(),
                "support".to_string(),
            ],
        )
        .classification(
            "priority",
            vec!["urgent".to_string(), "normal".to_string(), "low".to_string()],
        )
        .structure("order_info")
        .field(StructureFieldSpec::new("order_number").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("items"))
        .field(StructureFieldSpec::new("total").dtype(FieldDtype::Str))
        .field(
            StructureFieldSpec::new("status")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "pending".to_string(),
                    "processing".to_string(),
                    "shipped".to_string(),
                    "delivered".to_string(),
                ]),
        )
        .finish()
        .build();

    let text = "Order confirmation: Order #ORD-42 includes 2x Laptop Stand and 1x Mouse, total $120. Status: processing. Please ship ASAP.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Complete Document Analysis ---
    let invoice_schema = SchemaBuilder::new()
        .classification(
            "document_type",
            vec![
                "invoice".to_string(),
                "credit_note".to_string(),
                "purchase_order".to_string(),
                "receipt".to_string(),
            ],
        )
        .classification(
            "payment_status",
            vec![
                "paid".to_string(),
                "unpaid".to_string(),
                "partial".to_string(),
                "overdue".to_string(),
            ],
        )
        .entities(vec![
            EntitySpec::new("company").description("Company names (buyer or seller)"),
            EntitySpec::new("person").description("Contact person names"),
            EntitySpec::new("date").description("Important dates"),
            EntitySpec::new("amount").description("Monetary amounts"),
        ])
        .structure("invoice_header")
        .field(StructureFieldSpec::new("invoice_number").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("issue_date").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("due_date").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("vendor_name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("customer_name").dtype(FieldDtype::Str))
        .finish()
        .structure("line_item")
        .field(StructureFieldSpec::new("description").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("quantity"))
        .field(StructureFieldSpec::new("unit_price"))
        .field(StructureFieldSpec::new("amount"))
        .field(
            StructureFieldSpec::new("tax_rate")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "0%".to_string(),
                    "5%".to_string(),
                    "10%".to_string(),
                    "20%".to_string(),
                ]),
        )
        .finish()
        .structure("payment_info")
        .field(
            StructureFieldSpec::new("method")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "bank_transfer".to_string(),
                    "credit_card".to_string(),
                    "check".to_string(),
                    "cash".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("terms").description("Payment terms like NET30"))
        .field(StructureFieldSpec::new("bank_details").dtype(FieldDtype::List))
        .finish()
        .build();

    let invoice_text = r#"
INVOICE #INV-2024-001
Vendor: Acme Consulting LLC
Customer: Globex Corp
Issue date: March 1, 2024
Due date: March 31, 2024

Line items:
- Consulting services 10 hours @ $150/hr = $1500 (tax 0%)
- Support retainer 1 month @ $500 = $500 (tax 0%)

Payment status: unpaid. Terms: NET30. Pay by bank_transfer.
"#;
    let start = Instant::now();
    let out = pipeline.extract(invoice_text, &invoice_schema, 0.5)?;
    println!("invoice_text: {invoice_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Customer Feedback Analysis ---
    let feedback_schema = SchemaBuilder::new()
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
                "mixed".to_string(),
            ],
        )
        .classification_with_options(
            "intent",
            BTreeMap::from([
                (
                    "complaint".to_string(),
                    "Customer expressing dissatisfaction".to_string(),
                ),
                (
                    "compliment".to_string(),
                    "Customer expressing satisfaction".to_string(),
                ),
                (
                    "suggestion".to_string(),
                    "Customer providing improvement ideas".to_string(),
                ),
                (
                    "question".to_string(),
                    "Customer asking for information".to_string(),
                ),
            ]),
            ClassificationOptions {
                multi_label: true,
                ..ClassificationOptions::default()
            },
        )
        .entities(vec![
            EntitySpec::new("product").description("Products or services mentioned"),
            EntitySpec::new("feature").description("Specific features discussed"),
            EntitySpec::new("competitor").description("Competing products mentioned"),
            EntitySpec::new("price_mention").description("Price points or cost references"),
        ])
        .structure("issue")
        .field(StructureFieldSpec::new("problem").dtype(FieldDtype::Str))
        .field(
            StructureFieldSpec::new("severity")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "critical".to_string(),
                    "major".to_string(),
                    "minor".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("affected_area").dtype(FieldDtype::List))
        .finish()
        .structure("suggestion")
        .field(StructureFieldSpec::new("improvement").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("benefit").description("Expected benefit of the suggestion"))
        .finish()
        .build();

    let feedback_text = "I love the new camera feature on the iPhone, but the battery life is majorly disappointing. Samsung lasts longer at this price. Please improve battery optimization; it would help travel use.";
    let start = Instant::now();
    let out = pipeline.extract(feedback_text, &feedback_schema, 0.5)?;
    println!("feedback_text: {feedback_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- News Article Analysis ---
    let news_schema = SchemaBuilder::new()
        .classification(
            "category",
            vec![
                "politics".to_string(),
                "business".to_string(),
                "technology".to_string(),
                "sports".to_string(),
                "entertainment".to_string(),
            ],
        )
        .classification(
            "bias",
            vec![
                "left".to_string(),
                "center".to_string(),
                "right".to_string(),
                "neutral".to_string(),
            ],
        )
        .classification(
            "factuality",
            vec![
                "fact".to_string(),
                "opinion".to_string(),
                "analysis".to_string(),
                "speculation".to_string(),
            ],
        )
        .entities(vec![
            EntitySpec::new("person").description("People mentioned in the article"),
            EntitySpec::new("organization").description("Companies, agencies, or groups"),
            EntitySpec::new("location").description("Places, cities, or countries"),
            EntitySpec::new("event").description("Named events or incidents"),
        ])
        .structure("quote")
        .field(StructureFieldSpec::new("speaker").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("statement").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("context").description("Context of the quote"))
        .finish()
        .structure("claim")
        .field(StructureFieldSpec::new("statement").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("source").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("evidence").dtype(FieldDtype::List))
        .finish()
        .build();

    let news_text = r#"
In Cupertino, Apple CEO Tim Cook said "we're thrilled with iPhone 15 demand" during the September event.
Analysts at Goldman Sachs claim the launch could boost revenue, citing pre-order numbers and supply-chain checks.
"#;
    let start = Instant::now();
    let out = pipeline.extract(news_text, &news_schema, 0.5)?;
    println!("news_text: {news_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- E-commerce Product Listing ---
    let product_schema = SchemaBuilder::new()
        .classification(
            "condition",
            vec![
                "new".to_string(),
                "used".to_string(),
                "refurbished".to_string(),
                "for_parts".to_string(),
            ],
        )
        .classification(
            "listing_type",
            vec!["buy_now".to_string(), "auction".to_string(), "best_offer".to_string()],
        )
        .entities(vec![
            EntitySpec::new("brand").description("Product brand or manufacturer"),
            EntitySpec::new("model").description("Specific model name or number"),
            EntitySpec::new("color").description("Product colors mentioned"),
            EntitySpec::new("size").description("Size specifications"),
        ])
        .structure("product")
        .field(StructureFieldSpec::new("title").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("price").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("features").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("category").dtype(FieldDtype::Str))
        .finish()
        .structure("shipping")
        .field(
            StructureFieldSpec::new("method")
                .dtype(FieldDtype::List)
                .choices(vec![
                    "standard".to_string(),
                    "express".to_string(),
                    "overnight".to_string(),
                    "international".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("cost").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("delivery_time").description("Estimated delivery timeframe"))
        .finish()
        .structure("seller")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("rating").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("location").dtype(FieldDtype::Str))
        .finish()
        .build();

    let listing_text = r#"
Listing: Apple iPhone 15 Pro Max (Blue) 256GB - Buy Now $1099
Features: amazing new camera, 120Hz display, USB-C.
Condition: new. Shipping: express or overnight, cost $25, delivery 1-2 days.
Seller: PhoneWorld rating 4.9, location: San Francisco.
"#;
    let start = Instant::now();
    let out = pipeline.extract(listing_text, &product_schema, 0.5)?;
    println!("listing_text: {listing_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Healthcare Clinical Note ---
    let clinical_schema = SchemaBuilder::new()
        .classification(
            "visit_type",
            vec![
                "initial_consultation".to_string(),
                "follow_up".to_string(),
                "emergency".to_string(),
                "routine_checkup".to_string(),
            ],
        )
        .classification(
            "urgency",
            vec!["urgent".to_string(), "routine".to_string(), "elective".to_string()],
        )
        .entities(vec![
            EntitySpec::new("symptom").description("Patient reported symptoms"),
            EntitySpec::new("diagnosis").description("Medical diagnoses or conditions"),
            EntitySpec::new("medication").description("Prescribed or mentioned medications"),
            EntitySpec::new("procedure").description("Medical procedures or tests"),
            EntitySpec::new("body_part").description("Anatomical references"),
        ])
        .structure("patient_info")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("age").dtype(FieldDtype::Str))
        .field(
            StructureFieldSpec::new("gender")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "male".to_string(),
                    "female".to_string(),
                    "other".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("chief_complaint").dtype(FieldDtype::Str))
        .finish()
        .structure("vital_signs")
        .field(StructureFieldSpec::new("blood_pressure").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("heart_rate").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("temperature").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("respiratory_rate").dtype(FieldDtype::Str))
        .finish()
        .structure("prescription")
        .field(StructureFieldSpec::new("medication").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("dosage").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("frequency"))
        .field(StructureFieldSpec::new("duration"))
        .field(
            StructureFieldSpec::new("route")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "oral".to_string(),
                    "IV".to_string(),
                    "topical".to_string(),
                    "injection".to_string(),
                ]),
        )
        .finish()
        .build();

    let clinical_text = r#"
Patient: Sarah Johnson, 34, female. Chief complaint: chest pain.
Vitals: BP 120/80, HR 72, Temp 98.6F, RR 16.
Assessment: possible hypertension. Prescribed Lisinopril 10mg daily for 30 days, route oral.
"#;
    let start = Instant::now();
    let out = pipeline.extract(clinical_text, &clinical_schema, 0.5)?;
    println!("clinical_text: {clinical_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Legal Document Analysis ---
    let legal_schema = SchemaBuilder::new()
        .classification(
            "document_type",
            vec![
                "contract".to_string(),
                "memorandum".to_string(),
                "brief".to_string(),
                "motion".to_string(),
                "order".to_string(),
            ],
        )
        .classification(
            "jurisdiction",
            vec![
                "federal".to_string(),
                "state".to_string(),
                "local".to_string(),
                "international".to_string(),
            ],
        )
        .entities(vec![
            EntitySpec::new("party").description("Parties involved (plaintiff, defendant, etc.)"),
            EntitySpec::new("attorney").description("Legal representatives"),
            EntitySpec::new("judge").description("Judicial officers"),
            EntitySpec::new("statute").description("Laws or regulations cited"),
            EntitySpec::new("case_citation").description("Referenced legal cases"),
        ])
        .structure("contract_term")
        .field(
            StructureFieldSpec::new("clause_type")
                .dtype(FieldDtype::Str)
                .choices(vec![
                    "payment".to_string(),
                    "delivery".to_string(),
                    "warranty".to_string(),
                    "liability".to_string(),
                    "termination".to_string(),
                ]),
        )
        .field(StructureFieldSpec::new("obligation").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("party_responsible").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("deadline"))
        .finish()
        .structure("claim")
        .field(StructureFieldSpec::new("type").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("plaintiff").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("defendant").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("amount").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("basis").description("Legal basis for the claim"))
        .finish()
        .build();

    let legal_text = r#"
In the State Court, plaintiff Acme Corp alleges breach of contract by defendant Globex Inc.
The payment clause requires delivery by March 31, 2024. The amount in dispute is $50,000.
"#;
    let start = Instant::now();
    let out = pipeline.extract(legal_text, &legal_schema, 0.5)?;
    println!("legal_text: {legal_text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Using Confidence Scores and Character Positions with Combined Schemas ---
    let schema = SchemaBuilder::new()
        .entities(vec!["person".to_string(), "company".to_string()])
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .relations(vec!["works_for".to_string()])
        .structure("product")
        .field(StructureFieldSpec::new("name").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("price").dtype(FieldDtype::Str))
        .finish()
        .build();

    let text = "Tim Cook works for Apple. The iPhone 15 costs $999. This is exciting!";
    let start = Instant::now();
    let out = pipeline.extract_with_confidence_and_spans(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("out: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
