use std::{collections::BTreeMap, time::Instant};

use anyhow::anyhow;
use gliner2_rs::{
    Result,
    classification::ClassAct,
    pipeline::Gliner2Pipeline,
    schema_spec::{ClassificationOptions, QuickClassificationTask, SchemaBuilder},
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

    // Mirrors `tutorial/1-classification.md`.

    // --- Single-label classification (basic) ---
    let schema = SchemaBuilder::new()
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .build();

    let text = "This product exceeded my expectations! Absolutely love it.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("sentiment: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Single-label classification (with confidence) ---
    let text = "The service was okay, nothing special but not bad either.";
    let start = Instant::now();
    let out = pipeline.extract_with_confidence(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("sentiment (confidence): {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multi-label classification ---
    let topics_schema = SchemaBuilder::new()
        .classification_with_options(
            "topics",
            vec![
                "technology".to_string(),
                "business".to_string(),
                "health".to_string(),
                "politics".to_string(),
                "sports".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.3,
                ..Default::default()
            },
        )
        .build();

    let text = "Apple announced new health monitoring features in their latest smartwatch, boosting their stock price.";
    let start = Instant::now();
    let out = pipeline.extract(text, &topics_schema, 0.5)?;
    println!("text: {text}");
    println!("topics: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let start = Instant::now();
    let out = pipeline.extract_with_confidence(text, &topics_schema, 0.5)?;
    println!("topics (confidence): {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Classification with label descriptions ---
    let doc_schema = SchemaBuilder::new()
        .classification(
            "document_type",
            vec![
                (
                    "invoice".to_string(),
                    "A bill for goods or services with payment details".to_string(),
                ),
                (
                    "receipt".to_string(),
                    "Proof of payment for a completed transaction".to_string(),
                ),
                (
                    "contract".to_string(),
                    "Legal agreement between parties with terms and conditions".to_string(),
                ),
                (
                    "proposal".to_string(),
                    "Document outlining suggested plans or services with pricing".to_string(),
                ),
            ],
        )
        .build();

    let text = "Please find attached the itemized bill for consulting services rendered in Q3 2024. Payment is due within 30 days.";
    let start = Instant::now();
    let out = pipeline.extract(text, &doc_schema, 0.5)?;
    println!("text: {text}");
    println!("document_type: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let text = "Thank you for your payment of $500. This confirms your transaction was completed on March 1st, 2024.";
    let start = Instant::now();
    let out = pipeline.extract(text, &doc_schema, 0.5)?;
    println!("text: {text}");
    println!("document_type: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Quick API (classify_text) ---
    let tasks: BTreeMap<String, QuickClassificationTask> = BTreeMap::from([(
        "sentiment".to_string(),
        vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]
        .into(),
    )]);

    let text = "The new AI model shows remarkable performance improvements.";
    let start = Instant::now();
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    println!("text: {text}");
    println!("quick_api: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let text = "The software keeps crashing and customer support is unresponsive.";
    let start = Instant::now();
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    println!("text: {text}");
    println!("quick_api: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let mut tasks = BTreeMap::new();
    tasks.insert(
        "sentiment".to_string(),
        vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]
        .into(),
    );
    tasks.insert(
        "urgency".to_string(),
        vec!["high".to_string(), "medium".to_string(), "low".to_string()].into(),
    );
    tasks.insert(
        "category".to_string(),
        QuickClassificationTask::config(
            vec![
                "tech".to_string(),
                "finance".to_string(),
                "politics".to_string(),
                "sports".to_string(),
            ],
            ClassificationOptions {
                multi_label: false,
                ..Default::default()
            },
        ),
    );

    let text = "Breaking: Tech giant announces major layoffs amid market downturn";
    let start = Instant::now();
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    println!("text: {text}");
    println!("quick_api_multi: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Quick API: multi-label with config ---
    let mut tasks = BTreeMap::new();
    tasks.insert(
        "product_aspects".to_string(),
        QuickClassificationTask::config(
            vec![
                "camera".to_string(),
                "battery".to_string(),
                "display".to_string(),
                "performance".to_string(),
                "design".to_string(),
                "heating".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.4,
                ..Default::default()
            },
        ),
    );

    let text = "The smartphone features an amazing camera but disappointing battery life and overheats frequently.";
    let start = Instant::now();
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    println!("text: {text}");
    println!("quick_api_multi_label: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let text = "Beautiful design with vibrant display, though the camera could be better.";
    let start = Instant::now();
    let out = pipeline.classify_text(text, &tasks, 0.5, false)?;
    println!("text: {text}");
    println!("quick_api_multi_label: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multiple Classification Tasks: basic multiple classifications ---
    let schema = SchemaBuilder::new()
        .classification(
            "sentiment",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        )
        .classification(
            "language",
            vec![
                "english".to_string(),
                "spanish".to_string(),
                "french".to_string(),
                "german".to_string(),
                "other".to_string(),
            ],
        )
        .classification(
            "formality",
            vec![
                "formal".to_string(),
                "informal".to_string(),
                "semi-formal".to_string(),
            ],
        )
        .classification(
            "intent",
            vec![
                "question".to_string(),
                "statement".to_string(),
                "request".to_string(),
                "complaint".to_string(),
            ],
        )
        .build();

    let text = "Could you please help me with my order? The service has been disappointing.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("multi_classification: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let text = "Hey! Just wanted to say your product rocks! 🎉";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("multi_classification: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Mixed single + multi-label classifications ---
    let schema = SchemaBuilder::new()
        .classification(
            "primary_topic",
            vec![
                "tech".to_string(),
                "business".to_string(),
                "health".to_string(),
                "sports".to_string(),
                "politics".to_string(),
            ],
        )
        .classification(
            "urgency",
            vec![
                "immediate".to_string(),
                "soon".to_string(),
                "later".to_string(),
                "not_urgent".to_string(),
            ],
        )
        .classification_with_options(
            "emotions",
            vec![
                "happy".to_string(),
                "sad".to_string(),
                "angry".to_string(),
                "surprised".to_string(),
                "fearful".to_string(),
                "disgusted".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.4,
                ..Default::default()
            },
        )
        .classification_with_options(
            "content_flags",
            vec![
                "inappropriate".to_string(),
                "spam".to_string(),
                "promotional".to_string(),
                "personal_info".to_string(),
                "financial_info".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.3,
                ..Default::default()
            },
        )
        .build();

    let text = "URGENT: I'm thrilled to announce our new product! But concerned about competitor reactions. Please keep confidential.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("mixed_single_multi: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let text = "Just saw the game - absolutely devastated by the loss. Can't believe the referee's terrible decision!";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("mixed_single_multi: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Domain-specific multiple classifications (support tickets) ---
    let support_schema = SchemaBuilder::new()
        .classification(
            "ticket_type",
            vec![
                "technical_issue".to_string(),
                "billing".to_string(),
                "feature_request".to_string(),
                "bug_report".to_string(),
                "other".to_string(),
            ],
        )
        .classification_with_options(
            "priority",
            vec![
                "critical".to_string(),
                "high".to_string(),
                "medium".to_string(),
                "low".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.7,
                ..Default::default()
            },
        )
        .classification_with_options(
            "product_area",
            vec![
                (
                    "authentication".to_string(),
                    "Login, passwords, security".to_string(),
                ),
                (
                    "payment".to_string(),
                    "Payment processing, subscriptions".to_string(),
                ),
                (
                    "ui".to_string(),
                    "User interface, design issues".to_string(),
                ),
                (
                    "performance".to_string(),
                    "Speed, loading, responsiveness".to_string(),
                ),
                (
                    "data".to_string(),
                    "Data loss, corruption, sync issues".to_string(),
                ),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.5,
                ..Default::default()
            },
        )
        .classification_with_options(
            "customer_sentiment",
            vec![
                "very_satisfied".to_string(),
                "satisfied".to_string(),
                "neutral".to_string(),
                "frustrated".to_string(),
                "very_frustrated".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.6,
                ..Default::default()
            },
        )
        .classification_with_options(
            "requires_action",
            vec![
                "immediate_response".to_string(),
                "investigation_needed".to_string(),
                "waiting_customer".to_string(),
                "resolved".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                ..Default::default()
            },
        )
        .build();

    let ticket_text = r#"
Subject: Cannot login - Urgent!

I've been trying to login for the past hour but keep getting error messages. 
This is critical as I need to process payments for my customers today. 
The page just keeps spinning and then times out. I'm extremely frustrated 
as this is costing me business. Please fix this immediately!
"#;
    let start = Instant::now();
    let out = pipeline.extract(ticket_text, &support_schema, 0.5)?;
    println!("ticket_text: {ticket_text}");
    println!("support_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let ticket_text2 = r#"
Hi team,

Thanks for the great product! I was wondering if you could add a dark mode feature? 
It would really help with eye strain during late night work sessions.

Best regards,
Happy Customer
"#;
    let start = Instant::now();
    let out = pipeline.extract(ticket_text2, &support_schema, 0.5)?;
    println!("ticket_text2: {ticket_text2}");
    println!("support_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Sequential classification (email routing) ---
    let email_schema = SchemaBuilder::new()
        .classification_with_options(
            "email_category",
            vec![
                "sales".to_string(),
                "support".to_string(),
                "hr".to_string(),
                "legal".to_string(),
                "general".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.6,
                ..Default::default()
            },
        )
        .classification_with_options(
            "sales_stage",
            vec![
                "lead".to_string(),
                "qualified".to_string(),
                "proposal".to_string(),
                "negotiation".to_string(),
                "closed".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.5,
                ..Default::default()
            },
        )
        .classification_with_options(
            "support_type",
            vec![
                "pre_sales".to_string(),
                "technical".to_string(),
                "account".to_string(),
                "billing".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.5,
                ..Default::default()
            },
        )
        .classification_with_options(
            "required_action",
            vec![
                "reply_needed".to_string(),
                "forward_to_team".to_string(),
                "schedule_meeting".to_string(),
                "no_action".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.4,
                ..Default::default()
            },
        )
        .classification_with_options(
            "response_timeframe",
            vec![
                "within_1_hour".to_string(),
                "within_24_hours".to_string(),
                "within_week".to_string(),
                "non_urgent".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.6,
                ..Default::default()
            },
        )
        .build();

    let email = r#"
Hi Sales Team,

I'm interested in your enterprise solution. We're currently evaluating vendors 
for our upcoming project. Could we schedule a demo next week? We need to make 
a decision by month end.

Best regards,
John from TechCorp
"#;
    let start = Instant::now();
    let out = pipeline.extract(email, &email_schema, 0.5)?;
    println!("email: {email}");
    println!("email_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let email2 = r#"
Dear HR Department,

I need to update my tax withholding information. Could someone please send me 
the necessary forms? This is somewhat urgent as I need this changed before the 
next payroll cycle.

Thank you,
Sarah
"#;
    let start = Instant::now();
    let out = pipeline.extract(email2, &email_schema, 0.5)?;
    println!("email2: {email2}");
    println!("email_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Complex analysis with multiple classifications ---
    let content_schema = SchemaBuilder::new()
        .classification(
            "content_type",
            vec![
                "article".to_string(),
                "comment".to_string(),
                "review".to_string(),
                "social_post".to_string(),
                "message".to_string(),
            ],
        )
        .classification(
            "primary_language",
            vec![
                "english".to_string(),
                "spanish".to_string(),
                "french".to_string(),
                "other".to_string(),
            ],
        )
        .classification_with_options(
            "quality_score",
            vec![
                "excellent".to_string(),
                "good".to_string(),
                "average".to_string(),
                "poor".to_string(),
                "spam".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.7,
                ..Default::default()
            },
        )
        .classification_with_options(
            "originality",
            vec![
                "original".to_string(),
                "derivative".to_string(),
                "duplicate".to_string(),
                "plagiarized".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.8,
                ..Default::default()
            },
        )
        .classification_with_options(
            "safety_flags",
            vec![
                (
                    "hate_speech".to_string(),
                    "Contains discriminatory or hateful content".to_string(),
                ),
                (
                    "violence".to_string(),
                    "Contains violent or threatening content".to_string(),
                ),
                (
                    "adult".to_string(),
                    "Contains adult or explicit content".to_string(),
                ),
                (
                    "misinformation".to_string(),
                    "Contains potentially false information".to_string(),
                ),
                (
                    "personal_info".to_string(),
                    "Contains personal identifying information".to_string(),
                ),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.3,
                ..Default::default()
            },
        )
        .classification_with_options(
            "engagement_potential",
            vec![
                "viral".to_string(),
                "high".to_string(),
                "medium".to_string(),
                "low".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.6,
                ..Default::default()
            },
        )
        .classification_with_options(
            "audience_fit",
            vec![
                "general".to_string(),
                "professional".to_string(),
                "academic".to_string(),
                "youth".to_string(),
                "senior".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.5,
                ..Default::default()
            },
        )
        .build();

    let content_text = r#"
Just discovered this amazing productivity hack that doubled my output! 
Here's what I do: I wake up at 5 AM, meditate for 20 minutes, then work 
in 90-minute focused blocks. The results have been incredible. My email 
is john.doe@example.com if you want more tips!
"#;
    let start = Instant::now();
    let out = pipeline.extract(content_text, &content_schema, 0.5)?;
    println!("content_text: {content_text}");
    println!("content_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let review_text = r#"
Worst product ever!!! Total scam! Don't buy this garbage. The company should 
be shut down for selling this junk. I'm going to report them to authorities.
"#;
    let start = Instant::now();
    let out = pipeline.extract(review_text, &content_schema, 0.5)?;
    println!("review_text: {review_text}");
    println!("content_schema: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Advanced configurations: custom thresholds ---
    let schema = SchemaBuilder::new()
        .classification_with_options(
            "is_spam",
            vec!["spam".to_string(), "not_spam".to_string()],
            ClassificationOptions {
                cls_threshold: 0.9,
                ..Default::default()
            },
        )
        .build();

    let text = "Congratulations! You've won $1,000,000! Click here to claim your prize now!";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("is_spam: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let schema = SchemaBuilder::new()
        .classification_with_options(
            "priority",
            vec![
                "urgent".to_string(),
                "high".to_string(),
                "normal".to_string(),
                "low".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.8,
                ..Default::default()
            },
        )
        .classification_with_options(
            "department",
            vec![
                "sales".to_string(),
                "support".to_string(),
                "billing".to_string(),
                "other".to_string(),
            ],
            ClassificationOptions {
                cls_threshold: 0.5,
                ..Default::default()
            },
        )
        .build();

    let text = "URGENT: Customer threatening to cancel $50k contract due to billing error";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("priority/department: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Advanced config: force activation ---
    let schema = SchemaBuilder::new()
        .classification_with_options(
            "category",
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            ClassificationOptions {
                class_act: ClassAct::Softmax,
                ..Default::default()
            },
        )
        .build();
    let text = "This clearly belongs to category B based on the criteria.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("category (class_act=softmax): {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Advanced config: complex multi-label example ---
    let schema = SchemaBuilder::new()
        .classification_with_options(
            "email_tags",
            vec![
                (
                    "action_required".to_string(),
                    "Email requires recipient to take action".to_string(),
                ),
                (
                    "meeting_request".to_string(),
                    "Email contains meeting invitation or scheduling".to_string(),
                ),
                (
                    "project_update".to_string(),
                    "Email contains project status or updates".to_string(),
                ),
                (
                    "urgent".to_string(),
                    "Email marked as urgent or time-sensitive".to_string(),
                ),
                (
                    "question".to_string(),
                    "Email contains questions requiring answers".to_string(),
                ),
                (
                    "fyi".to_string(),
                    "Informational email requiring no action".to_string(),
                ),
            ],
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.35,
                ..Default::default()
            },
        )
        .build();

    let email_text = r#"
Hi team,

Quick update on Project Alpha: We're ahead of schedule! 

However, I need your input on the design mockups by EOD tomorrow. 
Can we schedule a 30-min call this week to discuss?

This is quite urgent as the client is waiting.

Best,
Sarah
"#;
    let start = Instant::now();
    let out = pipeline.extract(email_text, &schema, 0.5)?;
    println!("email_text: {email_text}");
    println!("email_tags: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let email_text2 = r#"
Team,

Just wanted to let everyone know that I'll be out of office next Monday for a 
doctor's appointment. I'll be back Tuesday morning.

Thanks,
Mark
"#;
    let start = Instant::now();
    let out = pipeline.extract(email_text2, &schema, 0.5)?;
    println!("email_text2: {email_text2}");
    println!("email_tags: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Best practices: descriptions vs no descriptions ---
    let schema_with_desc = SchemaBuilder::new()
        .classification(
            "intent",
            vec![
                (
                    "purchase".to_string(),
                    "User wants to buy a product".to_string(),
                ),
                (
                    "return".to_string(),
                    "User wants to return a product".to_string(),
                ),
                (
                    "inquiry".to_string(),
                    "User asking for information".to_string(),
                ),
            ],
        )
        .build();

    let schema_no_desc = SchemaBuilder::new()
        .classification(
            "intent",
            vec![
                "purchase".to_string(),
                "return".to_string(),
                "inquiry".to_string(),
            ],
        )
        .build();

    let text = "Hi there — I want to return my order. What's the process?";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema_with_desc, 0.5)?;
    println!("text: {text}");
    println!("intent (with desc): {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let start = Instant::now();
    let out = pipeline.extract(text, &schema_no_desc, 0.5)?;
    println!("intent (no desc): {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Best practices: multi-label strategy ---
    let schema = SchemaBuilder::new()
        .classification_with_options(
            "product_features",
            vec![
                "waterproof".to_string(),
                "wireless".to_string(),
                "rechargeable".to_string(),
                "portable".to_string(),
            ],
            ClassificationOptions {
                multi_label: true,
                ..Default::default()
            },
        )
        .build();

    let text = "Looking for a portable waterproof speaker that's wireless and rechargeable.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("product_features: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let schema = SchemaBuilder::new()
        .classification_with_options(
            "size",
            vec![
                "small".to_string(),
                "medium".to_string(),
                "large".to_string(),
            ],
            ClassificationOptions {
                multi_label: false,
                ..Default::default()
            },
        )
        .build();
    let text = "A medium would fit me best.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("size: {:#?}", out.classifications);
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
