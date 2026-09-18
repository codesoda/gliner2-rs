use std::time::Instant;

use gliner2_rs::{
    Result,
    pipeline::Gliner2Pipeline,
    schema_spec::{FieldDtype, SchemaBuilder, StructureFieldSpec},
    validators::{RegexMode, RegexValidator},
};
mod common;
use common::model_paths_from_args;

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2-base-v1");

    let load_start = Instant::now();
    let pipeline = Gliner2Pipeline::new(&paths.model_dir, &paths.encoder, &paths.extractor)?;
    println!(
        "model load took: {:.2?} (onnx={})",
        load_start.elapsed(),
        paths.onnx_dir.display()
    );
    println!("-----------------");

    // Mirrors `tutorial/5-validator.md`.

    // --- Quick Start ---
    let email_validator = RegexValidator::new(r"^[\w\.-]+@[\w\.-]+\.\w+$");
    let schema = SchemaBuilder::new()
        .structure("contact")
        .field(
            StructureFieldSpec::new("email")
                .dtype(FieldDtype::Str)
                .validators(vec![email_validator]),
        )
        .finish()
        .build();

    let text = "Contact: john@company.com, not-an-email, jane@domain.org";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("contact: {:#?}", out.structures.get("contact"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Email Validation ---
    let email_validator = RegexValidator::new(r"^[\w\.-]+@[\w\.-]+\.\w+$");
    let schema = SchemaBuilder::new()
        .structure("contact")
        .field(
            StructureFieldSpec::new("email")
                .dtype(FieldDtype::List)
                .validators(vec![email_validator]),
        )
        .finish()
        .build();

    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("email_validation: {:#?}", out.structures.get("contact"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Phone Numbers (US Format) ---
    let phone_validator = RegexValidator::new(r"\(\d{3}\)\s\d{3}-\d{4}").mode(RegexMode::Partial);
    let schema = SchemaBuilder::new()
        .structure("contact")
        .field(
            StructureFieldSpec::new("phone")
                .dtype(FieldDtype::List)
                .validators(vec![phone_validator]),
        )
        .finish()
        .build();

    let text = "Call (555) 123-4567 or 5551234567";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("phone_validation: {:#?}", out.structures.get("contact"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- URLs Only ---
    let url_validator = RegexValidator::new(r"^https?://").mode(RegexMode::Partial);
    let schema = SchemaBuilder::new()
        .structure("resource")
        .field(
            StructureFieldSpec::new("url")
                .dtype(FieldDtype::List)
                .validators(vec![url_validator]),
        )
        .finish()
        .build();

    let text = "Visit https://example.com or www.site.com";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("url_validation: {:#?}", out.structures.get("resource"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Exclude Test Data ---
    // Note: Rust's `regex` crate doesn't support look-around assertions, so we use `^test|demo|sample`-style patterns.
    let no_test_validator = RegexValidator::new(r"^(test|demo|sample)")
        .mode(RegexMode::Partial)
        .exclude(true);
    let schema = SchemaBuilder::new()
        .structure("product")
        .field(
            StructureFieldSpec::new("name")
                .dtype(FieldDtype::List)
                .validators(vec![no_test_validator]),
        )
        .finish()
        .build();

    let text = "Products: iPhone, Test Phone, Samsung Galaxy";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("exclude_test_data: {:#?}", out.structures.get("product"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Length Constraints ---
    let length_validator = RegexValidator::new(r"^.{5,50}$");
    let schema = SchemaBuilder::new()
        .structure("names")
        .field(
            StructureFieldSpec::new("name")
                .dtype(FieldDtype::List)
                .validators(vec![length_validator]),
        )
        .finish()
        .build();

    let text = "Names: Jo, Alexander, A Very Long Name That Exceeds Fifty Characters";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("length_constraints: {:#?}", out.structures.get("names"));
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multiple Validators ---
    let username_validators = vec![
        RegexValidator::new(r"^[a-zA-Z0-9_]+$"),
        RegexValidator::new(r"^.{3,20}$"),
        RegexValidator::new(r"^admin").exclude(true).case_insensitive(true),
    ];

    let schema = SchemaBuilder::new()
        .structure("user")
        .field(
            StructureFieldSpec::new("username")
                .dtype(FieldDtype::List)
                .validators(username_validators),
        )
        .finish()
        .build();

    let text = "Users: ab, john_doe, user@domain, admin, valid_user123";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("multiple_validators: {:#?}", out.structures.get("user"));
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
