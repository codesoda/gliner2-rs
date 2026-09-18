use std::{collections::BTreeMap, time::Instant};

use anyhow::anyhow;
use gliner2_rs::{
    Result,
    pipeline::Gliner2Pipeline,
    schema_spec::{FieldDtype, RelationSpec, SchemaBuilder, StructureFieldSpec},
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

    // Mirrors `tutorial/6-relation_extraction.md`.

    // --- Basic Relation Extraction (Simple Example) ---
    let text = "John works for Apple Inc. and lives in San Francisco.";
    let relation_types = vec!["works_for".to_string(), "lives_in".to_string()];
    let start = Instant::now();
    let out = pipeline.extract_relations(text, &relation_types, 0.5)?;
    println!("text: {text}");
    println!("relation_extraction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Using Schema Builder ---
    let schema = SchemaBuilder::new().relations(relation_types.clone()).build();
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("schema.relations + extract(): {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Understanding the Output Format (include empty requested types) ---
    let text = "Alice manages the Engineering team. Bob reports to Alice.";
    let relation_types = vec![
        "manages".to_string(),
        "reports_to".to_string(),
        "founded".to_string(),
    ];
    let start = Instant::now();
    let out = pipeline.extract_relations(text, &relation_types, 0.5)?;
    println!("text: {text}");
    println!("relation_extraction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multiple Relation Types ---
    let text = r#"
Sarah founded TechCorp in 2020. She is married to Mike, 
who works at Google. TechCorp is located in Seattle.
"#;
    let relation_types = vec![
        "founded".to_string(),
        "married_to".to_string(),
        "works_at".to_string(),
        "located_in".to_string(),
    ];
    let start = Instant::now();
    let out = pipeline.extract_relations(text, &relation_types, 0.5)?;
    println!("text: {text}");
    println!("relation_extraction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Multiple Instances per Relation Type ---
    let text = r#"
John works for Microsoft. Mary works for Google. 
Bob works for Apple. All three live in California.
"#;
    let relation_types = vec!["works_for".to_string(), "lives_in".to_string()];
    let start = Instant::now();
    let out = pipeline.extract_relations(text, &relation_types, 0.5)?;
    println!("text: {text}");
    println!("relation_extraction: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Relation Extraction with Descriptions ---
    // (Descriptions are passed as a prompt to the relation schema.)
    let schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "works_for".to_string(),
                "Employment relationship where person works at organization".to_string(),
            ),
            (
                "founded".to_string(),
                "Founding relationship where person created organization".to_string(),
            ),
            (
                "acquired".to_string(),
                "Acquisition relationship where company bought another company".to_string(),
            ),
            (
                "located_in".to_string(),
                "Geographic relationship where entity is in a location".to_string(),
            ),
        ]))
        .build();

    let text = r#"
Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne, California.
Tesla acquired SolarCity in 2016. Many engineers work for SpaceX.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("relations_with_descriptions: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Advanced Configuration (per-relation thresholds) ---
    let schema = SchemaBuilder::new()
        .relations(vec![
            RelationSpec::new("works_for")
                .description("Employment or professional relationship")
                .threshold(0.7),
            RelationSpec::new("located_in")
                .description("Geographic containment relationship")
                .threshold(0.6),
            RelationSpec::new("reports_to")
                .description("Organizational hierarchy relationship")
                .threshold(0.8),
        ])
        .build();

    let text = "Sundar Pichai reports to the board of directors. Many engineers work for Google. Google is located in Mountain View.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("advanced_config: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Custom Thresholds ---
    // Global threshold (quick API)
    let text = "Meta acquired Oculus. Google merged with NobodyCorp.";
    let relation_types = vec!["acquired".to_string(), "merged_with".to_string()];
    let start = Instant::now();
    let out = pipeline.extract_relations(text, &relation_types, 0.8)?;
    println!("text: {text}");
    println!("threshold=0.8: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- With Confidence Scores and Character Positions ---
    let text = "John works for Apple Inc. and lives in San Francisco.";
    let relation_types = vec!["works_for".to_string(), "lives_in".to_string()];

    let start = Instant::now();
    let out = pipeline.extract_relations_with_confidence(text, &relation_types, 0.5)?;
    println!("include_confidence: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let start = Instant::now();
    let out = pipeline.extract_relations_with_spans(text, &relation_types, 0.5)?;
    println!("include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    let start = Instant::now();
    let out = pipeline.extract_relations_with_confidence_and_spans(text, &relation_types, 0.5)?;
    println!("include_confidence+include_spans: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Batch Processing ---
    let texts = vec![
        "John works for Microsoft and lives in Seattle.",
        "Sarah founded TechStartup in 2020.",
        "Bob reports to Alice at Google.",
    ];
    let relation_types = vec![
        "works_for".to_string(),
        "founded".to_string(),
        "reports_to".to_string(),
        "lives_in".to_string(),
    ];
    let start = Instant::now();
    let batch = pipeline.batch_extract_relations(&texts, &relation_types, 0.5, 8)?;
    println!("batch_extract_relations: {batch:#?}");
    println!("batch took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Combining with Other Tasks ---
    // Relations + Entities
    let schema = SchemaBuilder::new()
        .entities(vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
        ])
        .relations(vec!["works_for".to_string(), "located_in".to_string()])
        .build();

    let text = "Tim Cook works for Apple Inc., which is located in Cupertino, California.";
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("relations_plus_entities: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Relations + Classification + Structures
    let schema = SchemaBuilder::new()
        .classification(
            "document_type",
            vec!["news".to_string(), "report".to_string(), "announcement".to_string()],
        )
        .entities(vec!["person".to_string(), "company".to_string()])
        .relations(vec!["works_for".to_string(), "acquired".to_string()])
        .structure("event")
        .field(StructureFieldSpec::new("date").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("description").dtype(FieldDtype::Str))
        .finish()
        .build();

    let text = r#"
BREAKING: Microsoft announced today that it acquired GitHub. 
Satya Nadella, CEO of Microsoft, confirmed the deal. 
The acquisition was finalized on October 26, 2018.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &schema, 0.5)?;
    println!("text: {text}");
    println!("relations_plus_classification_plus_structures: {out:#?}");
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // --- Real-World Examples ---
    // Organizational Relationships
    let org_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "reports_to".to_string(),
                "Direct reporting relationship in organizational hierarchy".to_string(),
            ),
            (
                "manages".to_string(),
                "Management relationship where person manages team/department".to_string(),
            ),
            (
                "works_for".to_string(),
                "Employment relationship".to_string(),
            ),
            (
                "founded".to_string(),
                "Founding relationship".to_string(),
            ),
            (
                "acquired".to_string(),
                "Company acquisition relationship".to_string(),
            ),
        ]))
        .build();

    let text = r#"
Sundar Pichai is the CEO of Google. He reports to the board of directors.
Google acquired YouTube in 2006. Many engineers work for Google.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &org_schema, 0.5)?;
    println!("org_text: {text}");
    println!("org_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Medical Relationships
    let medical_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "treats".to_string(),
                "Medical treatment relationship between doctor and patient".to_string(),
            ),
            (
                "prescribed_for".to_string(),
                "Prescription relationship between medication and condition".to_string(),
            ),
            (
                "causes".to_string(),
                "Causal relationship between condition and symptom".to_string(),
            ),
            (
                "located_in".to_string(),
                "Anatomical location relationship".to_string(),
            ),
        ]))
        .build();

    let text = r#"
Dr. Smith treats patients with diabetes. Metformin is prescribed for Type 2 Diabetes.
High blood sugar causes frequent urination. The pancreas is located in the abdomen.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &medical_schema, 0.5)?;
    println!("medical_text: {text}");
    println!("medical_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Financial Relationships
    let finance_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "invested_in".to_string(),
                "Investment relationship between investor and company".to_string(),
            ),
            (
                "acquired".to_string(),
                "Company acquisition relationship".to_string(),
            ),
            (
                "merged_with".to_string(),
                "Merger relationship between companies".to_string(),
            ),
            (
                "owns".to_string(),
                "Ownership relationship".to_string(),
            ),
        ]))
        .build();

    let text = r#"
SoftBank invested in Uber in 2018. Microsoft acquired LinkedIn in 2016.
Disney merged with 21st Century Fox. Berkshire Hathaway owns Geico.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &finance_schema, 0.5)?;
    println!("finance_text: {text}");
    println!("finance_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Geographic Relationships
    let geo_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "located_in".to_string(),
                "Geographic containment (city in country, etc.)".to_string(),
            ),
            (
                "borders".to_string(),
                "Geographic adjacency relationship".to_string(),
            ),
            (
                "capital_of".to_string(),
                "Capital city relationship".to_string(),
            ),
            (
                "flows_through".to_string(),
                "River or waterway relationship".to_string(),
            ),
        ]))
        .build();

    let text = r#"
Paris is the capital of France. France borders Germany and Spain.
The Seine flows through Paris. Paris is located in France.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &geo_schema, 0.5)?;
    println!("geo_text: {text}");
    println!("geo_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Family Relationships
    let family_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            ("married_to".to_string(), "Marriage relationship".to_string()),
            ("parent_of".to_string(), "Parent-child relationship".to_string()),
            ("sibling_of".to_string(), "Sibling relationship".to_string()),
            ("related_to".to_string(), "General family relationship".to_string()),
        ]))
        .build();

    let text = r#"
John is married to Mary. They are parents of two children: Alice and Bob.
Alice and Bob are siblings. Mary is related to her sister Sarah.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &family_schema, 0.5)?;
    println!("family_text: {text}");
    println!("family_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    // Academic Relationships
    let academic_schema = SchemaBuilder::new()
        .relations(BTreeMap::from([
            (
                "authored".to_string(),
                "Publication relationship between author and paper".to_string(),
            ),
            (
                "cited".to_string(),
                "Citation relationship between papers".to_string(),
            ),
            (
                "supervised".to_string(),
                "Academic supervision relationship".to_string(),
            ),
            (
                "affiliated_with".to_string(),
                "Institutional affiliation relationship".to_string(),
            ),
        ]))
        .build();

    let text = r#"
Dr. Johnson authored the paper on machine learning. The paper cited 
previous work by Dr. Smith. Dr. Johnson supervises graduate students 
at MIT, where she is affiliated with the Computer Science department.
"#;
    let start = Instant::now();
    let out = pipeline.extract(text, &academic_schema, 0.5)?;
    println!("academic_text: {text}");
    println!("academic_relations: {:#?}", out.relations);
    println!("inference took: {:.2?}", start.elapsed());

    Ok(())
}
