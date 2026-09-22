use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    path::{Path, PathBuf},
};

use gliner2_rs::{
    boundary::record_schema::{RecordConfig, RecordMetadata},
    bundle::BundleManifest,
    classification::{ClassAct, FormattedClassification},
    entities::{FormattedEntitySpan, FormattedEntityValue},
    json::{JsonExtraction, JsonSchema},
    pipeline::AutoPipeline,
    relations::FormattedRelationExtraction,
    schema_spec::{
        FieldDtype, QuickClassificationTask, SchemaBuilder, SchemaSpec, StructureFieldSpec,
        StructureSpec,
    },
};

type SmokeResult<T> = Result<T, Box<dyn Error>>;

const TEXT: &str = "Alice joined Acme in Berlin. Bob works with Acme.";
const EXPLICIT_SPANS: [[usize; 2]; 3] = [[0, 5], [13, 17], [21, 27]];

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn require(condition: bool, message: impl Into<String>) -> SmokeResult<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn canonical_bundle(argument: Option<String>, name: &str) -> SmokeResult<PathBuf> {
    let raw = argument.ok_or_else(|| invalid(format!("missing {name} argument")))?;
    let path = PathBuf::from(raw);
    require(path.is_absolute(), format!("{name} must be absolute"))?;
    let canonical = path.canonicalize()?;
    require(canonical.is_dir(), format!("{name} is not a directory"))?;
    Ok(canonical)
}

fn inspect_boundary_manifest(manifest: &BundleManifest) -> SmokeResult<()> {
    require(
        manifest.hf_revision.len() == 40
            && manifest
                .hf_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "boundary manifest hf_revision is not an immutable lowercase commit SHA",
    )?;
    require(
        !manifest.files.is_empty(),
        "boundary manifest files map is empty",
    )?;
    println!(
        "validated boundary metadata: hf_model={} hf_revision={} files={}",
        manifest.hf_model,
        manifest.hf_revision,
        manifest.files.len()
    );
    Ok(())
}

fn validate_slice(source: &str, expected: &str, start: usize, end: usize) -> SmokeResult<()> {
    require(
        start < end,
        format!("empty or reversed span [{start}, {end})"),
    )?;
    require(
        source.is_char_boundary(start) && source.is_char_boundary(end),
        format!("span [{start}, {end}) is not on UTF-8 boundaries"),
    )?;
    let actual = source
        .get(start..end)
        .ok_or_else(|| invalid(format!("span [{start}, {end}) is outside the source")))?;
    require(
        actual == expected,
        format!("span [{start}, {end}) text mismatch: {actual:?} != {expected:?}"),
    )
}

fn validate_entity_span(source: &str, span: &FormattedEntitySpan) -> SmokeResult<()> {
    match span {
        FormattedEntitySpan::Text(_) => Ok(()),
        FormattedEntitySpan::TextWithConfidence { confidence, .. } => {
            require(confidence.is_finite(), "non-finite entity confidence")
        }
        FormattedEntitySpan::TextWithSpans { text, start, end } => {
            validate_slice(source, text, *start, *end)
        }
        FormattedEntitySpan::TextWithConfidenceAndSpans {
            text,
            confidence,
            start,
            end,
        } => {
            require(confidence.is_finite(), "non-finite entity confidence")?;
            validate_slice(source, text, *start, *end)
        }
    }
}

fn validate_entity_value(source: &str, value: &FormattedEntityValue) -> SmokeResult<()> {
    match value {
        FormattedEntityValue::List(spans) => {
            for span in spans {
                validate_entity_span(source, span)?;
            }
        }
        FormattedEntityValue::Single(span) => {
            if let Some(span) = span {
                validate_entity_span(source, span)?;
            }
        }
    }
    Ok(())
}

fn validate_structures(
    source: &str,
    structures: &JsonExtraction,
    expected: &str,
) -> SmokeResult<()> {
    let records = structures
        .get(expected)
        .ok_or_else(|| invalid(format!("missing requested structure task {expected:?}")))?;
    for record in records {
        for value in record.values() {
            validate_entity_value(source, value)?;
        }
    }
    Ok(())
}

fn validate_classification(
    value: &FormattedClassification,
    allowed_labels: &BTreeSet<&str>,
) -> SmokeResult<()> {
    let check_label = |label: &str| {
        require(
            allowed_labels.contains(label),
            format!("classification returned undeclared label {label:?}"),
        )
    };
    match value {
        FormattedClassification::Single(label) => check_label(label),
        FormattedClassification::SingleWithConfidence { label, confidence } => {
            check_label(label)?;
            require(
                confidence.is_finite(),
                "non-finite classification confidence",
            )
        }
        FormattedClassification::Multi(labels) => {
            for label in labels {
                check_label(label)?;
            }
            Ok(())
        }
        FormattedClassification::MultiWithConfidence(labels) => {
            for (label, confidence) in labels {
                check_label(label)?;
                require(
                    confidence.is_finite(),
                    "non-finite classification confidence",
                )?;
            }
            Ok(())
        }
    }
}

fn entity_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let labels = vec!["person".to_owned(), "organization".to_owned()];
    let output = pipeline.extract_entities_text(TEXT, labels.clone(), 0.0, true, true)?;
    for label in labels {
        let value = output
            .get(&label)
            .ok_or_else(|| invalid(format!("missing requested entity task {label:?}")))?;
        validate_entity_value(TEXT, value)?;
    }
    println!("entity API: task presence, finite confidence, and source slices checked");
    Ok(())
}

fn classification_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let task = "sentiment";
    let labels = vec!["positive".to_owned(), "negative".to_owned()];
    let tasks = BTreeMap::from([(
        task.to_owned(),
        QuickClassificationTask::labels(labels.clone()),
    )]);
    let output = pipeline.classify_text(TEXT, &tasks, 0.0, true)?;
    let value = output
        .get(task)
        .ok_or_else(|| invalid(format!("missing requested classification task {task:?}")))?;
    validate_classification(value, &labels.iter().map(String::as_str).collect())?;
    let allowed = labels.iter().map(String::as_str).collect();
    let options =
        pipeline.classify_with_options(TEXT, task, &labels, true, 0.0, ClassAct::Sigmoid)?;
    validate_classification(&options.format(true), &allowed)?;
    let descriptions = vec![
        ("positive".to_owned(), "A positive statement".to_owned()),
        ("negative".to_owned(), "A negative statement".to_owned()),
    ];
    let described = pipeline.classify_with_descriptions_and_options(
        TEXT,
        task,
        &labels,
        &descriptions,
        false,
        0.0,
        ClassAct::Softmax,
    )?;
    validate_classification(&described.format(true), &allowed)?;
    println!(
        "classification text/options/descriptions APIs: declared labels and finite confidence checked"
    );
    Ok(())
}

fn legacy_json_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let task = "legacy_person";
    let schema = JsonSchema::new().structure(
        task,
        vec!["name::str".to_owned(), "organization::str".to_owned()],
    );
    let output = pipeline.extract_json_with_confidence_and_spans(TEXT, &schema, 0.0)?;
    validate_structures(TEXT, &output, task)?;
    // Exercise the options entry point separately from the convenience wrapper.
    let options = pipeline.extract_json_with_options(TEXT, &schema, 0.0, false, true)?;
    validate_structures(TEXT, &options, task)?;
    println!(
        "legacy JSON and options APIs: task presence, finite confidence, and source slices checked"
    );
    Ok(())
}

fn annotated_record_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let task = "annotated_people";
    let schema = SchemaSpec {
        structures: vec![StructureSpec {
            name: task.to_owned(),
            fields: vec![
                StructureFieldSpec::new("name").dtype(FieldDtype::Str),
                StructureFieldSpec::new("organization").dtype(FieldDtype::Str),
            ],
        }],
        ..SchemaSpec::default()
    };
    let metadata = RecordMetadata::from([(task.to_owned(), RecordConfig::natural("name"))]);
    let output = pipeline.extract_with_records(TEXT, &schema, &metadata, 0.0, true, true)?;
    validate_structures(TEXT, &output.structures, task)?;
    let json_schema = JsonSchema::new().structure(
        task,
        vec!["name::str".to_owned(), "organization::str".to_owned()],
    );
    let json_records =
        pipeline.extract_json_with_records(TEXT, &json_schema, &metadata, 0.0, true, true)?;
    validate_structures(TEXT, &json_records, task)?;
    println!(
        "extract_with_records and extract_json_with_records: task presence, finite confidence, and source slices checked"
    );
    Ok(())
}

fn relation_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let task = "works for";
    let output =
        pipeline.extract_relations_with_confidence_and_spans(TEXT, &[task.to_owned()], 0.0)?;
    validate_relations(TEXT, &output, task)?;
    let options =
        pipeline.extract_relations_with_options(TEXT, &[task.to_owned()], 0.0, false, true)?;
    validate_relations(TEXT, &options, task)?;
    // Distinct texts and batch_size=1 exercise chunking and output cardinality.
    let texts = [TEXT, "Carol joined Delta in Rome."];
    let batch = pipeline.batch_extract_relations(&texts, &[task.to_owned()], 0.0, 1)?;
    require(
        batch.len() == texts.len(),
        "relation batch changed input cardinality",
    )?;
    for (source, output) in texts.iter().zip(&batch) {
        let pairs = output
            .get(task)
            .ok_or_else(|| invalid("relation batch omitted the requested task"))?;
        for (head, tail) in pairs {
            require(
                !head.is_empty() && source.contains(head.as_str()),
                "batch head not in source",
            )?;
            require(
                !tail.is_empty() && source.contains(tail.as_str()),
                "batch tail not in source",
            )?;
        }
    }
    println!(
        "relation/options/batch APIs: tasks, cardinality, finite confidence, and source text checked"
    );
    Ok(())
}

fn validate_relations(
    source: &str,
    output: &FormattedRelationExtraction,
    expected: &str,
) -> SmokeResult<()> {
    let pairs = output
        .get(expected)
        .ok_or_else(|| invalid(format!("missing requested relation task {expected:?}")))?;
    for pair in pairs {
        validate_entity_span(source, &pair.head)?;
        validate_entity_span(source, &pair.tail)?;
    }
    Ok(())
}

fn explicit_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let labels = vec!["person".to_owned(), "organization".to_owned()];
    let groups = pipeline.score_explicit_spans(TEXT, &labels, &EXPLICIT_SPANS)?;
    require(
        groups.len() == labels.len(),
        "explicit scoring changed the label-group count",
    )?;
    for (group, expected_label) in groups.iter().zip(&labels) {
        require(
            &group.label == expected_label,
            "explicit scoring changed label order",
        )?;
        require(
            group.spans.len() == EXPLICIT_SPANS.len(),
            "explicit scoring changed span count",
        )?;
        for (score, [start, end]) in group.spans.iter().zip(EXPLICIT_SPANS) {
            require(
                score.start == start && score.end == end,
                "explicit scoring changed caller span order or coordinates",
            )?;
            validate_slice(TEXT, &score.text, score.start, score.end)?;
            require(score.logit.is_finite(), "non-finite explicit logit")?;
            require(
                score.confidence.is_finite(),
                "non-finite explicit confidence",
            )?;
        }
    }
    println!("explicit API: order, finite scores, and source slices checked");
    Ok(())
}

fn boundary_smoke(bundle: &Path) -> SmokeResult<()> {
    let validated = gliner2_rs::bundle::validate_bundle(bundle)?;
    inspect_boundary_manifest(&validated.manifest)?;
    let pipeline = AutoPipeline::from_dir(bundle)?;
    entity_smoke(&pipeline)?;
    classification_smoke(&pipeline)?;
    legacy_json_smoke(&pipeline)?;
    annotated_record_smoke(&pipeline)?;
    relation_smoke(&pipeline)?;
    explicit_smoke(&pipeline)?;
    combined_smoke(&pipeline)?;
    println!("GLiNER2.5 AutoPipeline structural smoke passed (no accuracy assertion)");
    Ok(())
}

fn v2_smoke(bundle: &Path) -> SmokeResult<()> {
    // v2 has no boundary export-manifest contract and is intentionally not
    // passed to bundle::validate_bundle. AutoPipeline still requires all
    // tokenizer/config/graph files to be colocated in this clean bundle.
    let pipeline = AutoPipeline::from_dir(bundle)?;
    entity_smoke(&pipeline)?;
    classification_smoke(&pipeline)?;
    legacy_json_smoke(&pipeline)?;
    relation_smoke(&pipeline)?;
    combined_smoke(&pipeline)?;

    // Boundary-only APIs must fail explicitly on v2, not silently discard inputs.
    let schema = SchemaSpec::default();
    let metadata = RecordMetadata::from([("people".to_owned(), RecordConfig::natural("name"))]);
    let json_schema = JsonSchema::new().structure("people", vec!["name::str".to_owned()]);
    require(
        pipeline
            .extract_with_records(TEXT, &schema, &metadata, 0.0, true, true)
            .is_err(),
        "v2 accepted boundary record metadata",
    )?;
    require(
        pipeline
            .extract_json_with_records(TEXT, &json_schema, &metadata, 0.0, true, true)
            .is_err(),
        "v2 JSON accepted boundary record metadata",
    )?;
    require(
        pipeline
            .score_explicit_spans(TEXT, &["person".to_owned()], &EXPLICIT_SPANS)
            .is_err(),
        "v2 accepted boundary explicit-span scoring",
    )?;
    println!("v2 AutoPipeline compatibility smoke passed (no boundary-manifest validation)");
    Ok(())
}

fn combined_smoke(pipeline: &AutoPipeline) -> SmokeResult<()> {
    let combined = SchemaBuilder::new()
        .entities(vec!["person".to_owned()])
        .classification(
            "sentiment",
            vec!["positive".to_owned(), "negative".to_owned()],
        )
        .build();
    let output = pipeline.extract_with_confidence_and_spans(TEXT, &combined, 0.0)?;
    require(
        output.entities.contains_key("person"),
        "combined extraction omitted entity task",
    )?;
    require(
        output.classifications.contains_key("sentiment"),
        "combined extraction omitted classification task",
    )?;
    for value in output.entities.values() {
        validate_entity_value(TEXT, value)?;
    }
    let allowed = BTreeSet::from(["positive", "negative"]);
    validate_classification(&output.classifications["sentiment"], &allowed)?;
    println!(
        "combined extraction API: requested tasks, source slices and finite confidence checked"
    );
    Ok(())
}

fn main() -> SmokeResult<()> {
    let mut args = env::args().skip(1);
    let boundary = canonical_bundle(args.next(), "boundary bundle")?;
    let v2 = canonical_bundle(args.next(), "v2 bundle")?;
    require(args.next().is_none(), "unexpected extra argument")?;
    require(boundary != v2, "boundary and v2 bundles must differ")?;

    boundary_smoke(&boundary)?;
    v2_smoke(&v2)?;
    println!("selected public task-family and major records/options/batch smoke checks passed");
    println!(
        "not exhaustive method/adapter/raw-helper coverage; no semantic accuracy or release-readiness claim"
    );
    Ok(())
}
