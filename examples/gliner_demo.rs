use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use anyhow::anyhow;
use gliner2_rs::{
    Result,
    classification::FormattedClassification,
    entities::{FormattedEntitySpan, FormattedEntityValue},
    pipeline::Gliner2Pipeline,
    schema_spec::{
        ClassificationOptions, EntityOptions, FieldDtype, RelationSpec, SchemaBuilder,
        StructureFieldSpec,
    },
};
mod common;
use common::model_paths_from_args;

fn main() -> Result<()> {
    // Mirrors: https://gist.github.com/LxYuan0420/d87d91142715cd9624c6f343975c4ffa
    //
    // NOTE: our Rust `ExtractionResult` groups results by task type (entities/classifications/
    // structures/relations), while the Python demo prints a flattened JSON. The content should be
    // comparable.
    //
    // This example also compares our Rust output against the Python "repr output" embedded at the
    // top of the gist (as a comment). Exact scores/spans can differ slightly due to decoding and
    // model/export details, so we print a structured diff.

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

    println!("Model: {} on CPU", paths.onnx_dir.display());
    println!("Running multi-task extraction on dummy memo...");
    println!("-----------------");

    let dummy_report = r#"
HarborView Capital reiterated its quarterly outlook after the portfolio review with
Seaside Renewables and Anchor Freight. The capital committee kept the cash dividend
at $0.34 per unit and guided to a payout ratio between 55% and 60% as shipping cash
flows normalize. Management said the liquidity cushion sits at $1.6B including an
undrawn revolver that BlueCurrent Bank continues to oversee. The memo noted that
HarborView allocates to Seaside to accelerate grid interconnects while Anchor Freight
funds the port automation push. The policy signal emphasized steady distributions
while keeping dry powder for bolt-on deals. Record date is penciled in for March 18
with payment the following week unless regulators request a pause. Watchlist items
include board approval of the dividend ladder, execution on Asia feeder routes, and
the impact of a stricter Basel-oriented liquidity floor set by the maritime authority.
"#
    .trim();

    let schema = SchemaBuilder::new()
        .entities_with_options(
            BTreeMap::from([
                (
                    "asset_manager".to_string(),
                    "Named fund or investment firm steering capital allocation".to_string(),
                ),
                (
                    "portfolio_company".to_string(),
                    "Operating companies receiving capital or attention".to_string(),
                ),
                (
                    "policy_signal".to_string(),
                    "Language that hints at payout or capital allocation posture".to_string(),
                ),
                (
                    "regulatory_body".to_string(),
                    "Oversight entities or watchdogs mentioned explicitly".to_string(),
                ),
                (
                    "liquidity_cushion".to_string(),
                    "References to liquidity buffers, coverage, or reserves".to_string(),
                ),
            ]),
            EntityOptions {
                dtype: FieldDtype::List,
                threshold: Some(0.35),
            },
        )
        .relations(vec![
            RelationSpec::new("allocates_to")
                .description("Source allocates capital or focus toward a target entity")
                .threshold(0.3),
            RelationSpec::new("overseen_by")
                .description("Entity under oversight from a regulator or committee")
                .threshold(0.3),
            RelationSpec::new("funds_from")
                .description("Capital inflow from a specified source into a target")
                .threshold(0.3),
        ])
        .classification_with_options(
            "sector_focus",
            BTreeMap::from([
                (
                    "finance".to_string(),
                    "Capital markets, funds, balance sheets, distributions".to_string(),
                ),
                (
                    "healthcare".to_string(),
                    "Providers, devices, and life sciences topics".to_string(),
                ),
                (
                    "sports".to_string(),
                    "Teams, leagues, games, or athletic venues".to_string(),
                ),
            ]),
            ClassificationOptions {
                multi_label: false,
                cls_threshold: 0.3,
                ..ClassificationOptions::default()
            },
        )
        .classification_with_options(
            "signal_axes",
            BTreeMap::from([
                (
                    "dividend_policy".to_string(),
                    "Mentions payouts, cash per share, or payout ratios".to_string(),
                ),
                (
                    "liquidity_guardrails".to_string(),
                    "Speaks about cushions, revolvers, or funding buffers".to_string(),
                ),
                (
                    "regulatory_watch".to_string(),
                    "References oversight, regulators, or authority action".to_string(),
                ),
                (
                    "growth_investment".to_string(),
                    "Signals expansion, capex, bolt-ons, or automation spend".to_string(),
                ),
                (
                    "operational_risk".to_string(),
                    "Flags execution risk or disruptions".to_string(),
                ),
            ]),
            ClassificationOptions {
                multi_label: true,
                cls_threshold: 0.28,
                ..ClassificationOptions::default()
            },
        )
        .structure("dividend_outlook")
        .field(StructureFieldSpec::new("record_date").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("cash_per_share").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("payout_ratio").dtype(FieldDtype::Str))
        .field(StructureFieldSpec::new("policy_notes").dtype(FieldDtype::List))
        .finish()
        .structure("forward_watchlist")
        .field(StructureFieldSpec::new("trigger").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("time_horizon").dtype(FieldDtype::List))
        .field(StructureFieldSpec::new("named_entity").dtype(FieldDtype::List))
        .finish()
        .build();

    let start = Instant::now();
    // Use `include_confidence=true` so we can compare signal axis confidences.
    let out = pipeline.extract_with_confidence(dummy_report, &schema, 0.5)?;
    println!("inference took: {:.2?}", start.elapsed());
    println!("-----------------");

    fn span_text(span: &FormattedEntitySpan) -> &str {
        match span {
            FormattedEntitySpan::Text(text) => text,
            FormattedEntitySpan::TextWithConfidence { text, .. } => text,
            FormattedEntitySpan::TextWithSpans { text, .. } => text,
            FormattedEntitySpan::TextWithConfidenceAndSpans { text, .. } => text,
        }
    }

    fn value_texts(value: &FormattedEntityValue) -> Vec<String> {
        match value {
            FormattedEntityValue::List(values) => {
                values.iter().map(|v| span_text(v).to_string()).collect()
            }
            FormattedEntityValue::Single(value) => value
                .as_ref()
                .map(|v| vec![span_text(v).to_string()])
                .unwrap_or_default(),
        }
    }

    fn normalize_list(mut values: Vec<String>) -> Vec<String> {
        values.sort();
        values.dedup();
        values
    }

    fn canonical_instance(instance: &BTreeMap<String, Vec<String>>) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in instance {
            let mut vv = v.clone();
            vv.sort();
            parts.push(format!("{k}=[{}]", vv.join("|")));
        }
        parts.join(";")
    }

    // --- Expected output (from the Python gist comment) ---
    let expected_sector_focus = "finance";

    let expected_entities: BTreeMap<&str, Vec<&str>> = BTreeMap::from([
        ("asset_manager", vec!["HarborView Capital"]),
        (
            "portfolio_company",
            vec!["Anchor Freight", "Seaside Renewables"],
        ),
        ("policy_signal", vec!["steady distributions"]),
        (
            "regulatory_body",
            vec!["maritime authority", "BlueCurrent Bank"],
        ),
        ("liquidity_cushion", vec!["$1.6B"]),
    ]);

    let expected_relations: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::from([
        (
            "allocates_to",
            vec![
                ("HarborView Capital", "Seaside Renewables"),
                ("HarborView Capital", "Anchor Freight"),
            ],
        ),
        ("overseen_by", vec![("revolver", "BlueCurrent Bank")]),
        ("funds_from", vec![("Anchor Freight", "Anchor Freight")]),
    ]);

    let expected_signal_axes: Vec<(&str, f32)> = vec![
        ("dividend_policy", 0.95),
        ("liquidity_guardrails", 0.69),
        ("regulatory_watch", 0.75),
        ("growth_investment", 0.65),
    ];

    let expected_dividend_outlook: Vec<BTreeMap<&str, Vec<&str>>> = vec![BTreeMap::from([
        ("record_date", vec!["March 18"]),
        ("cash_per_share", vec!["$0.34"]),
        ("payout_ratio", vec!["55% and 60%"]),
        ("policy_notes", vec![]),
    ])];

    let expected_forward_watchlist: Vec<BTreeMap<&str, Vec<&str>>> = vec![
        BTreeMap::from([
            ("trigger", vec!["unless regulators request a pause"]),
            ("time_horizon", vec![]),
            ("named_entity", vec!["dividend ladder"]),
        ]),
        BTreeMap::from([
            ("trigger", vec![]),
            ("time_horizon", vec![]),
            ("named_entity", vec!["execution on Asia feeder routes"]),
        ]),
    ];

    // --- Flatten & compare ---
    let got_sector_focus = match out.classifications.get("sector_focus") {
        Some(FormattedClassification::Single(label)) => Some(label.clone()),
        Some(FormattedClassification::SingleWithConfidence { label, .. }) => Some(label.clone()),
        _ => None,
    };

    let got_signal_axes: Vec<(String, f32)> = match out.classifications.get("signal_axes") {
        Some(FormattedClassification::MultiWithConfidence(values)) => values.clone(),
        Some(FormattedClassification::Multi(labels)) => {
            labels.iter().cloned().map(|l| (l, f32::NAN)).collect()
        }
        _ => Vec::new(),
    };

    let mut got_entities: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (label, value) in &out.entities {
        got_entities.insert(label.clone(), normalize_list(value_texts(value)));
    }

    let mut got_relations: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();
    for (rel, pairs) in &out.relations {
        let set: BTreeSet<(String, String)> = pairs
            .iter()
            .map(|p| {
                (
                    span_text(&p.head).to_string(),
                    span_text(&p.tail).to_string(),
                )
            })
            .collect();
        got_relations.insert(rel.clone(), set);
    }

    let got_dividend_outlook: Vec<BTreeMap<String, Vec<String>>> = out
        .structures
        .get("dividend_outlook")
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|inst| {
            inst.into_iter()
                .map(|(k, v)| (k, normalize_list(value_texts(&v))))
                .collect()
        })
        .collect();

    let got_forward_watchlist: Vec<BTreeMap<String, Vec<String>>> = out
        .structures
        .get("forward_watchlist")
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|inst| {
            inst.into_iter()
                .map(|(k, v)| (k, normalize_list(value_texts(&v))))
                .collect()
        })
        .collect();

    // --- Print + compare (Python expected vs Rust got) ---
    println!("Expected (Python) sector_focus: {expected_sector_focus}");
    println!("Got (Rust) sector_focus: {got_sector_focus:?}");
    println!(
        "sector_focus match: {}",
        got_sector_focus.as_deref() == Some(expected_sector_focus)
    );
    println!("-----------------");

    println!("Expected (Python) signal_axes (label, confidence): {expected_signal_axes:#?}");
    println!("Got (Rust) signal_axes (label, confidence): {got_signal_axes:#?}");
    let got_signal_map: BTreeMap<&str, f32> = got_signal_axes
        .iter()
        .map(|(l, c)| (l.as_str(), *c))
        .collect();
    for (label, expected_conf) in &expected_signal_axes {
        match got_signal_map.get(label) {
            Some(got_conf) => {
                let diff = (got_conf - expected_conf).abs();
                println!(
                    "signal_axes[{label}]: expected≈{expected_conf:.2}, got={got_conf:.4} (|Δ|={diff:.4})"
                );
            }
            None => println!("signal_axes[{label}]: MISSING (expected≈{expected_conf:.2})"),
        }
    }
    println!("-----------------");

    println!("Expected (Python) entities: {expected_entities:#?}");
    println!("Got (Rust) entities: {got_entities:#?}");
    for (label, expected_values) in &expected_entities {
        let expected: BTreeSet<String> = expected_values.iter().map(|s| s.to_string()).collect();
        let got: BTreeSet<String> = got_entities
            .get(*label)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let missing: Vec<_> = expected.difference(&got).cloned().collect();
        let extra: Vec<_> = got.difference(&expected).cloned().collect();
        println!("entities[{label}] missing: {missing:?} extra: {extra:?}");
    }
    println!("-----------------");

    println!("Expected (Python) relation_extraction: {expected_relations:#?}");
    let mut got_relations_as_vec: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (rel, pairs) in &got_relations {
        got_relations_as_vec.insert(rel.clone(), pairs.iter().cloned().collect());
    }
    println!("Got (Rust) relation_extraction: {got_relations_as_vec:#?}");
    for (rel, expected_pairs) in &expected_relations {
        let expected: BTreeSet<(String, String)> = expected_pairs
            .iter()
            .map(|(h, t)| ((*h).to_string(), (*t).to_string()))
            .collect();
        let got = got_relations.get(*rel).cloned().unwrap_or_default();
        let missing: Vec<_> = expected.difference(&got).cloned().collect();
        let extra: Vec<_> = got.difference(&expected).cloned().collect();
        println!("relations[{rel}] missing: {missing:?} extra: {extra:?}");
    }
    println!("-----------------");

    let expected_dividend_keys = expected_dividend_outlook
        .iter()
        .map(|inst| {
            inst.iter()
                .map(|(k, v)| {
                    (
                        (*k).to_string(),
                        v.iter().map(|s| (*s).to_string()).collect(),
                    )
                })
                .collect::<BTreeMap<String, Vec<String>>>()
        })
        .collect::<Vec<_>>();
    let expected_dividend_set: BTreeSet<String> = expected_dividend_keys
        .iter()
        .map(canonical_instance)
        .collect();
    let got_dividend_set: BTreeSet<String> = got_dividend_outlook
        .iter()
        .map(canonical_instance)
        .collect();
    println!("Expected (Python) dividend_outlook: {expected_dividend_keys:#?}");
    println!("Got (Rust) dividend_outlook: {got_dividend_outlook:#?}");
    println!(
        "dividend_outlook missing: {:?}",
        expected_dividend_set
            .difference(&got_dividend_set)
            .cloned()
            .collect::<Vec<_>>()
    );
    println!("-----------------");

    let expected_watch_keys = expected_forward_watchlist
        .iter()
        .map(|inst| {
            inst.iter()
                .map(|(k, v)| {
                    (
                        (*k).to_string(),
                        v.iter().map(|s| (*s).to_string()).collect(),
                    )
                })
                .collect::<BTreeMap<String, Vec<String>>>()
        })
        .collect::<Vec<_>>();
    let expected_watch_set: BTreeSet<String> =
        expected_watch_keys.iter().map(canonical_instance).collect();
    let got_watch_set: BTreeSet<String> = got_forward_watchlist
        .iter()
        .map(canonical_instance)
        .collect();
    println!("Expected (Python) forward_watchlist: {expected_watch_keys:#?}");
    println!("Got (Rust) forward_watchlist: {got_forward_watchlist:#?}");
    println!(
        "forward_watchlist missing: {:?}",
        expected_watch_set
            .difference(&got_watch_set)
            .cloned()
            .collect::<Vec<_>>()
    );

    Ok(())
}
