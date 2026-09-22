use std::{collections::BTreeMap, error::Error, fs, path::Path};

use gliner2_rs::boundary::record_decode::{
    Cardinality, DecodedRecord, FieldSpec, RecordGroup, RecordMode, decode_group, derive_count,
};
use ndarray::Array2;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    oracle: String,
    torch_version: String,
    assignment_backend: String,
    oracle_platform: String,
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    mode: String,
    fields: Vec<VectorField>,
    anchor_query_id: Option<usize>,
    object_logits: Vec<f32>,
    assign_logits: Vec<Vec<Vec<f32>>>,
    field_spans: Vec<Vec<[usize; 2]>>,
    instance_seed: Vec<Option<(usize, usize)>>,
    instance_spans: Vec<Option<[usize; 2]>>,
    anchor_threshold: f32,
    field_threshold: f32,
    object_threshold: f32,
    temperature: f32,
    expected: Vec<ExpectedRecord>,
}

#[derive(Debug, Deserialize)]
struct VectorField {
    query_id: usize,
    cardinality: String,
    exclusive: bool,
}

#[derive(Debug, Deserialize)]
struct ExpectedRecord {
    fields: Vec<ExpectedSpans>,
    field_scores: Vec<ExpectedScores>,
    anchor_span: Option<[usize; 2]>,
    score: f32,
}

#[derive(Debug, Deserialize)]
struct ExpectedSpans {
    query_id: usize,
    spans: Vec<[usize; 2]>,
}

#[derive(Debug, Deserialize)]
struct ExpectedScores {
    query_id: usize,
    scores: Vec<f32>,
}

fn vectors() -> Result<VectorFile, Box<dyn Error>> {
    let path = std::env::var_os("RECORD_DECODE_VECTOR_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/record-decode-vectors.json")
        });
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn mode(value: &str) -> Result<RecordMode, String> {
    match value {
        "natural" => Ok(RecordMode::Natural),
        "latent" => Ok(RecordMode::Latent),
        "anchorless" => Ok(RecordMode::Anchorless),
        other => Err(format!("unknown mode {other:?}")),
    }
}

fn cardinality(value: &str) -> Result<Cardinality, String> {
    match value {
        "optional_one" => Ok(Cardinality::OptionalOne),
        "required_one" => Ok(Cardinality::RequiredOne),
        "zero_or_more" => Ok(Cardinality::ZeroOrMore),
        "one_or_more" => Ok(Cardinality::OneOrMore),
        other => Err(format!("unknown cardinality {other:?}")),
    }
}

fn group(case: &VectorCase) -> Result<RecordGroup, Box<dyn Error>> {
    let fields: Vec<_> = case
        .fields
        .iter()
        .map(|field| {
            Ok(FieldSpec {
                query_id: field.query_id,
                cardinality: cardinality(&field.cardinality)?,
                exclusive: field.exclusive,
            })
        })
        .collect::<Result<_, String>>()?;
    let assignments: Vec<_> = case
        .assign_logits
        .iter()
        .enumerate()
        .map(|(field, rows)| {
            let columns = case.field_spans[field].len() + 1;
            let values: Vec<_> = rows.iter().flatten().copied().collect();
            Array2::from_shape_vec((case.object_logits.len(), columns), values)
        })
        .collect::<Result<_, ndarray::ShapeError>>()?;
    Ok(RecordGroup {
        mode: mode(&case.mode)?,
        field_query_ids: fields.iter().map(|field| field.query_id).collect(),
        field_specs: fields,
        natural_anchor_query_id: case.anchor_query_id,
        object_logits: case.object_logits.clone(),
        assign_logits: assignments,
        field_spans: case.field_spans.clone(),
        instance_seed: case.instance_seed.clone(),
        instance_spans: case.instance_spans.clone(),
    })
}

fn decoded_expected(record: &ExpectedRecord) -> DecodedRecord {
    DecodedRecord {
        fields: record
            .fields
            .iter()
            .map(|entry| (entry.query_id, entry.spans.clone()))
            .collect(),
        field_scores: record
            .field_scores
            .iter()
            .map(|entry| (entry.query_id, entry.scores.clone()))
            .collect(),
        anchor_span: record.anchor_span,
        score: record.score,
    }
}

fn assert_score(actual: f32, expected: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "{context}: {actual:?} != oracle {expected:?}"
    );
}

#[test]
fn real_upstream_record_group_vectors_match_spans_order_owners_and_scores()
-> Result<(), Box<dyn Error>> {
    let vectors = vectors()?;
    assert_eq!(vectors.format_version, 2);
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert_eq!(
        vectors.oracle,
        "unchanged pinned GLiNER2 RecordGroupOutput/RecordSpec decode_group"
    );
    assert_eq!(vectors.torch_version, "2.8.0");
    assert_eq!(
        vectors.assignment_backend,
        "internal_shortest_augmenting_path_scipy_absent"
    );
    assert_eq!(vectors.oracle_platform, "darwin-arm64");
    assert_eq!(vectors.cases.len(), 17);

    for case in vectors.cases {
        let actual = decode_group(
            &group(&case)?,
            case.anchor_threshold,
            case.field_threshold,
            case.object_threshold,
            case.temperature,
        )?;
        assert_eq!(
            derive_count(&actual),
            case.expected.len(),
            "{} count",
            case.name
        );
        assert_eq!(actual.len(), case.expected.len(), "{} records", case.name);
        for (record_index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
            let expected = decoded_expected(expected);
            assert_eq!(
                actual.fields, expected.fields,
                "{} record {record_index} exact fields/spans/order",
                case.name
            );
            assert_eq!(
                actual.anchor_span, expected.anchor_span,
                "{} record {record_index} anchor",
                case.name
            );
            assert_eq!(
                actual.field_scores.keys().collect::<Vec<_>>(),
                expected.field_scores.keys().collect::<Vec<_>>(),
                "{} record {record_index} score owners",
                case.name
            );
            for (query_id, actual_scores) in &actual.field_scores {
                let expected_scores = &expected.field_scores[query_id];
                assert_eq!(
                    actual_scores.len(),
                    expected_scores.len(),
                    "{} record {record_index} q{query_id} score count",
                    case.name
                );
                for (score_index, (&actual, &expected)) in
                    actual_scores.iter().zip(expected_scores).enumerate()
                {
                    assert_score(
                        actual,
                        expected,
                        &format!(
                            "{} record {record_index} q{query_id} score {score_index}",
                            case.name
                        ),
                    );
                }
            }
            assert_score(
                actual.score,
                expected.score,
                &format!("{} record {record_index} object score", case.name),
            );
        }
    }
    Ok(())
}

fn valid_group() -> RecordGroup {
    RecordGroup {
        mode: RecordMode::Latent,
        field_specs: vec![FieldSpec {
            query_id: 7,
            cardinality: Cardinality::OptionalOne,
            exclusive: false,
        }],
        field_query_ids: vec![7],
        natural_anchor_query_id: None,
        object_logits: vec![1.0],
        assign_logits: vec![Array2::zeros((1, 2))],
        field_spans: vec![vec![[0, 1]]],
        instance_seed: vec![None],
        instance_spans: vec![None],
    }
}

#[test]
fn malformed_shapes_and_nonfinite_logits_are_errors() {
    let mut malformed = valid_group();
    malformed.assign_logits[0] = Array2::zeros((1, 3));
    assert!(
        decode_group(&malformed, 0.5, 0.5, 0.5, 1.0)
            .unwrap_err()
            .to_string()
            .contains("shape")
    );

    let mut object_nan = valid_group();
    object_nan.object_logits[0] = f32::NAN;
    assert!(
        decode_group(&object_nan, 0.5, 0.5, 0.5, 1.0)
            .unwrap_err()
            .to_string()
            .contains("object_logits")
    );

    let mut assignment_nan = valid_group();
    assignment_nan.assign_logits[0][(0, 1)] = f32::NAN;
    assert!(
        decode_group(&assignment_nan, 0.5, 0.5, 0.5, 1.0)
            .unwrap_err()
            .to_string()
            .contains("assign_logits[0]")
    );

    let mut object_overflow = valid_group();
    object_overflow.object_logits[0] = f32::MAX;
    let error = decode_group(&object_overflow, 0.5, 0.5, 0.5, f32::MIN_POSITIVE)
        .unwrap_err()
        .to_string();
    assert!(error.contains("object_logits[0] / temperature"), "{error}");

    let mut assignment_overflow = valid_group();
    assignment_overflow.assign_logits[0][(0, 1)] = f32::MAX;
    let error = decode_group(&assignment_overflow, 0.5, 0.5, 0.5, f32::MIN_POSITIVE)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("assign_logits[0][0, 1] / temperature"),
        "{error}"
    );

    let mut mismatched_query = valid_group();
    mismatched_query.field_query_ids[0] = 8;
    assert!(decode_group(&mismatched_query, 0.5, 0.5, 0.5, 1.0).is_err());

    let mut short_seeds = valid_group();
    short_seeds.instance_seed.clear();
    assert!(decode_group(&short_seeds, 0.5, 0.5, 0.5, 1.0).is_err());
}

#[test]
fn thresholds_and_temperature_are_validated() {
    let group = valid_group();
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(decode_group(&group, 0.5, 0.5, 0.5, temperature).is_err());
    }
    for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert!(decode_group(&group, threshold, 0.5, 0.5, 1.0).is_err());
        assert!(decode_group(&group, 0.5, threshold, 0.5, 1.0).is_err());
        assert!(decode_group(&group, 0.5, 0.5, threshold, 1.0).is_err());
    }
    decode_group(&group, 0.0, 0.0, 0.0, f32::MIN_POSITIVE).unwrap();
    decode_group(&group, 1.0, 1.0, 1.0, 1.0).unwrap();
}

#[test]
fn mode_anchor_contract_is_checked_without_schema_dependencies() {
    let mut natural = valid_group();
    natural.mode = RecordMode::Natural;
    assert!(decode_group(&natural, 0.5, 0.5, 0.5, 1.0).is_err());

    natural.natural_anchor_query_id = Some(99);
    assert!(decode_group(&natural, 0.5, 0.5, 0.5, 1.0).is_err());

    let mut anchorless = valid_group();
    anchorless.mode = RecordMode::Anchorless;
    anchorless.natural_anchor_query_id = Some(7);
    assert!(decode_group(&anchorless, 0.5, 0.5, 0.5, 1.0).is_err());
}

#[test]
fn decoded_record_maps_are_query_sorted_but_span_vectors_keep_source_order() {
    let record = DecodedRecord {
        fields: BTreeMap::from([(9, vec![[4, 5], [0, 1]]), (2, vec![[8, 9]])]),
        field_scores: BTreeMap::new(),
        anchor_span: None,
        score: 1.0,
    };
    assert_eq!(
        record.fields.keys().copied().collect::<Vec<_>>(),
        vec![2, 9]
    );
    assert_eq!(record.fields[&9], vec![[4, 5], [0, 1]]);
}
