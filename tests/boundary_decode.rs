use std::{collections::BTreeMap, fs, fs::File, path::Path};

use anyhow::{Context, Result, ensure};
use gliner2_rs::boundary::decode::{
    DecodedValue, FieldDtype, OverlapPolicy, QueryScores, ScoredSpan, WordOffset, decode_to_utf8,
    group_scored_candidates, map_half_open_utf8, normalize_overlap_policy,
    passes_confidence_threshold, resolve_overlaps, select_field_value, should_abstain,
};
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzReader;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    overlap_cases: Vec<OverlapCase>,
    grouping_cases: Vec<GroupingCase>,
    entity_cases: Vec<EntityCase>,
}

#[derive(Debug, Deserialize)]
struct OverlapCase {
    name: String,
    policy: String,
    spans: Vec<(f32, usize, usize)>,
    expected: Vec<(f32, usize, usize)>,
}

#[derive(Debug, Deserialize)]
struct GroupingCase {
    name: String,
    indices: Vec<Vec<[usize; 2]>>,
    valid_mask: Vec<Vec<bool>>,
    query_mask: Vec<bool>,
    pair_logits: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
    temperature: f32,
    expected: Vec<Vec<(f32, usize, usize)>>,
}

#[derive(Debug, Deserialize)]
struct EntityCase {
    name: String,
    text: String,
    offsets: Vec<[usize; 2]>,
    choice_prefix_words: usize,
    dtype: String,
    null_probability: f32,
    policy: String,
    scored: Vec<(f32, usize, usize)>,
    expected: Vec<ExpectedEntity>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    text: String,
    confidence: f32,
    start: usize,
    end: usize,
}

fn vectors() -> Result<VectorFile> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/boundary-decode-vectors.json");
    serde_json::from_slice(&fs::read(&path).with_context(|| path.display().to_string())?)
        .context("parse boundary decode vectors")
}

fn scored(rows: &[(f32, usize, usize)]) -> Vec<ScoredSpan> {
    rows.iter()
        .map(|&(confidence, start, end)| ScoredSpan {
            confidence,
            start,
            end,
        })
        .collect()
}

fn assert_confidence(actual: f32, expected: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "{context}: confidence {actual:?} != oracle {expected:?}"
    );
}

#[test]
fn pinned_overlap_oracles_match_exactly() -> Result<()> {
    let vectors = vectors()?;
    assert_eq!(vectors.format_version, 1);
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert!(vectors.overlap_cases.len() >= 8);
    for case in vectors.overlap_cases {
        let actual = resolve_overlaps(&scored(&case.spans), case.policy.parse()?)?;
        let expected = scored(&case.expected);
        assert_eq!(actual, expected, "{}", case.name);
    }
    Ok(())
}

#[test]
fn pinned_grouping_oracles_preserve_masks_thresholds_and_order() -> Result<()> {
    for case in vectors()?.grouping_cases {
        let queries: Vec<_> = (0..case.query_mask.len())
            .map(|query| QueryScores {
                indices: case.indices[query].clone(),
                valid_mask: case.valid_mask[query].clone(),
                pair_logits: case.pair_logits[query].clone(),
                query_valid: case.query_mask[query],
                threshold: case.thresholds[query],
                null_logit: None,
            })
            .collect();
        let actual = group_scored_candidates(&queries, case.temperature, 0.5)?;
        assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
        for (query, (actual_query, expected_query)) in actual.iter().zip(&case.expected).enumerate()
        {
            assert_eq!(
                actual_query.len(),
                expected_query.len(),
                "{} q{query}",
                case.name
            );
            for (candidate, (actual_span, expected_span)) in
                actual_query.iter().zip(expected_query).enumerate()
            {
                assert_eq!(
                    (actual_span.start, actual_span.end),
                    (expected_span.1, expected_span.2),
                    "{} q{query} c{candidate}",
                    case.name
                );
                assert_confidence(
                    actual_span.confidence,
                    expected_span.0,
                    &format!("{} q{query} c{candidate}", case.name),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn pinned_engine_entities_match_choice_unicode_null_and_dtype() -> Result<()> {
    for case in vectors()?.entity_cases {
        let offsets: Vec<_> = case
            .offsets
            .iter()
            .map(|offset| WordOffset {
                start: offset[0],
                end: offset[1],
            })
            .collect();
        let mut actual = if should_abstain(case.null_probability, 0.5)? {
            Vec::new()
        } else {
            decode_to_utf8(
                &case.text,
                &offsets,
                &scored(&case.scored),
                case.policy.parse()?,
                case.choice_prefix_words,
            )?
        };
        let dtype = match case.dtype.as_str() {
            "list" => FieldDtype::List,
            "str" => FieldDtype::Scalar,
            other => panic!("{}: unknown dtype {other}", case.name),
        };
        actual = match select_field_value(actual, dtype) {
            DecodedValue::List(items) => items,
            DecodedValue::Scalar(item) => item.into_iter().collect(),
        };
        assert_eq!(actual.len(), case.expected.len(), "{}", case.name);
        for (index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
            assert_eq!(actual.text, expected.text, "{} item {index}", case.name);
            assert_eq!(
                (actual.start, actual.end),
                (expected.start, expected.end),
                "{} item {index}",
                case.name
            );
            assert_confidence(
                actual.confidence,
                expected.confidence,
                &format!("{} item {index}", case.name),
            );
            assert_eq!(
                &case.text[actual.start..actual.end],
                case.text
                    .get(actual.start..actual.end)
                    .expect("verified UTF-8 boundaries")
            );
        }
    }
    Ok(())
}

#[test]
fn aliases_and_gate_boundaries_match_upstream() -> Result<()> {
    for alias in ["allow", "all", "none"] {
        assert_eq!(alias.parse::<OverlapPolicy>()?, OverlapPolicy::Allow);
    }
    for alias in ["nested", "allow_nested", "allow-nested"] {
        assert_eq!(alias.parse::<OverlapPolicy>()?, OverlapPolicy::Nested);
    }
    for alias in ["flat", "disallow", "no_overlap", "non-overlapping"] {
        assert_eq!(alias.parse::<OverlapPolicy>()?, OverlapPolicy::Disallow);
    }
    for alias in ["longest", "keep_longest", "keep-longest"] {
        assert_eq!(alias.parse::<OverlapPolicy>()?, OverlapPolicy::Longest);
    }
    assert_eq!(
        normalize_overlap_policy(None, Some("flat"))?,
        OverlapPolicy::Disallow
    );
    assert!(normalize_overlap_policy(None, None).is_err());
    assert!("greedy".parse::<OverlapPolicy>().is_err());

    assert!(passes_confidence_threshold(0.5, 0.5)?);
    assert!(!passes_confidence_threshold(0.499_999, 0.5)?);
    assert!(!should_abstain(0.5, 0.5)?);
    assert!(should_abstain(0.500_001, 0.5)?);
    Ok(())
}

#[test]
fn malformed_shapes_ranges_and_nonfinite_values_are_contextual_errors() {
    let mismatched = QueryScores {
        indices: vec![[0, 1]],
        valid_mask: vec![],
        pair_logits: vec![0.0],
        query_valid: true,
        threshold: 0.5,
        null_logit: None,
    };
    let error = group_scored_candidates(&[mismatched], 1.0, 0.5)
        .unwrap_err()
        .to_string();
    assert!(error.contains("query 0") && error.contains("valid_mask"));

    let malformed = QueryScores {
        indices: vec![[2, 2]],
        valid_mask: vec![true],
        pair_logits: vec![0.0],
        query_valid: true,
        threshold: 0.5,
        null_logit: None,
    };
    assert!(group_scored_candidates(&[malformed], 1.0, 0.5).is_err());
    assert!(group_scored_candidates(&[], 1.0, 0.5).unwrap().is_empty());

    let nonfinite = ScoredSpan {
        confidence: f32::NAN,
        start: 0,
        end: 1,
    };
    assert!(resolve_overlaps(&[nonfinite], OverlapPolicy::Allow).is_err());
    assert!(passes_confidence_threshold(f32::INFINITY, 0.5).is_err());
    assert!(group_scored_candidates(&[], 0.0, 0.5).is_err());
}

#[test]
fn utf8_mapping_rejects_invalid_original_text_boundaries_and_suffixes() -> Result<()> {
    let text = "é東京😀";
    let offsets = [
        WordOffset { start: 0, end: 2 },
        WordOffset { start: 2, end: 8 },
        WordOffset { start: 8, end: 12 },
    ];
    let mapped = map_half_open_utf8(text, &offsets, 0, 3)?;
    assert_eq!((mapped.start, mapped.end), (0, text.len()));
    assert_eq!(mapped.text, text);

    let inside_codepoint = [WordOffset { start: 1, end: 2 }];
    assert!(map_half_open_utf8(text, &inside_codepoint, 0, 1).is_err());
    let synthetic_suffix = [WordOffset {
        start: 0,
        end: text.len() + 1,
    }];
    assert!(map_half_open_utf8(text, &synthetic_suffix, 0, 1).is_err());
    assert!(map_half_open_utf8(text, &offsets, 2, 2).is_err());
    assert!(map_half_open_utf8(text, &offsets, 0, 4).is_err());

    let filtered = decode_to_utf8(
        text,
        &offsets,
        &[
            ScoredSpan {
                confidence: 0.9,
                start: 0,
                end: 1,
            },
            ScoredSpan {
                confidence: 0.8,
                start: 1,
                end: 5,
            },
            ScoredSpan {
                confidence: 0.7,
                start: 1,
                end: 2,
            },
        ],
        OverlapPolicy::Allow,
        1,
    )?;
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].text, "é");
    Ok(())
}

fn codepoint_to_byte(text: &str, codepoint: usize) -> Result<usize> {
    if codepoint == text.chars().count() {
        return Ok(text.len());
    }
    text.char_indices()
        .nth(codepoint)
        .map(|(byte, _)| byte)
        .with_context(|| format!("code-point offset {codepoint} is outside text"))
}

#[test]
fn committed_pair_and_null_stage_reproduces_unicode_golden_entities() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/gliner2.5-base-v1-subset");
    let json_path = root.join("unicode_combining_emoji.json");
    let metadata: Value = serde_json::from_slice(&fs::read(&json_path)?)?;
    let text = metadata["original_text"]
        .as_str()
        .context("golden original_text")?;
    let expected_entities = metadata["final_result_utf8_offsets"]["entities"]
        .as_object()
        .context("golden entities")?;
    let query_metadata = metadata["query_metadata"]
        .as_array()
        .context("golden query metadata")?;

    let npz_path = root.join("unicode_combining_emoji.npz");
    let mut npz = NpzReader::new(File::open(&npz_path)?)?;
    let indices: Array4<i64> = npz.by_name("boundary_head_0_indices")?;
    let valid: Array3<bool> = npz.by_name("boundary_head_0_valid_mask")?;
    let pair_logits: Array3<f32> = npz.by_name("boundary_head_0_pair_logits")?;
    let query_mask: Array2<bool> = npz.by_name("boundary_head_0_query_mask")?;
    let null_logits: Array2<f32> = npz.by_name("boundary_head_0_null_logits")?;
    let starts: Array1<i64> = npz.by_name("start_mappings")?;
    let ends: Array1<i64> = npz.by_name("end_mappings")?;
    ensure!(
        starts.len() == ends.len(),
        "golden mappings differ in length"
    );
    let offsets: Vec<_> = starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| {
            Ok(WordOffset {
                start: codepoint_to_byte(text, usize::try_from(start)?)?,
                end: codepoint_to_byte(text, usize::try_from(end)?)?,
            })
        })
        .collect::<Result<_>>()?;

    let (_, query_count, candidate_count, coordinate_count) = indices.dim();
    ensure!(coordinate_count == 2, "golden candidate coordinates != 2");
    let queries: Vec<_> = (0..query_count)
        .map(|query| {
            let query_indices: Result<Vec<_>> = (0..candidate_count)
                .map(|candidate| {
                    Ok([
                        usize::try_from(indices[(0, query, candidate, 0)])?,
                        usize::try_from(indices[(0, query, candidate, 1)])?,
                    ])
                })
                .collect();
            Ok(QueryScores {
                indices: query_indices?,
                valid_mask: (0..candidate_count)
                    .map(|candidate| valid[(0, query, candidate)])
                    .collect(),
                pair_logits: (0..candidate_count)
                    .map(|candidate| pair_logits[(0, query, candidate)])
                    .collect(),
                query_valid: query_mask[(0, query)],
                threshold: 0.5,
                null_logit: Some(null_logits[(0, query)]),
            })
        })
        .collect::<Result<_>>()?;
    let grouped = group_scored_candidates(&queries, 1.0, 0.5)?;

    let mut actual = BTreeMap::new();
    for (query, candidates) in grouped.iter().enumerate() {
        let label = query_metadata[query]["field_path"][0]
            .as_str()
            .context("query field path")?;
        actual.insert(
            label,
            decode_to_utf8(text, &offsets, candidates, OverlapPolicy::Disallow, 0)?,
        );
    }
    for (label, expected) in expected_entities {
        let expected = expected.as_array().context("expected entity list")?;
        let actual = actual.get(label.as_str()).context("actual entity label")?;
        assert_eq!(actual.len(), expected.len(), "label {label}");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.text,
                expected["text"].as_str().unwrap(),
                "{label}[{index}]"
            );
            assert_eq!(actual.start, expected["start"].as_u64().unwrap() as usize);
            assert_eq!(actual.end, expected["end"].as_u64().unwrap() as usize);
            assert_confidence(
                actual.confidence,
                expected["confidence"].as_f64().unwrap() as f32,
                &format!("{label}[{index}]"),
            );
        }
    }
    Ok(())
}
