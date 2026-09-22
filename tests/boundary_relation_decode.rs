use std::{env, fs, path::PathBuf};

use anyhow::ensure;
use gliner2_rs::boundary::relation_decode::{
    RelationEdge, RelationMention, deduplicate_relation_edges,
};
use serde::Deserialize;

fn occurrence(text: &str, value: &str, occurrence: usize) -> (usize, usize) {
    let (start, _) = text
        .match_indices(value)
        .nth(occurrence)
        .unwrap_or_else(|| panic!("missing occurrence {occurrence} of {value:?} in {text:?}"));
    (start, start + value.len())
}

fn mention(text: &str, value: &str, occurrence_index: usize) -> RelationMention {
    let (start, end) = occurrence(text, value, occurrence_index);
    RelationMention {
        text: value.to_owned(),
        start,
        end,
    }
}

fn edge(head: RelationMention, tail: RelationMention, score: f32) -> RelationEdge {
    RelationEdge { head, tail, score }
}

#[test]
fn empty_and_single_edges_are_preserved_after_validation() {
    assert!(deduplicate_relation_edges("", &[]).unwrap().is_empty());

    let text = "Alice--Bob";
    let one = edge(mention(text, "Alice", 0), mention(text, "Bob", 0), 0.5);
    assert_eq!(
        deduplicate_relation_edges(text, std::slice::from_ref(&one)).unwrap(),
        vec![one]
    );
}

#[test]
fn containing_mentions_use_codepoint_length_before_exact_score_dedup() {
    // "aaaaX" is five codepoints/five bytes, while "X😀😀" is three
    // codepoints/nine bytes. Both contain X, so Python chooses "aaaaX".
    let text = "aaaaX😀😀 -> T";
    let tail = mention(text, "T", 0);
    let edges = vec![
        edge(mention(text, "X", 0), tail.clone(), 0.9),
        edge(mention(text, "aaaaX", 0), tail.clone(), 0.1),
        edge(mention(text, "X😀😀", 0), tail, 0.2),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].head.text, "aaaaX");
    assert_eq!(actual[0].score, 0.9);
    assert_eq!(actual[1].head.text, "X😀😀");
    assert_eq!(actual[1].score, 0.2);
}

#[test]
fn canonical_equal_length_ties_choose_earlier_codepoint_start() {
    let text = "abcdef -> T";
    let tail = mention(text, "T", 0);
    let edges = vec![
        edge(mention(text, "cd", 0), tail.clone(), 0.9),
        edge(mention(text, "abcde", 0), tail.clone(), 0.1),
        edge(mention(text, "bcdef", 0), tail, 0.2),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].head.text, "abcde");
    assert_eq!(actual[0].score, 0.9);
    assert_eq!(actual[1].head.text, "bcdef");
    assert_eq!(actual[1].score, 0.2);
}

#[test]
fn canonical_overlap_normalizes_before_strictly_higher_exact_replacement() {
    let text = "The New York office hired Ada";
    let ada = mention(text, "Ada", 0);
    let edges = vec![
        edge(mention(text, "York", 0), ada.clone(), 0.8),
        edge(mention(text, "New York", 0), ada.clone(), 0.4),
        edge(mention(text, "New York", 0), ada, 0.8),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].head.text, "New York");
    assert_eq!(actual[0].score, 0.8);
}

#[test]
fn semantic_dedup_uses_codepoint_gap_not_utf8_byte_gap() {
    let text = "H你你T H.....T";
    let edges = vec![
        edge(mention(text, "H", 0), mention(text, "T", 0), 0.2),
        edge(mention(text, "H", 1), mention(text, "T", 1), 0.9),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].head.start, 0);
    assert_eq!(actual[0].tail.start, occurrence(text, "T", 0).0);
    assert_eq!(actual[0].score, 0.2);
}

#[test]
fn semantic_dedup_full_casefolds_sharp_s_and_prefers_score_after_gap() {
    let text = "Straße--X STRASSE--x";
    let edges = vec![
        edge(mention(text, "Straße", 0), mention(text, "X", 0), 0.3),
        edge(mention(text, "STRASSE", 0), mention(text, "x", 0), 0.8),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].head.text, "STRASSE");
    assert_eq!(actual[0].score, 0.8);
}

#[test]
fn python_control_whitespace_is_used_for_trim_and_semantic_split() {
    let padded = "\u{1c}Alpha\u{1f}--Beta";
    let head_end = padded.find("--").unwrap();
    let padded_edge = edge(
        RelationMention {
            text: "Alpha".to_owned(),
            start: 0,
            end: head_end,
        },
        mention(padded, "Beta", 0),
        0.5,
    );
    assert_eq!(
        deduplicate_relation_edges(padded, std::slice::from_ref(&padded_edge)).unwrap(),
        vec![padded_edge]
    );

    let text = "ALPHA\u{1c}BETA--X alpha beta--x";
    let edges = vec![
        edge(
            mention(text, "ALPHA\u{1c}BETA", 0),
            mention(text, "X", 0),
            0.3,
        ),
        edge(mention(text, "alpha beta", 0), mention(text, "x", 0), 0.7),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].head.text, "alpha beta");
}

#[test]
fn strict_token_subsets_are_dominated_only_with_equal_opposite_tokens() {
    let text = "York--Acme New York--ACME York--Other";
    let edges = vec![
        edge(mention(text, "York", 0), mention(text, "Acme", 0), 0.9),
        edge(mention(text, "New York", 0), mention(text, "ACME", 0), 0.2),
        edge(mention(text, "York", 1), mention(text, "Other", 0), 0.1),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].head.text, "New York");
    assert_eq!(actual[0].tail.text, "ACME");
    assert_eq!(actual[1].tail.text, "Other");
}

#[test]
fn final_order_is_stable_head_then_tail_then_descending_score() {
    let text = "A B C D";
    let edges = vec![
        edge(mention(text, "C", 0), mention(text, "D", 0), 0.9),
        edge(mention(text, "A", 0), mention(text, "D", 0), 0.2),
        edge(mention(text, "A", 0), mention(text, "B", 0), 0.1),
    ];
    let actual = deduplicate_relation_edges(text, &edges).unwrap();
    let coordinates: Vec<_> = actual
        .iter()
        .map(|edge| (edge.head.start, edge.tail.start, edge.score))
        .collect();
    assert_eq!(coordinates, vec![(0, 2, 0.1), (0, 6, 0.2), (4, 6, 0.9)]);
}

#[test]
fn malformed_offsets_surfaces_and_scores_fail_closed() {
    let text = "éx";
    let valid_head = RelationMention {
        text: "é".to_owned(),
        start: 0,
        end: 2,
    };
    let valid_tail = RelationMention {
        text: "x".to_owned(),
        start: 2,
        end: 3,
    };
    let invalid_mentions = [
        RelationMention {
            text: String::new(),
            start: 0,
            end: 0,
        },
        RelationMention {
            text: "éx".to_owned(),
            start: 0,
            end: 4,
        },
        RelationMention {
            text: "?".to_owned(),
            start: 1,
            end: 2,
        },
        RelationMention {
            text: "wrong".to_owned(),
            start: 0,
            end: 2,
        },
    ];
    for invalid in invalid_mentions {
        assert!(
            deduplicate_relation_edges(text, &[edge(invalid, valid_tail.clone(), 0.5)]).is_err()
        );
    }
    for score in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(
            deduplicate_relation_edges(
                text,
                &[edge(valid_head.clone(), valid_tail.clone(), score)]
            )
            .is_err()
        );
    }
}

#[derive(Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    python_version: String,
    oracle: String,
    engine_sha256: String,
    method_sha256: String,
    cases: Vec<VectorCase>,
}

#[derive(Deserialize)]
struct VectorCase {
    name: String,
    text: String,
    edges: Vec<VectorEdge>,
    expected: Vec<VectorEdge>,
}

#[derive(Deserialize)]
struct VectorEdge {
    head: VectorMention,
    tail: VectorMention,
    score: f32,
}

#[derive(Deserialize)]
struct VectorMention {
    text: String,
    start: usize,
    end: usize,
}

impl From<&VectorMention> for RelationMention {
    fn from(value: &VectorMention) -> Self {
        Self {
            text: value.text.clone(),
            start: value.start,
            end: value.end,
        }
    }
}

impl From<&VectorEdge> for RelationEdge {
    fn from(value: &VectorEdge) -> Self {
        Self {
            head: (&value.head).into(),
            tail: (&value.tail).into(),
            score: value.score,
        }
    }
}

#[test]
fn unchanged_upstream_full_vectors_match_exactly_when_available() -> anyhow::Result<()> {
    let path = env::var_os("GLINER2_RELATION_DECODE_VECTORS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("GLINER2_TEST_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
                .join("fixtures/gliner2.5-base-v1/relation-aux/relation-decode-vectors.json")
        });
    if !path.is_file() {
        ensure!(
            env::var("GLINER2_BOUNDARY_FULL_FIXTURES").as_deref() != Ok("1"),
            "GLINER2_BOUNDARY_FULL_FIXTURES=1 requires relation decode vectors at {}",
            path.display()
        );
        eprintln!(
            "SKIP: optional unchanged-upstream relation decode vectors are absent at {}",
            path.display()
        );
        return Ok(());
    }

    let vectors: VectorFile = serde_json::from_slice(&fs::read(&path)?)?;
    assert_eq!(vectors.format_version, 1);
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert_eq!(vectors.python_version, "3.12.7");
    assert_eq!(
        vectors.oracle,
        "unchanged gliner2.models.boundary.engine.BoundaryExtractor._deduplicate_relation_edges"
    );
    assert_eq!(
        vectors.engine_sha256,
        "80a250e2454ae8b3ae832c98d93d1bedf84113b76b113b42e2eff813313d6595"
    );
    assert_eq!(
        vectors.method_sha256,
        "26274b4c3fde2efeb0cb5cd50e12a5b1d251ecf79979cd9eef2e2249c12bb018"
    );
    let expected_ids = [
        "empty",
        "single",
        "canonical_overlap_exact_higher_and_equal_tie",
        "canonical_equal_length_tie_earlier_start",
        "canonical_codepoint_width_not_byte_width",
        "repeated_semantic_occurrence_nearest",
        "semantic_codepoint_gap_not_byte_gap",
        "full_casefold_sharp_s",
        "python_control_whitespace_surface_trim",
        "python_control_whitespace_semantic_split",
        "strict_head_token_subset_with_equal_opposite",
        "strict_tail_token_subset_with_equal_opposite",
        "semantic_rank_first_exact_tie",
        "final_stable_coordinate_score_order",
    ];
    assert_eq!(
        vectors
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        expected_ids,
        "relation decode oracle case IDs must fail closed"
    );

    for case in vectors.cases {
        let input: Vec<_> = case.edges.iter().map(RelationEdge::from).collect();
        let expected: Vec<_> = case.expected.iter().map(RelationEdge::from).collect();
        let actual = deduplicate_relation_edges(&case.text, &input)?;
        assert_eq!(actual, expected, "{}", case.name);
    }
    Ok(())
}
