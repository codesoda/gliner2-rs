use std::{fs, fs::File, path::Path};

use gliner2_rs::boundary::preprocessing::{
    BoundaryPreprocessingPolicy, DEFAULT_MAX_LEN, OriginalCoverage, OriginalTokenOffset,
    WordSplitter,
};
use ndarray::Array1;
use ndarray_npy::NpzReader;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct VectorFile {
    format_version: u32,
    upstream_commit: String,
    offset_unit: String,
    cases: Vec<VectorCase>,
}

#[derive(Debug, Deserialize)]
struct VectorCase {
    name: String,
    text: String,
    splitter: String,
    max_len: usize,
    choice_prefix: Vec<String>,
    choice_prefix_words: usize,
    normalized_text: String,
    synthetic_suffix_added: bool,
    text_tokens: Vec<String>,
    original_offsets_utf8: Vec<VectorOffset>,
}

#[derive(Debug, Deserialize)]
struct VectorOffset {
    start: usize,
    end: usize,
    coverage: String,
}

fn coverage(value: &str) -> OriginalCoverage {
    match value {
        "original" => OriginalCoverage::Original,
        "original_with_synthetic_suffix" => OriginalCoverage::OriginalWithSyntheticSuffix,
        "synthetic_only" => OriginalCoverage::SyntheticOnly,
        other => panic!("unknown fixture coverage {other:?}"),
    }
}

#[test]
fn pinned_upstream_token_vectors_match() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/boundary-token-vectors.json");
    let vectors: VectorFile = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(vectors.format_version, 1);
    assert_eq!(vectors.offset_unit, "utf8_byte");
    assert_eq!(
        vectors.upstream_commit,
        "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
    );
    assert!(vectors.cases.len() >= 13);

    for case in vectors.cases {
        let splitter = case.splitter.parse::<WordSplitter>().unwrap();
        let policy = BoundaryPreprocessingPolicy::new(case.max_len, splitter).unwrap();
        let prepared = policy.prepare(&case.text, &case.choice_prefix);
        let expected_offsets = case
            .original_offsets_utf8
            .iter()
            .map(|offset| OriginalTokenOffset {
                start: offset.start,
                end: offset.end,
                coverage: coverage(&offset.coverage),
            })
            .collect::<Vec<_>>();

        assert_eq!(prepared.original_text, case.text, "{} original", case.name);
        assert_eq!(
            prepared.normalized_text, case.normalized_text,
            "{} normalized",
            case.name
        );
        assert_eq!(
            prepared.text_tokens, case.text_tokens,
            "{} tokens",
            case.name
        );
        assert_eq!(
            prepared.original_offsets, expected_offsets,
            "{} offsets",
            case.name
        );
        assert_eq!(
            prepared.choice_prefix_words, case.choice_prefix_words,
            "{} prefix",
            case.name
        );
        assert_eq!(
            prepared.synthetic_suffix_added, case.synthetic_suffix_added,
            "{} suffix",
            case.name
        );
        assert!(
            prepared
                .original_offsets
                .iter()
                .all(|offset| offset.end <= case.text.len())
        );
    }
}

#[test]
fn policy_defaults_and_rejects_nonpositive_caps_and_unknown_splitters() {
    let policy = BoundaryPreprocessingPolicy::default();
    assert_eq!(policy.max_len(), DEFAULT_MAX_LEN);
    assert_eq!(policy.splitter(), WordSplitter::Whitespace);
    assert_eq!(
        BoundaryPreprocessingPolicy::from_optional_max_len(None, WordSplitter::Char)
            .unwrap()
            .max_len(),
        DEFAULT_MAX_LEN
    );
    assert!(BoundaryPreprocessingPolicy::new(0, WordSplitter::Whitespace).is_err());
    assert!("characters".parse::<WordSplitter>().is_err());
}

#[test]
fn cap_applies_before_prefix_and_never_to_combined_length() {
    let policy = BoundaryPreprocessingPolicy::new(2, WordSplitter::Whitespace).unwrap();
    let prefix = vec!["(".into(), "kind:".into(), "red".into(), ")".into()];
    let prepared = policy.prepare("one two three", &prefix);

    assert_eq!(
        prepared.text_tokens,
        ["(", "kind:", "red", ")", "one", "two"]
    );
    assert_eq!(prepared.choice_prefix_words, 4);
    assert_eq!(prepared.original_token_count(), 2);
    assert_eq!(prepared.start_mappings(), [0, 4]);
    assert_eq!(prepared.end_mappings(), [3, 7]);
    assert!(prepared.text_tokens.len() > policy.max_len());
}

#[test]
fn synthetic_suffix_offsets_are_safe_without_changing_encoder_tokens() {
    let policy = BoundaryPreprocessingPolicy::default();

    let url = policy.prepare("https://example.test/path", &[]);
    assert_eq!(url.text_tokens, ["https://example.test/path."]);
    assert_eq!(url.original_offsets[0].end, url.original_text.len());
    assert_eq!(
        url.original_offsets[0].coverage,
        OriginalCoverage::OriginalWithSyntheticSuffix
    );
    assert!(url.original_offsets[0].has_original_text());
    assert!(url.original_offsets[0].contains_synthetic_suffix());

    let empty = policy.prepare("", &[]);
    assert_eq!(empty.text_tokens, ["."]);
    assert_eq!(
        (
            empty.original_offsets[0].start,
            empty.original_offsets[0].end
        ),
        (0, 0)
    );
    assert_eq!(
        empty.original_offsets[0].coverage,
        OriginalCoverage::SyntheticOnly
    );
    assert!(!empty.original_offsets[0].has_original_text());
}

fn codepoint_to_byte(text: &str, codepoint: usize) -> usize {
    if codepoint == text.chars().count() {
        return text.len();
    }
    text.char_indices().nth(codepoint).unwrap().0
}

fn assert_golden_directory(directory: &Path, required: bool) {
    if !directory.is_dir() {
        assert!(
            !required,
            "missing required golden directory {}",
            directory.display()
        );
        return;
    }
    let mut checked = 0;
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json")
            || path.file_name().and_then(|name| name.to_str()) == Some("manifest.json")
        {
            continue;
        }
        let metadata: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        if metadata.get("status").and_then(Value::as_str) != Some("ok")
            || metadata.get("text_tokens").is_none()
        {
            continue;
        }
        let npz_path = path.with_extension("npz");
        if !npz_path.is_file() {
            assert!(!required, "missing golden arrays {}", npz_path.display());
            continue;
        }

        let text = metadata["original_text"].as_str().unwrap();
        let expected_tokens = metadata["text_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let max_len = metadata["max_len"].as_u64().unwrap() as usize;
        let mut npz = NpzReader::new(File::open(&npz_path).unwrap()).unwrap();
        let expected_starts: Array1<i64> = npz.by_name("start_mappings").unwrap();
        let expected_ends: Array1<i64> = npz.by_name("end_mappings").unwrap();
        assert_eq!(expected_starts.len(), expected_ends.len());

        // Structure choice prefixes are present in `text_tokens` but absent
        // from original-text mappings. Recover and pass them explicitly.
        let prefix_words = expected_tokens.len() - expected_starts.len();
        let prefix = &expected_tokens[..prefix_words];
        let prepared = BoundaryPreprocessingPolicy::new(max_len, WordSplitter::Whitespace)
            .unwrap()
            .prepare(text, prefix);
        assert_eq!(
            prepared.text_tokens,
            expected_tokens,
            "{} tokens",
            path.display()
        );
        assert_eq!(prepared.choice_prefix_words, prefix_words);
        assert_eq!(
            prepared.start_mappings(),
            expected_starts
                .iter()
                .map(|&offset| codepoint_to_byte(text, offset as usize))
                .collect::<Vec<_>>(),
            "{} starts",
            path.display()
        );
        assert_eq!(
            prepared.end_mappings(),
            expected_ends
                .iter()
                .map(|&offset| codepoint_to_byte(text, offset as usize))
                .collect::<Vec<_>>(),
            "{} ends",
            path.display()
        );
        checked += 1;
    }
    if required {
        assert!(
            checked >= 3,
            "expected the committed boundary golden subset"
        );
    }
}

#[test]
fn actual_golden_text_tokens_and_mappings_match() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    assert_golden_directory(&root.join("gliner2.5-base-v1-subset"), true);
    // Developers with the optional full fixture bundle get the same check.
    assert_golden_directory(&root.join("gliner2.5-base-v1"), false);
}
