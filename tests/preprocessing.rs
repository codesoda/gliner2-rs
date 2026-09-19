use gliner2_rs::preprocessing::PreprocessingPolicy;

#[test]
fn truncates_tokens_and_byte_offsets_together() {
    let policy = PreprocessingPolicy::new(Some(2));
    let spans = policy.tokenize("Alpha βeta gamma", true);

    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].token, "alpha");
    assert_eq!((spans[0].start, spans[0].end), (0, 5));
    assert_eq!(spans[1].token, "βeta");
    assert_eq!((spans[1].start, spans[1].end), (6, 11));
}

#[test]
fn absent_cap_keeps_all_words() {
    let policy = PreprocessingPolicy::new(None);
    let spans = policy.tokenize("one two three", false);
    assert_eq!(spans.len(), 3);
    assert_eq!(policy.max_len(), None);
}

#[test]
fn choice_prefix_is_never_truncated_or_counted_against_original_cap() {
    let policy = PreprocessingPolicy::new(Some(2));
    let prefix = vec!["(".into(), "kind:".into(), "red".into(), ")".into()];
    let original = vec!["one".into(), "two".into(), "three".into()];

    let combined = policy.prepend_prefix(&prefix, &original);
    assert_eq!(combined, ["(", "kind:", "red", ")", "one", "two"]);
}
