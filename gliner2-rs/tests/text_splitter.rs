use gliner2_rs::text::tokenize_with_offsets;

#[test]
fn tokenize_with_offsets_matches_slices_lowercased() {
    let text = "Email Test.Email+1@example.com, visit https://example.com/path.";
    let toks = tokenize_with_offsets(text, true);
    assert!(!toks.is_empty());

    for t in toks {
        let slice = text.get(t.start..t.end).expect("valid utf-8 slice");
        assert_eq!(t.token, slice.to_lowercase());
    }
}

