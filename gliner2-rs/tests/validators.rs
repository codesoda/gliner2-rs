use gliner2_rs::validators::{RegexMode, RegexValidator};

#[test]
fn validator_full_mode_requires_whole_string_match() {
    let v = RegexValidator::new("abc");
    assert!(v.validate("abc"));
    assert!(v.validate("ABC")); // default is case-insensitive (matches Python default)
    assert!(!v.validate("zabc"));
    assert!(!v.validate("abcz"));
}

#[test]
fn validator_partial_mode_searches_anywhere() {
    let v = RegexValidator::new("abc").mode(RegexMode::Partial);
    assert!(v.validate("zabc"));
    assert!(v.validate("xxABCyy")); // case-insensitive by default
}

#[test]
fn validator_exclude_inverts_match() {
    let v = RegexValidator::new(r"^test")
        .mode(RegexMode::Partial)
        .exclude(true);
    assert!(!v.validate("Test Phone"));
    assert!(v.validate("iPhone"));
}

#[test]
fn validator_case_insensitive_can_be_disabled() {
    let v = RegexValidator::new(r"^test")
        .mode(RegexMode::Partial)
        .exclude(true)
        .case_insensitive(false);
    // Case-sensitive now: "Test ..." doesn't match "^test", so it passes.
    assert!(v.validate("Test Phone"));
    // Lowercase still matches, so it's excluded.
    assert!(!v.validate("test phone"));
}

