//! Pure choice lookup and literal-assignment helpers for boundary records.
//!
//! The three matching modes here are intentionally distinct: prefix lookup
//! uses Python `str.lower`, decoded-surface lookup uses full `str.casefold`, and
//! literal mentions use Python `re.IGNORECASE` scalar equivalence plus Python
//! `\w` boundaries. Public-facing coordinates are UTF-8 byte offsets.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, ensure};

use super::choice_unicode;

/// A source literal associated with its canonical configured choice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChoiceMention {
    pub(crate) choice: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Deduplicate exact choices in input order and locate their first lowercased
/// single-token occurrence in the schema prefix.
pub(crate) fn present_choices(choices: &[String], prefix: &[String]) -> Vec<(String, usize)> {
    let lowered_prefix: Vec<String> = prefix
        .iter()
        .map(|token| choice_unicode::lower(token))
        .collect();
    let mut seen = HashSet::new();
    let mut present = Vec::new();
    for choice in choices {
        if !seen.insert(choice.as_str()) {
            continue;
        }
        let lowered = choice_unicode::lower(choice);
        if let Some(index) = lowered_prefix.iter().position(|token| token == &lowered) {
            present.push((choice.clone(), index));
        }
    }
    present
}

/// Match a decoded surface by full casefold. If configured choices collide
/// after casefolding, the last configured choice is canonical, as in Python's
/// dictionary-comprehension construction.
pub(crate) fn match_choice_surface(surface: &str, choices: &[String]) -> Option<String> {
    let folded_surface = choice_unicode::casefold(surface);
    choices
        .iter()
        .rev()
        .find(|choice| choice_unicode::casefold(choice) == folded_surface)
        .cloned()
}

/// Find literal enum mentions and assign them to natural record anchors.
///
/// Matching mirrors per-choice Python `re.finditer` calls with
/// `(?<!\w)...(?!\w)` and `re.IGNORECASE`, including non-overlap and empty
/// literal behavior. Mentions are then stably merged by start offset. Every
/// anchor is validated as a UTF-8 byte span before matching.
pub(crate) fn literal_choice_mentions(
    text: &str,
    choices: &[String],
    anchors: &[Option<[usize; 2]>],
) -> Result<(bool, BTreeMap<usize, Vec<ChoiceMention>>)> {
    validate_anchors(text, anchors)?;

    let text_chars: Vec<char> = text.chars().collect();
    let mut byte_offsets: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    byte_offsets.push(text.len());

    let mut mentions = Vec::new();
    for choice in choices {
        let choice_chars: Vec<char> = choice.chars().collect();
        for (start, end) in literal_matches(&text_chars, &choice_chars) {
            mentions.push(ChoiceMention {
                choice: choice.clone(),
                start: byte_offsets[start],
                end: byte_offsets[end],
            });
        }
    }
    mentions.sort_by_key(|mention| mention.start);

    if mentions.is_empty() {
        return Ok((false, BTreeMap::new()));
    }

    let mut valid_anchors: Vec<(usize, [usize; 2])> = anchors
        .iter()
        .enumerate()
        .filter_map(|(index, anchor)| anchor.map(|span| (index, span)))
        .collect();
    if valid_anchors.is_empty() {
        return Ok((true, BTreeMap::new()));
    }
    valid_anchors.sort_by_key(|(index, span)| (span[0], *index));

    let mut assigned: BTreeMap<usize, Vec<ChoiceMention>> = BTreeMap::new();
    for mention in mentions {
        let owner = valid_anchors
            .iter()
            .rev()
            .find(|(_, span)| span[0] <= mention.start)
            .unwrap_or(&valid_anchors[0])
            .0;
        assigned.entry(owner).or_default().push(mention);
    }

    for owned in assigned.values_mut() {
        let mut seen = HashSet::new();
        owned.retain(|mention| seen.insert(mention.choice.clone()));
        owned.sort_by_key(|mention| mention.start);
    }
    Ok((true, assigned))
}

/// Select the mention with the smallest Unicode-codepoint gap to an anchor.
/// Coordinates are UTF-8 bytes, but upstream ranks distances in Python string
/// coordinates; using byte distance can select a different multilingual value.
/// Overlap has distance zero and equal distances retain the first mention.
pub(crate) fn nearest_choice<'a>(
    text: &str,
    anchor: [usize; 2],
    mentions: &'a [ChoiceMention],
) -> Result<Option<&'a ChoiceMention>> {
    validate_anchors(text, &[Some(anchor)])?;
    let mut best = None;
    let mut best_distance = usize::MAX;
    for (index, mention) in mentions.iter().enumerate() {
        validate_anchors(text, &[Some([mention.start, mention.end])])
            .with_context(|| format!("invalid choice mention {index}"))?;
        let distance = if mention.end < anchor[0] {
            text[mention.end..anchor[0]].chars().count()
        } else if anchor[1] < mention.start {
            text[anchor[1]..mention.start].chars().count()
        } else {
            0
        };
        if distance < best_distance {
            best = Some(mention);
            best_distance = distance;
        }
    }
    Ok(best)
}

fn validate_anchors(text: &str, anchors: &[Option<[usize; 2]>]) -> Result<()> {
    for (index, anchor) in anchors.iter().enumerate() {
        let Some([start, end]) = *anchor else {
            continue;
        };
        ensure!(
            start <= end,
            "choice anchor {index} has reversed byte span [{start}, {end})"
        );
        ensure!(
            end <= text.len(),
            "choice anchor {index} byte span [{start}, {end}) exceeds text length {}",
            text.len()
        );
        ensure!(
            text.is_char_boundary(start) && text.is_char_boundary(end),
            "choice anchor {index} byte span [{start}, {end}) is not on UTF-8 character boundaries"
        );
    }
    Ok(())
}

fn literal_matches(text: &[char], choice: &[char]) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    let mut cursor = 0;
    while cursor <= text.len() {
        let Some(end) = cursor.checked_add(choice.len()) else {
            break;
        };
        if end > text.len() {
            break;
        }
        let left_boundary = cursor == 0 || !choice_unicode::is_word(text[cursor - 1]);
        let right_boundary = end == text.len() || !choice_unicode::is_word(text[end]);
        let equal = text[cursor..end]
            .iter()
            .zip(choice)
            .all(|(&left, &right)| choice_unicode::ignorecase_equal(left, right));
        if left_boundary && right_boundary && equal {
            matches.push((cursor, end));
            // Python advances one scalar after an empty match to avoid looping.
            cursor += choice.len().max(1);
        } else {
            cursor += 1;
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn prefix_lookup_is_exact_deduplicated_lower_and_keeps_canonical_text() {
        let choices = strings(&["ΟΣ", "ΟΣ", "Ა", "multi word", "missing"]);
        let prefix = strings(&["other", "Ος", "ა", "multi", "word"]);
        assert_eq!(
            present_choices(&choices, &prefix),
            vec![("ΟΣ".into(), 1), ("Ა".into(), 2)]
        );
    }

    #[test]
    fn pinned_lower_handles_final_sigma_case_ignorable_and_mtavruli() {
        assert_eq!(choice_unicode::lower("ΟΣ"), "ος");
        assert_eq!(choice_unicode::lower("ΟΣΑ"), "οσα");
        assert_eq!(choice_unicode::lower("A\u{301}Σ"), "a\u{301}ς");
        assert_eq!(choice_unicode::lower("Σ"), "σ");
        assert_eq!(choice_unicode::lower("İΣ"), "i\u{307}ς");
        assert_eq!(choice_unicode::lower("ᲐΣ"), "აς");
    }

    #[test]
    fn surface_lookup_is_full_casefold_with_last_collision_winning() {
        let choices = strings(&["SS", "ß", "ΟΣ", "οσ"]);
        assert_eq!(match_choice_surface("ẞ", &choices), Some("ß".into()));
        assert_eq!(match_choice_surface("ος", &choices), Some("οσ".into()));
        assert_eq!(match_choice_surface("missing", &choices), None);
    }

    #[test]
    fn literal_ignorecase_is_not_full_casefold() -> Result<()> {
        let text = "ß ss ẞ SS";
        let (_, assigned) = literal_choice_mentions(text, &strings(&["ß"]), &[Some([0, 2])])?;
        assert_eq!(
            assigned[&0],
            vec![ChoiceMention {
                choice: "ß".into(),
                start: 0,
                end: 2,
            }]
        );
        Ok(())
    }

    #[test]
    fn literal_special_cases_and_utf8_spans_match_fixed_python_oracles() {
        let cases = [
            ("İ I ı i", "i", vec![[0, 2], [3, 4], [5, 7], [8, 9]]),
            ("ſ S s", "s", vec![[0, 2], [3, 4], [5, 6]]),
            ("K K k", "k", vec![[0, 3], [4, 5], [6, 7]]),
            ("Σ σ ς", "Σ", vec![[0, 2], [3, 5], [6, 8]]),
            ("ß ss ẞ SS", "ß", vec![[0, 2], [6, 9]]),
            (
                "a\u{301}a a\u{301} a",
                "a",
                vec![[0, 1], [3, 4], [5, 6], [9, 10]],
            ),
            ("猫,猫咪,猫", "猫", vec![[0, 3], [11, 14]]),
            ("🙂 ok 🙂", "🙂", vec![[0, 4], [8, 12]]),
        ];
        for (text, choice, expected) in cases {
            let chars: Vec<_> = text.chars().collect();
            let offsets: Vec<_> = text
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(text.len()))
                .collect();
            let actual: Vec<_> = literal_matches(&chars, &choice.chars().collect::<Vec<_>>())
                .into_iter()
                .map(|(start, end)| [offsets[start], offsets[end]])
                .collect();
            assert_eq!(actual, expected, "{text:?} / {choice:?}");
        }
    }

    #[test]
    fn combining_marks_are_not_word_characters() -> Result<()> {
        let text = "a\u{301}a a\u{301} a";
        let choices = strings(&["a", "a\u{301}"]);
        let (_, assigned) = literal_choice_mentions(text, &choices, &[Some([0, 0])])?;
        assert_eq!(
            assigned[&0],
            vec![
                ChoiceMention {
                    choice: "a".into(),
                    start: 0,
                    end: 1
                },
                ChoiceMention {
                    choice: "a\u{301}".into(),
                    start: 5,
                    end: 8
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn ownership_before_between_after_ties_and_choice_order() -> Result<()> {
        let text = "red A blue B green";
        let anchors = [Some([4, 5]), Some([11, 12]), Some([11, 12])];
        let (_, assigned) =
            literal_choice_mentions(text, &strings(&["red", "blue", "green", "BLUE"]), &anchors)?;
        assert_eq!(
            assigned[&0]
                .iter()
                .map(|m| m.choice.as_str())
                .collect::<Vec<_>>(),
            ["red", "blue", "BLUE"]
        );
        assert_eq!(
            assigned[&2]
                .iter()
                .map(|m| m.choice.as_str())
                .collect::<Vec<_>>(),
            ["green"]
        );
        assert!(!assigned.contains_key(&1));
        Ok(())
    }

    #[test]
    fn no_mentions_and_no_anchors_have_distinct_flags() -> Result<()> {
        let empty = literal_choice_mentions("text", &strings(&["none"]), &[])?;
        assert_eq!(empty, (false, BTreeMap::new()));
        let unowned = literal_choice_mentions("choice", &strings(&["choice"]), &[])?;
        assert_eq!(unowned, (true, BTreeMap::new()));
        Ok(())
    }

    #[test]
    fn punctuation_metacharacters_multiword_and_per_choice_nonoverlap() -> Result<()> {
        let text = "a+b aab a+b, red blue; ababa";
        let (_, assigned) =
            literal_choice_mentions(text, &strings(&["a+b", "red blue", "aba"]), &[Some([0, 0])])?;
        assert_eq!(
            assigned[&0]
                .iter()
                .map(|m| m.choice.as_str())
                .collect::<Vec<_>>(),
            ["a+b", "red blue"]
        );
        Ok(())
    }

    #[test]
    fn empty_choice_uses_python_boundaries_and_nonoverlap() -> Result<()> {
        assert_eq!(
            literal_matches(&".a..".chars().collect::<Vec<_>>(), &[]),
            vec![(0, 0), (3, 3), (4, 4)]
        );
        let (_, assigned) = literal_choice_mentions(".a..", &strings(&[""]), &[Some([0, 0])])?;
        // Exact-choice set semantics retains its first source occurrence.
        assert_eq!(assigned[&0][0].start, 0);
        assert_eq!(assigned[&0][0].end, 0);
        Ok(())
    }

    #[test]
    fn duplicate_exact_choice_retains_first_mention_per_owner() -> Result<()> {
        let (_, assigned) = literal_choice_mentions(
            "x red red y red",
            &strings(&["red", "red"]),
            &[Some([0, 1]), Some([12, 13])],
        )?;
        assert_eq!(
            assigned[&0],
            vec![ChoiceMention {
                choice: "red".into(),
                start: 2,
                end: 5
            }]
        );
        assert_eq!(
            assigned[&1],
            vec![ChoiceMention {
                choice: "red".into(),
                start: 12,
                end: 15
            }]
        );
        Ok(())
    }

    #[test]
    fn invalid_anchor_offsets_are_contextual_errors() {
        let unicode = literal_choice_mentions("é", &strings(&["none"]), &[Some([1, 2])])
            .unwrap_err()
            .to_string();
        assert!(unicode.contains("anchor 0"), "{unicode}");
        assert!(unicode.contains("UTF-8"), "{unicode}");

        let reversed = literal_choice_mentions("abc", &[], &[Some([2, 1])])
            .unwrap_err()
            .to_string();
        assert!(reversed.contains("reversed"), "{reversed}");

        let outside = literal_choice_mentions("abc", &[], &[Some([0, 4])])
            .unwrap_err()
            .to_string();
        assert!(outside.contains("text length 3"), "{outside}");
    }

    #[test]
    fn nearest_uses_python_character_distance_not_utf8_byte_distance() -> Result<()> {
        let text = "L你你A.....R";
        let mentions = vec![
            ChoiceMention {
                choice: "L".into(),
                start: 0,
                end: 1,
            },
            ChoiceMention {
                choice: "R".into(),
                start: 13,
                end: 14,
            },
        ];
        // Python gaps: left=2, right=5. Byte gaps: left=6, right=5.
        assert_eq!(
            nearest_choice(text, [7, 8], &mentions)?.unwrap().choice,
            "L"
        );
        assert!(nearest_choice(text, [2, 8], &mentions).is_err());
        let invalid = [ChoiceMention {
            choice: "bad".into(),
            start: 2,
            end: 4,
        }];
        assert!(nearest_choice(text, [7, 8], &invalid).is_err());
        Ok(())
    }

    #[test]
    fn nearest_uses_gap_and_first_tie() {
        let mentions = vec![
            ChoiceMention {
                choice: "left".into(),
                start: 1,
                end: 3,
            },
            ChoiceMention {
                choice: "right".into(),
                start: 7,
                end: 9,
            },
            ChoiceMention {
                choice: "overlap".into(),
                start: 4,
                end: 6,
            },
        ];
        assert_eq!(
            nearest_choice("0123456789", [5, 7], &mentions)
                .unwrap()
                .unwrap()
                .choice,
            "right"
        );
        assert_eq!(
            nearest_choice("0123456789", [5, 5], &mentions[..2])
                .unwrap()
                .unwrap()
                .choice,
            "left"
        );
        assert!(nearest_choice("", [0, 0], &[]).unwrap().is_none());
    }
}
