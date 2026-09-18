use ndarray::Array2;

use gliner2_rs::decode::{find_valid_spans, greedy_non_overlapping};

#[test]
fn greedy_non_overlapping_prefers_highest_score() {
    let text = "alice works at acme";
    let tokens = vec![
        "alice".to_string(),
        "works".to_string(),
        "at".to_string(),
        "acme".to_string(),
    ];
    let starts = vec![0usize, 6, 12, 15];
    let ends = vec![5usize, 11, 14, 19];

    // [text_len=4, max_width=3]
    let mut logits = Array2::<f32>::zeros((4, 3));
    // span A: tokens[0..2] = "alice works"
    logits[[0, 1]] = 10.0;
    // span B: tokens[1..2] = "works" (overlaps with A)
    logits[[1, 0]] = 9.0;
    // span C: tokens[2..3] = "at" (disjoint with A)
    logits[[2, 0]] = 8.0;

    let spans = find_valid_spans(
        logits.view(),
        0.9,
        text,
        &starts,
        &ends,
        &tokens,
    );
    let spans = greedy_non_overlapping(spans);

    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].text, "alice works");
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 11);
    assert_eq!(spans[1].text, "at");
    assert_eq!(spans[1].start, 12);
    assert_eq!(spans[1].end, 14);
}
