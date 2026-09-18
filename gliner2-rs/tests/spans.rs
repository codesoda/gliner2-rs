use gliner2_rs::spans::build_spans;

#[test]
fn spans_match_python_layout() {
    let spans = build_spans(5, 3);
    assert_eq!(spans.shape(), &[15, 2]);

    let got: Vec<(i64, i64)> = spans
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    let expected = vec![
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
        (-1, -1),
        (4, 4),
        (-1, -1),
        (-1, -1),
    ];

    assert_eq!(got, expected);
}
