use ndarray::Array2;

/// Generate spans in the same layout as GLiNER2 `compute_span_rep`:
/// for each `start` in `[0..text_len)`, for each `width` in `[0..max_width)`,
/// produce `(start, start+width)` if valid else `(-1,-1)`.
pub fn build_spans(text_len: usize, max_width: usize) -> Array2<i64> {
    let num_spans = text_len * max_width;
    let mut spans = Array2::<i64>::zeros((num_spans, 2));

    let mut idx = 0usize;
    for start in 0..text_len {
        for width in 0..max_width {
            if start + width < text_len {
                spans[[idx, 0]] = start as i64;
                spans[[idx, 1]] = (start + width) as i64; // inclusive end
            } else {
                spans[[idx, 0]] = -1;
                spans[[idx, 1]] = -1;
            }
            idx += 1;
        }
    }

    spans
}
