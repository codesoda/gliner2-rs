use gliner2_rs::Result;

mod common;
use common::{load_auto_pipeline, model_paths_from_args};

fn trimmed_line_bounds(text: &str) -> Vec<[usize; 2]> {
    let mut bounds = Vec::new();
    let mut cursor = 0;
    for chunk in text.split_inclusive('\n') {
        let line = chunk.strip_suffix('\n').unwrap_or(chunk);
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let leading = line.len() - line.trim_start().len();
            bounds.push([cursor + leading, cursor + leading + trimmed.len()]);
        }
        cursor += chunk.len();
    }
    bounds
}

fn main() -> Result<()> {
    let paths = model_paths_from_args("onnx/gliner2.5-base-v1");
    let pipeline = load_auto_pipeline(&paths, false)?;

    let text = "Alice joined Acme in Berlin.\nZoë spoke in São Paulo.\n東京 welcomed 李雷.";
    let labels = vec![
        "person".to_owned(),
        "organization".to_owned(),
        "location".to_owned(),
    ];
    let spans = trimmed_line_bounds(text);
    let groups = pipeline.score_explicit_spans(text, &labels, &spans)?;

    println!("text:\n{text}\n");
    println!("scored line bounds: {spans:?}");
    println!("Explicit confidences are independent scores; they need not sum to one.");
    println!("This demonstrates the API and is not an accuracy claim.\n");
    for group in groups {
        println!("label: {}", group.label);
        for span in group.spans {
            println!(
                "  [{}, {}) {:?}: logit={:.6}, confidence={:.6}",
                span.start, span.end, span.text, span.logit, span.confidence
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_trimmed_utf8_line_bounds() {
        let text = "  Alice  \n\n東京  ";
        let bounds = trimmed_line_bounds(text);
        assert_eq!(bounds, vec![[2, 7], [11, 17]]);
        assert_eq!(&text[bounds[0][0]..bounds[0][1]], "Alice");
        assert_eq!(&text[bounds[1][0]..bounds[1][1]], "東京");
    }
}
