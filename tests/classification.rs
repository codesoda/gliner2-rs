use gliner2_rs::classification::{
    ClassAct, ClassificationOutput, FormattedClassification, decode_classification,
};

#[test]
fn classification_formatting_matches_include_confidence_flag() {
    let out = ClassificationOutput::Single {
        label: "neutral".to_string(),
        confidence: 0.82,
    };
    assert_eq!(
        out.format(false),
        FormattedClassification::Single("neutral".to_string())
    );
    assert_eq!(
        out.format(true),
        FormattedClassification::SingleWithConfidence {
            label: "neutral".to_string(),
            confidence: 0.82
        }
    );

    let out = ClassificationOutput::Multi {
        labels: vec![
            ("technology".to_string(), 0.92),
            ("business".to_string(), 0.78),
        ],
    };
    assert_eq!(
        out.format(false),
        FormattedClassification::Multi(vec!["technology".to_string(), "business".to_string()])
    );
    assert_eq!(
        out.format(true),
        FormattedClassification::MultiWithConfidence(vec![
            ("technology".to_string(), 0.92),
            ("business".to_string(), 0.78),
        ])
    );
}

#[test]
fn decode_classification_respects_class_act() {
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let logits = vec![0.0, 1.0, 2.0];

    let out = decode_classification(&labels, &logits, false, 0.5, ClassAct::Softmax);
    match out {
        ClassificationOutput::Single { label, confidence } => {
            assert_eq!(label, "C");
            // softmax([0,1,2]) ~= [0.09, 0.24, 0.66]
            assert!(confidence > 0.6 && confidence < 0.8);
        }
        _ => panic!("expected single-label output"),
    }

    let out = decode_classification(&labels, &logits, false, 0.5, ClassAct::Sigmoid);
    match out {
        ClassificationOutput::Single { label, confidence } => {
            assert_eq!(label, "C");
            // sigmoid(2) ~= 0.88
            assert!(confidence > 0.85 && confidence < 0.95);
        }
        _ => panic!("expected single-label output"),
    }

    let out = decode_classification(&labels, &logits, false, 0.5, ClassAct::Auto);
    match out {
        ClassificationOutput::Single { label, confidence } => {
            assert_eq!(label, "C");
            // Auto should choose softmax for single-label.
            assert!(confidence > 0.6 && confidence < 0.8);
        }
        _ => panic!("expected single-label output"),
    }

    let out = decode_classification(&labels, &logits, true, 0.8, ClassAct::Auto);
    match out {
        ClassificationOutput::Multi { labels: chosen } => {
            // Auto should choose sigmoid for multi-label.
            assert_eq!(chosen.len(), 1);
            assert_eq!(chosen[0].0, "C");
            assert!(chosen[0].1 > 0.85 && chosen[0].1 < 0.95);
        }
        _ => panic!("expected multi-label output"),
    }
}
