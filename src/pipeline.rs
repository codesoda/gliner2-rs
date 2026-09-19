use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, anyhow};
use ndarray::{Array1, Array2, Axis, s};

use crate::{
    Result,
    adapters::{AdapterConfig, read_lora_r},
    classification::{
        ClassAct, ClassificationOutput, FormattedClassification,
        build_classification_schema_tokens, build_classification_schema_tokens_with_descriptions,
        decode_classification,
    },
    classifier::Classifier,
    config::{Architecture, ModelConfig},
    decode::{find_valid_spans, greedy_non_overlapping},
    embeddings::extract_embeddings,
    encoder::Encoder,
    entities::{
        EntityMatches, EntitySpan, FormattedEntitySpan, FormattedEntityValue,
        build_entities_schema_tokens, build_entities_schema_tokens_with_descriptions,
        format_entity_spans,
    },
    extractor::{Extractor as SpanExtractor, ExtractorOutput},
    json::{JsonExtraction, JsonSchema},
    preprocessing::PreprocessingPolicy,
    relations::{
        FormattedRelationExtraction, FormattedRelationPair, RelationExtraction,
        build_relation_schema_tokens,
    },
    schema::format_input_with_mapping,
    schema_spec::{
        EntityLabels, EntitySpec, ExtractionResult, QuickClassificationTask, RelationSpec,
        SchemaBuilder, SchemaSpec, StructureSpec,
    },
    spans::build_spans,
    structures::{build_structure_choice_prefix, build_structure_schema_tokens},
    tokenizer::RuntimeTokenizer,
};

/// Minimal end-to-end pipeline that wires:
/// schema/text formatting -> encoder -> embedding extraction -> span generation -> extractor.
///
/// This is an internal stepping stone toward `extract_entities(text, labels)`.
pub struct SpanPipeline {
    tokenizer: RuntimeTokenizer,
    encoder: Encoder,
    base_encoder: Option<Encoder>,
    extractor: SpanExtractor,
    classifier: Option<Classifier>,
    adapter_config: Option<AdapterConfig>,
    max_width: usize,
    extractor_max_fields: usize,
    preprocessing: PreprocessingPolicy,
}

/// Backward-compatible name for the GLiNER2 span implementation.
pub type Gliner2Pipeline = SpanPipeline;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SpanPipeline {
    pub fn new(
        model_dir: impl AsRef<Path>,
        encoder_onnx: impl AsRef<Path>,
        extractor_onnx: impl AsRef<Path>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = ModelConfig::from_dir(model_dir)?;
        if config.architecture != Architecture::Span {
            return Err(anyhow!(
                "SpanPipeline cannot load a boundary architecture; use AutoPipeline::from_dir"
            ));
        }

        Ok(Self {
            tokenizer: RuntimeTokenizer::from_dir(model_dir)?,
            encoder: Encoder::new(encoder_onnx)?,
            base_encoder: None,
            extractor: SpanExtractor::new(extractor_onnx)?,
            classifier: None,
            adapter_config: None,
            max_width: 8,
            // IMPORTANT: `extractor_padded.onnx` was exported with a fixed MAX_FIELDS.
            // Keep this aligned with export/export_extractor_padded.py.
            //
            // TODO: If you hit the "schema has N fields but extractor supports at most 64" error:
            // 1) bump `MAX_FIELDS` in `export/export_extractor_padded.py`,
            // 2) rerun `./.venv/bin/python export/export_extractor_padded.py` to regenerate
            //    `onnx/gliner2-base-v1/extractor_padded.onnx`,
            // 3) update this constant (and any tests/examples that assume 64),
            // 4) rerun `cd gliner2-rs && cargo test`.
            extractor_max_fields: 64,
            preprocessing: PreprocessingPolicy::new(config.max_len),
        })
    }

    /// Load a complete span bundle with tokenizer and all ONNX graphs colocated.
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        required_file(bundle, "tokenizer.json")?;
        let encoder = required_file(bundle, "encoder.onnx")?;
        let extractor = required_file(bundle, "extractor_padded.onnx")?;
        let classifier = required_file(bundle, "classifier.onnx")?;
        Self::new(bundle, encoder, extractor)?.with_classifier(classifier)
    }

    pub fn with_classifier(mut self, classifier_onnx: impl AsRef<Path>) -> Result<Self> {
        self.classifier = Some(Classifier::new(classifier_onnx)?);
        Ok(self)
    }

    pub fn has_adapter(&self) -> bool {
        self.adapter_config.is_some()
    }

    pub fn adapter_config(&self) -> Option<&AdapterConfig> {
        self.adapter_config.as_ref()
    }

    pub fn load_adapter(&mut self, _adapter_dir: impl AsRef<Path>) -> Result<()> {
        let adapter_dir = _adapter_dir.as_ref();

        // NOTE: ONNX Runtime sessions have static weights. For LoRA-style adapters we currently
        // require exporting an adapter-specific `encoder.onnx` (merged weights) and then we
        // swap the encoder session.
        let encoder_onnx = adapter_dir.join("encoder.onnx");
        if !encoder_onnx.exists() {
            return Err(anyhow!(
                "adapter bundle missing `encoder.onnx` at {} (export a merged ONNX encoder for this adapter)",
                encoder_onnx.display()
            ));
        }

        let new_encoder = Encoder::new(&encoder_onnx)?;
        let old_encoder = std::mem::replace(&mut self.encoder, new_encoder);

        // First adapter load: stash base encoder so `unload_adapter()` is O(1).
        if self.base_encoder.is_none() {
            self.base_encoder = Some(old_encoder);
        }

        self.adapter_config = Some(AdapterConfig {
            adapter_dir: adapter_dir.to_path_buf(),
            encoder_onnx,
            lora_r: read_lora_r(adapter_dir),
        });

        Ok(())
    }

    pub fn unload_adapter(&mut self) -> Result<()> {
        if self.adapter_config.is_none() {
            return Ok(());
        }

        let base = self
            .base_encoder
            .take()
            .context("adapter is loaded but base encoder was not saved")?;

        let _old_adapter = std::mem::replace(&mut self.encoder, base);
        self.adapter_config = None;
        Ok(())
    }

    pub fn infer_raw(
        &self,
        schema_tokens_list: &[Vec<String>],
        text_tokens: &[String],
    ) -> Result<ExtractorOutput> {
        let text_tokens = self.preprocessing.truncate_tokens(text_tokens);
        self.infer_prepared(schema_tokens_list, text_tokens)
    }

    /// Infer over text tokens already prepared by a high-level path. This is
    /// separate from `infer_raw` so structural prefixes are never capped as if
    /// they were original document words.
    fn infer_prepared(
        &self,
        schema_tokens_list: &[Vec<String>],
        text_tokens: &[String],
    ) -> Result<ExtractorOutput> {
        let formatted =
            format_input_with_mapping(&self.tokenizer, schema_tokens_list, text_tokens)?;
        let seq_len = formatted.input_ids.len();

        let input_ids = Array2::from_shape_vec((1, seq_len), formatted.input_ids.clone())?;
        let attention_mask =
            Array2::from_shape_vec((1, seq_len), formatted.attention_mask.clone())?;

        let hidden = self.encoder.infer(input_ids, attention_mask)?;

        let extracted = extract_embeddings(&hidden, &formatted, schema_tokens_list.len())?;
        let text_emb = extracted.text_emb;
        let text_len = text_emb.len_of(Axis(0));

        // Use schema 0 for now (milestone scope).
        let schema0 = extracted
            .schema_embs
            .first()
            .context("missing schema embeddings")?;
        if schema0.is_empty() {
            return Err(anyhow!(
                "schema embeddings are empty; expected at least [P]"
            ));
        }

        let field_count = schema0.len().saturating_sub(1);
        if field_count == 0 {
            return Err(anyhow!(
                "schema embeddings contain only [P]; expected at least one field marker"
            ));
        }
        if field_count > self.extractor_max_fields {
            return Err(anyhow!(
                "schema has {field_count} fields but extractor supports at most {} (re-export with a larger MAX_FIELDS)",
                self.extractor_max_fields
            ));
        }

        let hidden_size = schema0[0].len();
        let mut schema_emb_padded =
            Array2::<f32>::zeros((1 + self.extractor_max_fields, hidden_size));
        schema_emb_padded.row_mut(0).assign(&schema0[0]);
        for (field_idx, emb) in schema0.iter().skip(1).enumerate() {
            schema_emb_padded.row_mut(1 + field_idx).assign(emb);
        }

        let mut schema_mask = Array1::<bool>::from_elem(self.extractor_max_fields, false);
        for i in 0..field_count {
            schema_mask[i] = true;
        }

        let spans_idx = build_spans(text_len, self.max_width);

        self.extractor
            .infer(text_emb, schema_emb_padded, schema_mask, spans_idx)
    }

    pub fn extract_entities(
        &self,
        text: &str,
        entity_labels: &[String],
        threshold: f32,
    ) -> Result<Vec<EntityMatches>> {
        let specs: Vec<EntitySpec> = entity_labels.iter().cloned().map(EntitySpec::new).collect();
        self.extract_entities_with_specs(text, &specs, threshold)
    }

    pub fn extract_entities_text(
        &self,
        text: &str,
        entities: impl Into<EntityLabels>,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<BTreeMap<String, FormattedEntityValue>> {
        let schema = SchemaBuilder::new().entities(entities).build();
        let out =
            self.extract_internal(text, &schema, threshold, include_confidence, include_spans)?;
        Ok(out.entities)
    }

    fn extract_entities_with_specs(
        &self,
        text: &str,
        entity_specs: &[EntitySpec],
        default_threshold: f32,
    ) -> Result<Vec<EntityMatches>> {
        if entity_specs.is_empty() {
            return Ok(Vec::new());
        }

        let token_spans = self.preprocessing.tokenize(text, true);
        let text_tokens: Vec<String> = token_spans.iter().map(|t| t.token.clone()).collect();
        let token_starts: Vec<usize> = token_spans.iter().map(|t| t.start).collect();
        let token_ends: Vec<usize> = token_spans.iter().map(|t| t.end).collect();

        let entity_labels: Vec<String> = entity_specs.iter().map(|e| e.name.clone()).collect();
        let entity_desc: Vec<(String, String)> = entity_specs
            .iter()
            .filter_map(|e| e.description.as_ref().map(|d| (e.name.clone(), d.clone())))
            .collect();

        let schema_tokens = if entity_desc.is_empty() {
            build_entities_schema_tokens(&entity_labels, None)
        } else {
            build_entities_schema_tokens_with_descriptions(&entity_labels, None, &entity_desc)
        };
        let schema_tokens_list = vec![schema_tokens];

        let out = self.infer_prepared(&schema_tokens_list, &text_tokens)?;

        let count_row = out.count_logits.index_axis(Axis(0), 0);
        let mut best_idx = 0usize;
        let mut best_val = count_row[0];
        for (i, &v) in count_row.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let pred_count = best_idx;

        let mut results = Vec::with_capacity(entity_specs.len());
        for (entity_idx, spec) in entity_specs.iter().enumerate() {
            let threshold = spec.threshold.unwrap_or(default_threshold);
            let spans = if pred_count == 0 {
                Vec::new()
            } else {
                let field_logits = out.span_scores.slice(s![0, entity_idx, .., ..]);
                let spans = find_valid_spans(
                    field_logits,
                    threshold,
                    text,
                    &token_starts,
                    &token_ends,
                    &text_tokens,
                );
                let spans = greedy_non_overlapping(spans);
                spans
                    .into_iter()
                    .map(|s| EntitySpan {
                        start: s.start,
                        end: s.end,
                        text: s.text,
                        score: s.score,
                    })
                    .collect()
            };

            results.push(EntityMatches {
                label: spec.name.clone(),
                spans,
            });
        }

        Ok(results)
    }

    fn extract_structures_with_specs(
        &self,
        text: &str,
        structures: &[StructureSpec],
        default_threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<BTreeMap<String, Vec<BTreeMap<String, FormattedEntityValue>>>> {
        if structures.is_empty() {
            return Ok(BTreeMap::new());
        }

        let token_spans = self.preprocessing.tokenize(text, true);
        let text_tokens: Vec<String> = token_spans.iter().map(|t| t.token.clone()).collect();
        let token_starts: Vec<usize> = token_spans.iter().map(|t| t.start).collect();
        let token_ends: Vec<usize> = token_spans.iter().map(|t| t.end).collect();

        let mut results: BTreeMap<String, Vec<BTreeMap<String, FormattedEntityValue>>> =
            BTreeMap::new();

        for spec in structures {
            if spec.fields.is_empty() {
                results.insert(spec.name.clone(), Vec::new());
                continue;
            }

            let prefix_tokens = build_structure_choice_prefix(spec);
            let prefix_len = prefix_tokens.len();

            let combined_text_tokens = self
                .preprocessing
                .prepend_prefix(&prefix_tokens, &text_tokens);

            let schema_tokens = build_structure_schema_tokens(spec, None);
            let schema_tokens_list = vec![schema_tokens];

            let out = self.infer_prepared(&schema_tokens_list, &combined_text_tokens)?;

            let count_row = out.count_logits.index_axis(Axis(0), 0);
            let mut best_idx = 0usize;
            let mut best_val = count_row[0];
            for (i, &v) in count_row.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            let pred_count = best_idx;

            if pred_count == 0 {
                results.insert(spec.name.clone(), Vec::new());
                continue;
            }

            let mut instances: Vec<BTreeMap<String, FormattedEntityValue>> = Vec::new();
            for instance_idx in 0..pred_count {
                let mut instance: BTreeMap<String, FormattedEntityValue> = BTreeMap::new();

                for (field_idx, field) in spec.fields.iter().enumerate() {
                    let threshold = field.threshold.unwrap_or(default_threshold);

                    if !field.choices.is_empty() {
                        // Classification-within-structure: select among choices using prefix token scores.
                        let value = if prefix_len == 0 {
                            match field.dtype {
                                crate::schema_spec::FieldDtype::List => {
                                    FormattedEntityValue::List(Vec::new())
                                }
                                crate::schema_spec::FieldDtype::Str => {
                                    FormattedEntityValue::Single(None)
                                }
                            }
                        } else {
                            let logits = out.span_scores.slice(s![
                                instance_idx,
                                field_idx,
                                ..prefix_len,
                                ..
                            ]);

                            match field.dtype {
                                crate::schema_spec::FieldDtype::List => {
                                    let mut selected: Vec<FormattedEntitySpan> = Vec::new();
                                    let mut seen: std::collections::HashSet<String> =
                                        std::collections::HashSet::new();
                                    for choice in &field.choices {
                                        let key = choice.to_ascii_lowercase();
                                        if !seen.insert(key) {
                                            continue;
                                        }

                                        let mut ok = false;
                                        for (idx, tok) in prefix_tokens.iter().enumerate() {
                                            let tok_lower = tok.to_ascii_lowercase();
                                            let choice_lower = choice.to_ascii_lowercase();
                                            if tok_lower == choice_lower
                                                || tok_lower.contains(&choice_lower)
                                            {
                                                let score = sigmoid(logits[[idx, 0]]);
                                                if score >= threshold {
                                                    ok = true;
                                                    break;
                                                }
                                            }
                                        }

                                        if ok {
                                            selected
                                                .push(FormattedEntitySpan::Text(choice.clone()));
                                        }
                                    }
                                    FormattedEntityValue::List(selected)
                                }
                                crate::schema_spec::FieldDtype::Str => {
                                    let mut best_choice: Option<String> = None;
                                    let mut best_score = f32::NEG_INFINITY;

                                    for choice in &field.choices {
                                        for (idx, tok) in prefix_tokens.iter().enumerate() {
                                            let tok_lower = tok.to_ascii_lowercase();
                                            let choice_lower = choice.to_ascii_lowercase();
                                            if tok_lower == choice_lower
                                                || tok_lower.contains(&choice_lower)
                                            {
                                                let score = sigmoid(logits[[idx, 0]]);
                                                if score > best_score {
                                                    best_score = score;
                                                    best_choice = Some(choice.clone());
                                                }
                                            }
                                        }
                                    }

                                    let chosen = if let Some(choice) = best_choice {
                                        if best_score >= threshold
                                            || (threshold == 0.0 && best_score.is_finite())
                                        {
                                            Some(FormattedEntitySpan::Text(choice))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    FormattedEntityValue::Single(chosen)
                                }
                            }
                        };

                        instance.insert(field.name.clone(), value);
                        continue;
                    }

                    // Regular span extraction from the text portion (ignore prefix).
                    let logits =
                        out.span_scores
                            .slice(s![instance_idx, field_idx, prefix_len.., ..]);
                    let mut spans = find_valid_spans(
                        logits,
                        threshold,
                        text,
                        &token_starts,
                        &token_ends,
                        &text_tokens,
                    );
                    if !field.validators.is_empty() {
                        spans.retain(|s| field.validators.iter().all(|v| v.validate(&s.text)));
                    }
                    let spans = greedy_non_overlapping(spans);
                    let spans: Vec<EntitySpan> = spans
                        .into_iter()
                        .map(|s| EntitySpan {
                            start: s.start,
                            end: s.end,
                            text: s.text,
                            score: s.score,
                        })
                        .collect();

                    instance.insert(
                        field.name.clone(),
                        format_entity_spans(
                            &spans,
                            field.dtype.clone(),
                            include_confidence,
                            include_spans,
                        ),
                    );
                }

                // Only keep instances that contain at least one non-empty field.
                let has_content = instance.values().any(|v| match v {
                    FormattedEntityValue::List(values) => !values.is_empty(),
                    FormattedEntityValue::Single(value) => value.is_some(),
                });
                if has_content {
                    instances.push(instance);
                }
            }

            results.insert(spec.name.clone(), instances);
        }

        Ok(results)
    }

    fn extract_relations_with_specs(
        &self,
        text: &str,
        relations: &[RelationSpec],
        default_threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<FormattedRelationExtraction> {
        if relations.is_empty() {
            return Ok(BTreeMap::new());
        }

        let token_spans = self.preprocessing.tokenize(text, true);
        let text_tokens: Vec<String> = token_spans.iter().map(|t| t.token.clone()).collect();
        let token_starts: Vec<usize> = token_spans.iter().map(|t| t.start).collect();
        let token_ends: Vec<usize> = token_spans.iter().map(|t| t.end).collect();

        let mut results: FormattedRelationExtraction = BTreeMap::new();

        for spec in relations {
            let schema_tokens =
                build_relation_schema_tokens(&spec.name, spec.description.as_deref());
            let schema_tokens_list = vec![schema_tokens];

            let out = self.infer_prepared(&schema_tokens_list, &text_tokens)?;

            let count_row = out.count_logits.index_axis(Axis(0), 0);
            let mut best_idx = 0usize;
            let mut best_val = count_row[0];
            for (i, &v) in count_row.iter().enumerate().skip(1) {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            let pred_count = best_idx;

            let threshold = spec.threshold.unwrap_or(default_threshold);

            let mut pairs: Vec<FormattedRelationPair> = Vec::new();
            for instance_idx in 0..pred_count {
                // head
                let head_logits = out.span_scores.slice(s![instance_idx, 0, .., ..]);
                let head_spans = find_valid_spans(
                    head_logits,
                    threshold,
                    text,
                    &token_starts,
                    &token_ends,
                    &text_tokens,
                );
                let head_spans = greedy_non_overlapping(head_spans);
                let head_best = head_spans.into_iter().next().map(|s| EntitySpan {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                    score: s.score,
                });

                // tail
                let tail_logits = out.span_scores.slice(s![instance_idx, 1, .., ..]);
                let tail_spans = find_valid_spans(
                    tail_logits,
                    threshold,
                    text,
                    &token_starts,
                    &token_ends,
                    &text_tokens,
                );
                let tail_spans = greedy_non_overlapping(tail_spans);
                let tail_best = tail_spans.into_iter().next().map(|s| EntitySpan {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                    score: s.score,
                });

                if let (Some(head), Some(tail)) = (head_best, tail_best) {
                    pairs.push(FormattedRelationPair {
                        head: head.format(include_confidence, include_spans),
                        tail: tail.format(include_confidence, include_spans),
                    });
                }
            }

            results.insert(spec.name.clone(), pairs);
        }

        Ok(results)
    }

    pub fn extract_json(&self, text: &str, schema: &JsonSchema) -> Result<JsonExtraction> {
        self.extract_json_with_options(text, schema, 0.5, false, false)
    }

    pub fn extract_json_with_confidence(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        self.extract_json_with_options(text, schema, threshold, true, false)
    }

    pub fn extract_json_with_spans(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        self.extract_json_with_options(text, schema, threshold, false, true)
    }

    pub fn extract_json_with_confidence_and_spans(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        self.extract_json_with_options(text, schema, threshold, true, true)
    }

    pub fn extract_json_with_options(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonExtraction> {
        let mut structures: Vec<StructureSpec> = Vec::new();
        for s in &schema.structures {
            let mut fields = Vec::new();
            for field in &s.fields {
                fields.push(crate::json::parse_field_spec(field)?);
            }
            structures.push(StructureSpec {
                name: s.name.clone(),
                fields,
            });
        }

        self.extract_structures_with_specs(
            text,
            &structures,
            threshold,
            include_confidence,
            include_spans,
        )
    }

    pub fn extract_relations(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<RelationExtraction> {
        fn text_of(span: &FormattedEntitySpan) -> &str {
            match span {
                FormattedEntitySpan::Text(text) => text,
                FormattedEntitySpan::TextWithConfidence { text, .. } => text,
                FormattedEntitySpan::TextWithSpans { text, .. } => text,
                FormattedEntitySpan::TextWithConfidenceAndSpans { text, .. } => text,
            }
        }

        let specs: Vec<RelationSpec> = relation_types
            .iter()
            .cloned()
            .map(RelationSpec::new)
            .collect();

        let formatted = self.extract_relations_with_specs(text, &specs, threshold, false, false)?;
        let mut tuples: RelationExtraction = BTreeMap::new();
        for (rel, pairs) in formatted {
            tuples.insert(
                rel,
                pairs
                    .into_iter()
                    .map(|p| (text_of(&p.head).to_string(), text_of(&p.tail).to_string()))
                    .collect(),
            );
        }
        Ok(tuples)
    }

    pub fn extract_relations_with_confidence(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        self.extract_relations_with_options(text, relation_types, threshold, true, false)
    }

    pub fn extract_relations_with_spans(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        self.extract_relations_with_options(text, relation_types, threshold, false, true)
    }

    pub fn extract_relations_with_confidence_and_spans(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        self.extract_relations_with_options(text, relation_types, threshold, true, true)
    }

    pub fn extract_relations_with_options(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<FormattedRelationExtraction> {
        let specs: Vec<RelationSpec> = relation_types
            .iter()
            .cloned()
            .map(RelationSpec::new)
            .collect();
        self.extract_relations_with_specs(
            text,
            &specs,
            threshold,
            include_confidence,
            include_spans,
        )
    }

    pub fn batch_extract_relations<T: AsRef<str>>(
        &self,
        texts: &[T],
        relation_types: &[String],
        threshold: f32,
        batch_size: usize,
    ) -> Result<Vec<RelationExtraction>> {
        if batch_size == 0 {
            return Err(anyhow!("batch_size must be > 0"));
        }

        // NOTE: we don't currently run the ONNX models with `batch > 1`; this batches the
        // *driver loop* only so we keep the same API surface as the Python tutorial.
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch_size) {
            for text in chunk {
                results.push(self.extract_relations(text.as_ref(), relation_types, threshold)?);
            }
        }
        Ok(results)
    }

    pub fn extract(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, threshold, false, false)
    }

    pub fn extract_with_confidence(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, threshold, true, false)
    }

    pub fn extract_with_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, threshold, false, true)
    }

    pub fn extract_with_confidence_and_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, threshold, true, true)
    }

    fn extract_internal(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        let mut out = ExtractionResult::default();

        if !schema.structures.is_empty() {
            out.structures = self.extract_structures_with_specs(
                text,
                &schema.structures,
                threshold,
                include_confidence,
                include_spans,
            )?;
        }

        if !schema.relations.is_empty() {
            out.relations = self.extract_relations_with_specs(
                text,
                &schema.relations,
                threshold,
                include_confidence,
                include_spans,
            )?;
        }

        if !schema.entities.is_empty() {
            let entity_matches =
                self.extract_entities_with_specs(text, &schema.entities, threshold)?;
            let mut entities: BTreeMap<String, _> = BTreeMap::new();
            for (idx, m) in entity_matches.into_iter().enumerate() {
                let spec = schema
                    .entities
                    .get(idx)
                    .context("entity spec index mismatch")?;
                entities.insert(
                    m.label,
                    format_entity_spans(
                        &m.spans,
                        spec.dtype.clone(),
                        include_confidence,
                        include_spans,
                    ),
                );
            }
            out.entities = entities;
        }

        if !schema.classifications.is_empty() {
            let classifier = self
                .classifier
                .as_ref()
                .context("classifier not loaded; call Gliner2Pipeline::with_classifier(...)")?;

            let token_spans = self.preprocessing.tokenize(text, true);
            let text_tokens: Vec<String> = token_spans.iter().map(|t| t.token.clone()).collect();

            let schema_tokens_list: Vec<Vec<String>> = schema
                .classifications
                .iter()
                .map(|cls| {
                    if cls.label_descriptions.is_empty() {
                        build_classification_schema_tokens(
                            &cls.task,
                            &cls.labels,
                            cls.prompt.as_deref(),
                        )
                    } else {
                        build_classification_schema_tokens_with_descriptions(
                            &cls.task,
                            &cls.labels,
                            cls.prompt.as_deref(),
                            &cls.label_descriptions,
                        )
                    }
                })
                .collect();

            let formatted =
                format_input_with_mapping(&self.tokenizer, &schema_tokens_list, &text_tokens)?;
            let seq_len = formatted.input_ids.len();

            let input_ids = Array2::from_shape_vec((1, seq_len), formatted.input_ids.clone())?;
            let attention_mask =
                Array2::from_shape_vec((1, seq_len), formatted.attention_mask.clone())?;

            let hidden = self.encoder.infer(input_ids, attention_mask)?;
            let extracted = extract_embeddings(&hidden, &formatted, schema_tokens_list.len())?;

            for (schema_idx, cls) in schema.classifications.iter().enumerate() {
                let schema_emb = extracted
                    .schema_embs
                    .get(schema_idx)
                    .context("missing schema embeddings for classification schema")?;
                if schema_emb.len() < 2 {
                    return Err(anyhow!(
                        "classification schema embeddings are missing label markers; expected [P] + [L]*"
                    ));
                }

                let label_embs = &schema_emb[1..];
                if label_embs.len() != cls.labels.len() {
                    return Err(anyhow!(
                        "expected {} label embeddings but got {}; check schema tokenization",
                        cls.labels.len(),
                        label_embs.len()
                    ));
                }

                let hidden_size = label_embs[0].len();
                let mut cls_embeds = Array2::<f32>::zeros((label_embs.len(), hidden_size));
                for (row, emb) in label_embs.iter().enumerate() {
                    cls_embeds.row_mut(row).assign(emb);
                }

                let logits = classifier.infer(cls_embeds)?;
                let cls_out = decode_classification(
                    &cls.labels,
                    logits.as_slice().expect("contiguous logits"),
                    cls.multi_label,
                    cls.cls_threshold,
                    cls.class_act,
                );

                out.classifications
                    .insert(cls.task.clone(), cls_out.format(include_confidence));
            }
        }

        Ok(out)
    }

    pub fn classify_text(
        &self,
        text: &str,
        tasks: &BTreeMap<String, QuickClassificationTask>,
        threshold: f32,
        include_confidence: bool,
    ) -> Result<BTreeMap<String, FormattedClassification>> {
        let mut builder = SchemaBuilder::new();
        for (task_name, task) in tasks {
            builder = match task {
                QuickClassificationTask::Labels(labels) => {
                    builder.classification(task_name.clone(), labels.clone())
                }
                QuickClassificationTask::Config { labels, options } => builder
                    .classification_with_options(
                        task_name.clone(),
                        labels.clone(),
                        options.clone(),
                    ),
            };
        }

        let schema = builder.build();
        let out = if include_confidence {
            self.extract_with_confidence(text, &schema, threshold)?
        } else {
            self.extract(text, &schema, threshold)?
        };

        Ok(out.classifications)
    }

    pub fn classify(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
    ) -> Result<ClassificationOutput> {
        self.classify_with_options(
            text,
            task,
            labels,
            multi_label,
            cls_threshold,
            ClassAct::Auto,
        )
    }

    pub fn classify_with_options(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        let schema_tokens = build_classification_schema_tokens(task, labels, None);
        self.classify_from_schema_tokens(
            text,
            schema_tokens,
            labels,
            multi_label,
            cls_threshold,
            class_act,
        )
    }

    pub fn classify_with_descriptions(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        label_descriptions: &[(String, String)],
        multi_label: bool,
        cls_threshold: f32,
    ) -> Result<ClassificationOutput> {
        self.classify_with_descriptions_and_options(
            text,
            task,
            labels,
            label_descriptions,
            multi_label,
            cls_threshold,
            ClassAct::Auto,
        )
    }

    /// The argument list is retained to preserve the existing public API.
    #[allow(clippy::too_many_arguments)]
    pub fn classify_with_descriptions_and_options(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        label_descriptions: &[(String, String)],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        let schema_tokens = build_classification_schema_tokens_with_descriptions(
            task,
            labels,
            None,
            label_descriptions,
        );
        self.classify_from_schema_tokens(
            text,
            schema_tokens,
            labels,
            multi_label,
            cls_threshold,
            class_act,
        )
    }

    fn classify_from_schema_tokens(
        &self,
        text: &str,
        schema_tokens: Vec<String>,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        let classifier = self
            .classifier
            .as_ref()
            .context("classifier not loaded; call Gliner2Pipeline::with_classifier(...)")?;

        let token_spans = self.preprocessing.tokenize(text, true);
        let text_tokens: Vec<String> = token_spans.iter().map(|t| t.token.clone()).collect();

        let schema_tokens_list = vec![schema_tokens];

        let formatted =
            format_input_with_mapping(&self.tokenizer, &schema_tokens_list, &text_tokens)?;
        let seq_len = formatted.input_ids.len();

        let input_ids = Array2::from_shape_vec((1, seq_len), formatted.input_ids.clone())?;
        let attention_mask =
            Array2::from_shape_vec((1, seq_len), formatted.attention_mask.clone())?;

        let hidden = self.encoder.infer(input_ids, attention_mask)?;
        let extracted = extract_embeddings(&hidden, &formatted, schema_tokens_list.len())?;

        let schema0 = extracted
            .schema_embs
            .first()
            .context("missing schema embeddings")?;
        if schema0.len() < 2 {
            return Err(anyhow!(
                "classification schema embeddings are missing label markers; expected [P] + [L]*"
            ));
        }

        let label_embs = &schema0[1..];
        if label_embs.len() != labels.len() {
            return Err(anyhow!(
                "expected {} label embeddings but got {}; check schema tokenization",
                labels.len(),
                label_embs.len()
            ));
        }

        let hidden_size = label_embs[0].len();
        let mut cls_embeds = Array2::<f32>::zeros((label_embs.len(), hidden_size));
        for (row, emb) in label_embs.iter().enumerate() {
            cls_embeds.row_mut(row).assign(emb);
        }

        let logits = classifier.infer(cls_embeds)?;
        Ok(decode_classification(
            labels,
            logits.as_slice().expect("contiguous logits"),
            multi_label,
            cls_threshold,
            class_act,
        ))
    }
}

/// High-level GLiNER2.5 boundary implementation.
pub use crate::boundary::pipeline::BoundaryPipeline;

/// Architecture-aware high-level pipeline. Methods delegate without coercing a
/// boundary model into the legacy span implementation.
pub enum AutoPipeline {
    Span(Box<SpanPipeline>),
    Boundary(Box<BoundaryPipeline>),
}

macro_rules! delegate_auto {
    ($pipeline:expr, $method:ident($($arg:expr),* $(,)?)) => {
        match $pipeline {
            AutoPipeline::Span(pipeline) => pipeline.$method($($arg),*),
            AutoPipeline::Boundary(pipeline) => pipeline.$method($($arg),*),
        }
    };
}

impl AutoPipeline {
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        match ModelConfig::from_dir(bundle)?.architecture {
            Architecture::Span => Ok(Self::Span(Box::new(SpanPipeline::from_dir(bundle)?))),
            Architecture::Boundary => Ok(Self::Boundary(Box::new(BoundaryPipeline::from_dir(
                bundle,
            )?))),
        }
    }

    pub fn with_classifier(self, classifier_onnx: impl AsRef<Path>) -> Result<Self> {
        match self {
            Self::Span(pipeline) => Ok(Self::Span(Box::new(
                (*pipeline).with_classifier(classifier_onnx)?,
            ))),
            Self::Boundary(pipeline) => Ok(Self::Boundary(Box::new(
                (*pipeline).with_classifier(classifier_onnx)?,
            ))),
        }
    }

    pub fn has_adapter(&self) -> bool {
        match self {
            Self::Span(pipeline) => pipeline.has_adapter(),
            Self::Boundary(pipeline) => pipeline.has_adapter(),
        }
    }

    pub fn adapter_config(&self) -> Option<&AdapterConfig> {
        match self {
            Self::Span(pipeline) => pipeline.adapter_config(),
            Self::Boundary(pipeline) => pipeline.adapter_config(),
        }
    }

    /// Override boundary overlap handling. Span behavior is intentionally
    /// unchanged and reports that the option is architecture-specific.
    pub fn set_boundary_overlap_policy(
        &mut self,
        policy: crate::boundary::decode::OverlapPolicy,
    ) -> Result<()> {
        match self {
            Self::Boundary(pipeline) => {
                pipeline.set_overlap_policy(policy);
                Ok(())
            }
            Self::Span(_) => Err(anyhow!(
                "boundary overlap policy is only available for boundary models"
            )),
        }
    }

    /// Override boundary preprocessing without changing the legacy v2 splitter.
    pub fn set_boundary_word_splitter(
        &mut self,
        splitter: crate::boundary::preprocessing::WordSplitter,
    ) -> Result<()> {
        match self {
            Self::Boundary(pipeline) => pipeline.set_word_splitter(splitter),
            Self::Span(_) => Err(anyhow!(
                "boundary word splitter is only available for boundary models"
            )),
        }
    }

    pub fn load_adapter(&mut self, adapter_dir: impl AsRef<Path>) -> Result<()> {
        delegate_auto!(self, load_adapter(adapter_dir))
    }

    pub fn unload_adapter(&mut self) -> Result<()> {
        delegate_auto!(self, unload_adapter())
    }

    pub fn infer_raw(
        &self,
        schema_tokens_list: &[Vec<String>],
        text_tokens: &[String],
    ) -> Result<ExtractorOutput> {
        match self {
            Self::Span(pipeline) => pipeline.infer_raw(schema_tokens_list, text_tokens),
            Self::Boundary(_) => Err(anyhow!(
                "AutoPipeline::infer_raw returns the legacy span-head output and is unsupported for boundary models; use typed boundary high-level methods"
            )),
        }
    }

    pub fn extract_entities(
        &self,
        text: &str,
        entity_labels: &[String],
        threshold: f32,
    ) -> Result<Vec<EntityMatches>> {
        delegate_auto!(self, extract_entities(text, entity_labels, threshold))
    }

    pub fn extract_entities_text(
        &self,
        text: &str,
        entities: impl Into<EntityLabels>,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<BTreeMap<String, FormattedEntityValue>> {
        delegate_auto!(
            self,
            extract_entities_text(text, entities, threshold, include_confidence, include_spans)
        )
    }

    pub fn extract_json(&self, text: &str, schema: &JsonSchema) -> Result<JsonExtraction> {
        delegate_auto!(self, extract_json(text, schema))
    }

    pub fn extract_json_with_confidence(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        delegate_auto!(self, extract_json_with_confidence(text, schema, threshold))
    }

    pub fn extract_json_with_spans(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        delegate_auto!(self, extract_json_with_spans(text, schema, threshold))
    }

    pub fn extract_json_with_confidence_and_spans(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
    ) -> Result<JsonExtraction> {
        delegate_auto!(
            self,
            extract_json_with_confidence_and_spans(text, schema, threshold)
        )
    }

    pub fn extract_json_with_options(
        &self,
        text: &str,
        schema: &JsonSchema,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonExtraction> {
        delegate_auto!(
            self,
            extract_json_with_options(text, schema, threshold, include_confidence, include_spans)
        )
    }

    pub fn extract_relations(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<RelationExtraction> {
        delegate_auto!(self, extract_relations(text, relation_types, threshold))
    }

    pub fn extract_relations_with_confidence(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        delegate_auto!(
            self,
            extract_relations_with_confidence(text, relation_types, threshold)
        )
    }

    pub fn extract_relations_with_spans(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        delegate_auto!(
            self,
            extract_relations_with_spans(text, relation_types, threshold)
        )
    }

    pub fn extract_relations_with_confidence_and_spans(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        delegate_auto!(
            self,
            extract_relations_with_confidence_and_spans(text, relation_types, threshold)
        )
    }

    pub fn extract_relations_with_options(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<FormattedRelationExtraction> {
        delegate_auto!(
            self,
            extract_relations_with_options(
                text,
                relation_types,
                threshold,
                include_confidence,
                include_spans
            )
        )
    }

    pub fn batch_extract_relations<T: AsRef<str>>(
        &self,
        texts: &[T],
        relation_types: &[String],
        threshold: f32,
        batch_size: usize,
    ) -> Result<Vec<RelationExtraction>> {
        delegate_auto!(
            self,
            batch_extract_relations(texts, relation_types, threshold, batch_size)
        )
    }

    pub fn extract(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        delegate_auto!(self, extract(text, schema, threshold))
    }

    pub fn extract_with_confidence(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        delegate_auto!(self, extract_with_confidence(text, schema, threshold))
    }

    pub fn extract_with_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        delegate_auto!(self, extract_with_spans(text, schema, threshold))
    }

    pub fn extract_with_confidence_and_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        delegate_auto!(
            self,
            extract_with_confidence_and_spans(text, schema, threshold)
        )
    }

    pub fn classify_text(
        &self,
        text: &str,
        tasks: &BTreeMap<String, QuickClassificationTask>,
        threshold: f32,
        include_confidence: bool,
    ) -> Result<BTreeMap<String, FormattedClassification>> {
        delegate_auto!(
            self,
            classify_text(text, tasks, threshold, include_confidence)
        )
    }

    pub fn classify(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
    ) -> Result<ClassificationOutput> {
        delegate_auto!(
            self,
            classify(text, task, labels, multi_label, cls_threshold)
        )
    }

    pub fn classify_with_options(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        delegate_auto!(
            self,
            classify_with_options(text, task, labels, multi_label, cls_threshold, class_act)
        )
    }

    pub fn classify_with_descriptions(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        label_descriptions: &[(String, String)],
        multi_label: bool,
        cls_threshold: f32,
    ) -> Result<ClassificationOutput> {
        delegate_auto!(
            self,
            classify_with_descriptions(
                text,
                task,
                labels,
                label_descriptions,
                multi_label,
                cls_threshold
            )
        )
    }

    /// The argument list is retained for source compatibility with SpanPipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn classify_with_descriptions_and_options(
        &self,
        text: &str,
        task: &str,
        labels: &[String],
        label_descriptions: &[(String, String)],
        multi_label: bool,
        cls_threshold: f32,
        class_act: ClassAct,
    ) -> Result<ClassificationOutput> {
        delegate_auto!(
            self,
            classify_with_descriptions_and_options(
                text,
                task,
                labels,
                label_descriptions,
                multi_label,
                cls_threshold,
                class_act
            )
        )
    }
}

fn required_file(bundle: &Path, name: &str) -> Result<std::path::PathBuf> {
    let path = bundle.join(name);
    if !path.is_file() {
        return Err(anyhow!(
            "span bundle missing required `{name}` at {}",
            path.display()
        ));
    }
    Ok(path)
}
