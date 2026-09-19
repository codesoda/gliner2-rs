use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array2, Axis};

use super::{
    classification::decode_classification,
    config::BoundaryRuntimeConfig,
    decode::{OverlapPolicy, QueryScores, WordOffset, decode_to_utf8, group_scored_candidates},
    marginals::{MarginalInput, MarginalModel},
    pool::build_shared_candidate_pool,
    preprocessing::{BoundaryPreprocessingPolicy, WordSplitter},
    scorer::{ScorerInput, ScorerModel},
};
use crate::{
    adapters::{AdapterConfig, read_lora_r},
    classification::{
        ClassAct, ClassificationOutput, FormattedClassification,
        build_classification_schema_tokens, build_classification_schema_tokens_with_descriptions,
    },
    classifier::Classifier,
    embeddings::{QueryKind, extract_boundary_queries, extract_embeddings},
    encoder::Encoder,
    entities::{
        EntityMatches, EntitySpan, FormattedEntityValue, build_entities_schema_tokens,
        build_entities_schema_tokens_with_descriptions,
    },
    json::{JsonExtraction, JsonSchema},
    relations::{FormattedRelationExtraction, RelationExtraction},
    schema::format_input_with_mapping,
    schema_spec::{
        ClassificationSpec, EntityLabels, EntitySpec, ExtractionResult, QuickClassificationTask,
        SchemaBuilder, SchemaSpec,
    },
    tokenizer::RuntimeTokenizer,
};

/// High-level GLiNER2.5 boundary pipeline.
///
/// M4 supports entities and classifications. Structure and relation schemas are
/// rejected before encoder inference; their prompt positions remain reserved so
/// later heads can be added without changing mixed-task encoder context.
pub struct BoundaryPipeline {
    tokenizer: RuntimeTokenizer,
    encoder: Encoder,
    base_encoder: Option<Encoder>,
    classifier: Classifier,
    marginals: MarginalModel,
    scorer: ScorerModel,
    adapter_config: Option<AdapterConfig>,
    runtime: BoundaryRuntimeConfig,
    preprocessing: BoundaryPreprocessingPolicy,
}

#[derive(Default)]
struct BoundaryOutput {
    entities: Vec<EntityMatches>,
    classifications: Vec<(String, ClassificationOutput)>,
}

impl BoundaryPipeline {
    /// Load the graphs required for boundary entity/classification inference
    /// and validate their runtime contract.
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        let runtime = BoundaryRuntimeConfig::from_dir(bundle)?;
        required_file(bundle, "tokenizer.json")?;
        let encoder = required_file(bundle, "encoder.onnx")?;
        let classifier = required_file(bundle, "classifier.onnx")?;
        let marginals = required_file(bundle, "boundary_marginals.onnx")?;
        let scorer = required_file(bundle, "boundary_scorer.onnx")?;
        let preprocessing =
            BoundaryPreprocessingPolicy::new(runtime.max_len, WordSplitter::Whitespace)
                .map_err(|error| anyhow!(error))?;

        Ok(Self {
            tokenizer: RuntimeTokenizer::from_dir(bundle).with_context(|| {
                format!(
                    "failed to load boundary tokenizer from {}",
                    bundle.display()
                )
            })?,
            encoder: Encoder::new(&encoder).with_context(|| {
                format!("failed to load boundary encoder at {}", encoder.display())
            })?,
            base_encoder: None,
            classifier: Classifier::new(&classifier).with_context(|| {
                format!(
                    "failed to load boundary classifier at {}",
                    classifier.display()
                )
            })?,
            marginals: MarginalModel::new(&marginals).with_context(|| {
                format!(
                    "failed to load boundary marginals at {}",
                    marginals.display()
                )
            })?,
            scorer: ScorerModel::new(&scorer).with_context(|| {
                format!("failed to load boundary scorer at {}", scorer.display())
            })?,
            adapter_config: None,
            runtime,
            preprocessing,
        })
    }

    /// Replace the classifier graph while retaining all boundary heads.
    pub fn with_classifier(mut self, classifier_onnx: impl AsRef<Path>) -> Result<Self> {
        self.classifier = Classifier::new(classifier_onnx)?;
        Ok(self)
    }

    /// Override the model's default entity overlap policy.
    pub fn with_overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.runtime.overlap_policy = policy;
        self
    }

    pub fn set_overlap_policy(&mut self, policy: OverlapPolicy) {
        self.runtime.overlap_policy = policy;
    }

    /// Select the boundary-only upstream word splitter.
    pub fn with_word_splitter(mut self, splitter: WordSplitter) -> Result<Self> {
        self.set_word_splitter(splitter)?;
        Ok(self)
    }

    pub fn set_word_splitter(&mut self, splitter: WordSplitter) -> Result<()> {
        self.preprocessing = BoundaryPreprocessingPolicy::new(self.runtime.max_len, splitter)
            .map_err(|error| anyhow!(error))?;
        Ok(())
    }

    pub fn overlap_policy(&self) -> OverlapPolicy {
        self.runtime.overlap_policy
    }

    pub fn word_splitter(&self) -> WordSplitter {
        self.preprocessing.splitter()
    }

    pub fn has_adapter(&self) -> bool {
        self.adapter_config.is_some()
    }

    pub fn adapter_config(&self) -> Option<&AdapterConfig> {
        self.adapter_config.as_ref()
    }

    /// Swap in an adapter-merged encoder. The first swap retains the base
    /// session so unloading remains lossless and O(1).
    pub fn load_adapter(&mut self, adapter_dir: impl AsRef<Path>) -> Result<()> {
        let adapter_dir = adapter_dir.as_ref();
        let encoder_onnx = adapter_dir.join("encoder.onnx");
        ensure!(
            encoder_onnx.is_file(),
            "adapter bundle missing `encoder.onnx` at {} (export a merged ONNX encoder for this adapter)",
            encoder_onnx.display()
        );
        let replacement = Encoder::new(&encoder_onnx)?;
        let previous = std::mem::replace(&mut self.encoder, replacement);
        if self.base_encoder.is_none() {
            self.base_encoder = Some(previous);
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
            .context("adapter is loaded but base boundary encoder was not saved")?;
        let _adapter = std::mem::replace(&mut self.encoder, base);
        self.adapter_config = None;
        Ok(())
    }

    pub fn extract_entities(
        &self,
        text: &str,
        entity_labels: &[String],
        threshold: f32,
    ) -> Result<Vec<EntityMatches>> {
        let schema = SchemaBuilder::new()
            .entities(entity_labels.to_vec())
            .build();
        Ok(self.run_schema(text, &schema, threshold)?.entities)
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
        Ok(self
            .extract_internal(text, &schema, threshold, include_confidence, include_spans)?
            .entities)
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
        let raw = self.run_schema(text, schema, threshold)?;
        let mut result = ExtractionResult::default();
        for (index, matches) in raw.entities.into_iter().enumerate() {
            let spec = schema
                .entities
                .get(index)
                .context("boundary entity result/spec index mismatch")?;
            result.entities.insert(
                matches.label,
                format_boundary_entity_spans(
                    &matches.spans,
                    spec.dtype.clone(),
                    include_confidence,
                    include_spans,
                ),
            );
        }
        for (task, output) in raw.classifications {
            result
                .classifications
                .insert(task, output.format(include_confidence));
        }
        Ok(result)
    }

    fn run_schema(
        &self,
        text: &str,
        schema: &SchemaSpec,
        default_threshold: f32,
    ) -> Result<BoundaryOutput> {
        reject_pending_schema(schema)?;
        validate_entity_specs(&schema.entities, default_threshold)?;

        // Upstream task order is structures, entities, relations,
        // classifications. Unsupported families were rejected above, but the
        // supported families are still encoded together in their assigned
        // order in one encoder invocation.
        let mut schema_tokens = Vec::new();
        let entity_schema_idx = if schema.entities.is_empty() {
            None
        } else {
            let labels: Vec<_> = schema
                .entities
                .iter()
                .map(|entity| entity.name.clone())
                .collect();
            let descriptions: Vec<_> = schema
                .entities
                .iter()
                .filter_map(|entity| {
                    entity
                        .description
                        .as_ref()
                        .map(|description| (entity.name.clone(), description.clone()))
                })
                .collect();
            let tokens = if descriptions.is_empty() {
                build_entities_schema_tokens(&labels, None)
            } else {
                build_entities_schema_tokens_with_descriptions(&labels, None, &descriptions)
            };
            let index = schema_tokens.len();
            schema_tokens.push(tokens);
            Some(index)
        };

        let mut classification_schema_indices = Vec::with_capacity(schema.classifications.len());
        for classification in &schema.classifications {
            ensure!(
                !classification.labels.is_empty(),
                "classification task {:?} must contain at least one label",
                classification.task
            );
            let tokens = if classification.label_descriptions.is_empty() {
                build_classification_schema_tokens(
                    &classification.task,
                    &classification.labels,
                    classification.prompt.as_deref(),
                )
            } else {
                build_classification_schema_tokens_with_descriptions(
                    &classification.task,
                    &classification.labels,
                    classification.prompt.as_deref(),
                    &classification.label_descriptions,
                )
            };
            classification_schema_indices.push(schema_tokens.len());
            schema_tokens.push(tokens);
        }

        let prepared = self.preprocessing.prepare(text, &[]);
        let formatted =
            format_input_with_mapping(&self.tokenizer, &schema_tokens, &prepared.text_tokens)?;
        let sequence = formatted.input_ids.len();
        let input_ids = Array2::from_shape_vec((1, sequence), formatted.input_ids.clone())?;
        let attention_mask =
            Array2::from_shape_vec((1, sequence), formatted.attention_mask.clone())?;
        let hidden = self.encoder.infer(input_ids, attention_mask)?;
        let extracted = extract_embeddings(&hidden, &formatted, schema_tokens.len())?;
        ensure!(
            extracted.text_emb.nrows() == prepared.text_tokens.len(),
            "boundary text pooling produced {} words for {} prepared tokens",
            extracted.text_emb.nrows(),
            prepared.text_tokens.len()
        );

        let mut output = BoundaryOutput::default();
        for (classification, &schema_index) in schema
            .classifications
            .iter()
            .zip(&classification_schema_indices)
        {
            let embeddings = extracted
                .schema_embs
                .get(schema_index)
                .context("missing boundary classification schema embeddings")?;
            ensure!(
                embeddings.len() == classification.labels.len() + 1,
                "classification task {:?} expected [P] plus {} [L] states, got {} marker states",
                classification.task,
                classification.labels.len(),
                embeddings.len()
            );
            let hidden_size = extracted.text_emb.ncols();
            let mut label_states = Array2::zeros((classification.labels.len(), hidden_size));
            for (row, embedding) in embeddings.iter().skip(1).enumerate() {
                ensure!(
                    embedding.len() == hidden_size,
                    "classification label state width {} differs from encoder width {hidden_size}",
                    embedding.len()
                );
                label_states.row_mut(row).assign(embedding);
            }
            let logits = self.classifier.infer(label_states)?;
            let decoded = decode_classification(
                &classification.labels,
                logits
                    .as_slice()
                    .context("classifier logits are not contiguous")?,
                classification.multi_label,
                classification.cls_threshold,
                classification.class_act,
                self.runtime.classification_temperature,
            )?;
            output
                .classifications
                .push((classification.task.clone(), decoded));
        }

        let queries = extract_boundary_queries(&hidden, &formatted, schema_tokens.len())?;
        if queries.metadata.is_empty() {
            // Classification-only and empty-schema requests intentionally never
            // enter the Q=0 marginal/pool/scorer graphs.
            return Ok(output);
        }

        let expected_entity_schema = entity_schema_idx
            .context("boundary query markers exist without a supported entity schema")?;
        ensure!(
            queries.metadata.len() == schema.entities.len(),
            "boundary entity query count {} differs from schema field count {}",
            queries.metadata.len(),
            schema.entities.len()
        );
        for (index, metadata) in queries.metadata.iter().enumerate() {
            ensure!(
                metadata.kind == QueryKind::Entity
                    && metadata.schema_idx == expected_entity_schema
                    && metadata.field_idx == index,
                "unexpected M4 boundary query routing at index {index}: {metadata:?}"
            );
        }

        let text_length = extracted.text_emb.nrows();
        let query_count = queries.query_emb.nrows();
        let text_states = extracted.text_emb.insert_axis(Axis(0));
        let query_states = queries.query_emb.insert_axis(Axis(0));
        let text_mask = Array2::from_elem((1, text_length), true);
        let query_mask = Array2::from_elem((1, query_count), true);
        let marginal = self.marginals.infer(MarginalInput {
            text_states: text_states.clone(),
            text_mask: text_mask.clone(),
            query_states: query_states.clone(),
            query_mask: query_mask.clone(),
        })?;

        let pool = build_shared_candidate_pool(
            marginal.boundary_mask.index_axis(Axis(0), 0),
            query_mask.index_axis(Axis(0), 0),
            marginal.start_logits.index_axis(Axis(0), 0),
            marginal.end_logits.index_axis(Axis(0), 0),
            marginal.start_all.index_axis(Axis(0), 0),
            marginal.end_all.index_axis(Axis(0), 0),
            self.runtime.pool,
        )?;
        let scorer = self.scorer.infer(ScorerInput {
            boundary_states: marginal.boundary_states,
            text_states,
            text_mask,
            query_states,
            query_mask: query_mask.clone(),
            start_logits: marginal.start_logits,
            end_logits: marginal.end_logits,
            inside_prefix: marginal.inside_prefix,
            inside_prefix_mean: marginal.inside_prefix_mean,
            candidate_indices: pool.indices.clone().insert_axis(Axis(0)),
            candidate_mask: pool.mask.clone().insert_axis(Axis(0)),
            candidate_compat: pool.compat_logits.clone().insert_axis(Axis(0)),
        })?;

        let candidate_count = pool.indices.nrows();
        let indices: Vec<[usize; 2]> = (0..candidate_count)
            .map(|candidate| {
                Ok([
                    usize::try_from(pool.indices[(candidate, 0)])?,
                    usize::try_from(pool.indices[(candidate, 1)])?,
                ])
            })
            .collect::<Result<_>>()?;
        let valid_mask: Vec<_> = pool.mask.iter().copied().collect();
        let grouped_inputs: Vec<_> = schema
            .entities
            .iter()
            .enumerate()
            .map(|(query, entity)| QueryScores {
                indices: indices.clone(),
                valid_mask: valid_mask.clone(),
                pair_logits: scorer
                    .pair_logits
                    .index_axis(Axis(0), 0)
                    .index_axis(Axis(0), query)
                    .iter()
                    .copied()
                    .collect(),
                query_valid: query_mask[(0, query)],
                threshold: entity.threshold.unwrap_or(default_threshold),
                null_logit: Some(scorer.null_logits[(0, query)]),
            })
            .collect();
        let grouped = group_scored_candidates(
            &grouped_inputs,
            self.runtime.pair_temperature,
            self.runtime.abstention_threshold,
        )?;
        let offsets: Vec<_> = prepared
            .original_offsets
            .iter()
            .map(|offset| WordOffset {
                start: offset.start,
                end: offset.end,
            })
            .collect();

        for (entity, candidates) in schema.entities.iter().zip(grouped) {
            let decoded = decode_to_utf8(
                &prepared.original_text,
                &offsets,
                &candidates,
                self.runtime.overlap_policy,
                prepared.choice_prefix_words,
            )?;
            output.entities.push(EntityMatches {
                label: entity.name.clone(),
                spans: decoded
                    .into_iter()
                    .map(|span| EntitySpan {
                        start: span.start,
                        end: span.end,
                        text: span.text,
                        score: span.confidence,
                    })
                    .collect(),
            });
        }

        Ok(output)
    }

    pub fn classify_text(
        &self,
        text: &str,
        tasks: &BTreeMap<String, QuickClassificationTask>,
        threshold: f32,
        include_confidence: bool,
    ) -> Result<BTreeMap<String, FormattedClassification>> {
        let mut builder = SchemaBuilder::new();
        for (task, labels) in tasks {
            builder = match labels {
                QuickClassificationTask::Labels(labels) => {
                    builder.classification(task.clone(), labels.clone())
                }
                QuickClassificationTask::Config { labels, options } => builder
                    .classification_with_options(task.clone(), labels.clone(), options.clone()),
            };
        }
        Ok(self
            .extract_internal(text, &builder.build(), threshold, include_confidence, false)?
            .classifications)
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
        self.classify_with_descriptions_and_options(
            text,
            task,
            labels,
            &[],
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
        let schema = SchemaSpec {
            classifications: vec![ClassificationSpec {
                task: task.to_owned(),
                labels: labels.to_vec(),
                label_descriptions: label_descriptions.to_vec(),
                multi_label,
                cls_threshold,
                class_act,
                prompt: None,
            }],
            ..SchemaSpec::default()
        };
        let mut outputs = self.run_schema(text, &schema, 0.5)?.classifications;
        ensure!(
            outputs.len() == 1,
            "single classification request produced {} task outputs",
            outputs.len()
        );
        Ok(outputs.remove(0).1)
    }

    pub fn extract_json(&self, _text: &str, _schema: &JsonSchema) -> Result<JsonExtraction> {
        Err(structures_pending())
    }

    pub fn extract_json_with_confidence(
        &self,
        _text: &str,
        _schema: &JsonSchema,
        _threshold: f32,
    ) -> Result<JsonExtraction> {
        Err(structures_pending())
    }

    pub fn extract_json_with_spans(
        &self,
        _text: &str,
        _schema: &JsonSchema,
        _threshold: f32,
    ) -> Result<JsonExtraction> {
        Err(structures_pending())
    }

    pub fn extract_json_with_confidence_and_spans(
        &self,
        _text: &str,
        _schema: &JsonSchema,
        _threshold: f32,
    ) -> Result<JsonExtraction> {
        Err(structures_pending())
    }

    pub fn extract_json_with_options(
        &self,
        _text: &str,
        _schema: &JsonSchema,
        _threshold: f32,
        _include_confidence: bool,
        _include_spans: bool,
    ) -> Result<JsonExtraction> {
        Err(structures_pending())
    }

    pub fn extract_relations(
        &self,
        _text: &str,
        _relation_types: &[String],
        _threshold: f32,
    ) -> Result<RelationExtraction> {
        Err(relations_pending())
    }

    pub fn extract_relations_with_confidence(
        &self,
        _text: &str,
        _relation_types: &[String],
        _threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        Err(relations_pending())
    }

    pub fn extract_relations_with_spans(
        &self,
        _text: &str,
        _relation_types: &[String],
        _threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        Err(relations_pending())
    }

    pub fn extract_relations_with_confidence_and_spans(
        &self,
        _text: &str,
        _relation_types: &[String],
        _threshold: f32,
    ) -> Result<FormattedRelationExtraction> {
        Err(relations_pending())
    }

    pub fn extract_relations_with_options(
        &self,
        _text: &str,
        _relation_types: &[String],
        _threshold: f32,
        _include_confidence: bool,
        _include_spans: bool,
    ) -> Result<FormattedRelationExtraction> {
        Err(relations_pending())
    }

    pub fn batch_extract_relations<T: AsRef<str>>(
        &self,
        _texts: &[T],
        _relation_types: &[String],
        _threshold: f32,
        _batch_size: usize,
    ) -> Result<Vec<RelationExtraction>> {
        Err(relations_pending())
    }
}

fn format_boundary_entity_spans(
    spans: &[EntitySpan],
    dtype: crate::schema_spec::FieldDtype,
    include_confidence: bool,
    include_spans: bool,
) -> FormattedEntityValue {
    // Boundary runtime preserves distinct occurrences even when their surface
    // text is identical. This intentionally differs from the legacy v2 helper,
    // whose formatter deduplicates case-insensitive text.
    match dtype {
        crate::schema_spec::FieldDtype::List => FormattedEntityValue::List(
            spans
                .iter()
                .map(|span| span.format(include_confidence, include_spans))
                .collect(),
        ),
        crate::schema_spec::FieldDtype::Str => FormattedEntityValue::Single(
            spans
                .first()
                .map(|span| span.format(include_confidence, include_spans)),
        ),
    }
}

fn validate_entity_specs(entities: &[EntitySpec], default_threshold: f32) -> Result<()> {
    for entity in entities {
        let threshold = entity.threshold.unwrap_or(default_threshold);
        ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "entity threshold for {:?} must be finite and in [0,1], got {threshold}",
            entity.name
        );
    }
    Ok(())
}

fn reject_pending_schema(schema: &SchemaSpec) -> Result<()> {
    if !schema.structures.is_empty() {
        return Err(structures_pending());
    }
    if !schema.relations.is_empty() {
        return Err(relations_pending());
    }
    Ok(())
}

fn structures_pending() -> anyhow::Error {
    anyhow!(
        "boundary structure/JSON extraction is a pending feature (milestone M5); request rejected before inference"
    )
}

fn relations_pending() -> anyhow::Error {
    anyhow!(
        "boundary relation extraction is a pending feature (milestone M6); request rejected before inference"
    )
}

fn required_file(bundle: &Path, name: &str) -> Result<std::path::PathBuf> {
    let path = bundle.join(name);
    ensure!(
        path.is_file(),
        "boundary bundle missing required `{name}` at {}",
        path.display()
    );
    Ok(path)
}
