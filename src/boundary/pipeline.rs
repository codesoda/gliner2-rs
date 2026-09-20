use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, anyhow, ensure};
use ndarray::{Array1, Array2, Array3, Array4, Axis, s};

use super::{
    choices::{literal_choice_mentions, match_choice_surface, nearest_choice, present_choices},
    classification::decode_classification,
    config::BoundaryRuntimeConfig,
    decode::{
        OverlapPolicy, QueryScores, ScoredSpan, WordOffset, decode_to_utf8,
        group_scored_candidates, map_half_open_utf8, resolve_overlaps, sigmoid_probability,
    },
    explicit::{ExplicitInput, ExplicitModel},
    explicit_spans::{ExplicitSpanScore, ExplicitSpanScores, map_byte_spans},
    marginals::{MarginalInput, MarginalModel, MarginalOutput},
    pool::{CandidatePool, build_shared_candidate_pool},
    preprocessing::{BoundaryPreprocessingPolicy, PreparedTokens, WordSplitter},
    record_decode::{Cardinality, DecodedRecord, decode_group},
    record_prepare::prepare_record_group,
    record_schema::{
        CompiledRecordSpec, RecordMetadata, compile_record_specs, validate_record_metadata,
    },
    records::RecordModel,
    relation_decode::{RelationEdge, RelationMention, deduplicate_relation_edges},
    relation_pairs::{RelationPair, RelationTypeSpec, generate_relation_pairs},
    relations::{RelationInput, RelationModel},
    scorer::{ScorerInput, ScorerModel, ScorerOutput},
};
use crate::{
    adapters::{AdapterConfig, read_lora_r},
    classification::{
        ClassAct, ClassificationOutput, FormattedClassification,
        build_classification_schema_tokens, build_classification_schema_tokens_with_descriptions,
    },
    classifier::Classifier,
    embeddings::{
        BoundaryQueryEmbeddings, ExtractedEmbeddings, QueryKind, QueryMetadata,
        extract_boundary_queries, extract_embeddings,
    },
    encoder::Encoder,
    entities::{
        EntityMatches, EntitySpan, FormattedEntityValue, build_entities_schema_tokens,
        build_entities_schema_tokens_with_descriptions,
    },
    json::{JsonExtraction, JsonRecord, JsonSchema},
    relations::{
        FormattedRelationExtraction, FormattedRelationPair, RelationExtraction,
        build_relation_schema_tokens,
    },
    schema::format_input_with_mapping,
    schema_spec::{
        ClassificationSpec, EntityLabels, ExtractionResult, FieldDtype, QuickClassificationTask,
        SchemaBuilder, SchemaSpec, StructureSpec,
    },
    structures::{build_structure_choice_prefix, build_structure_schema_tokens},
    tokenizer::RuntimeTokenizer,
};

/// High-level GLiNER2.5 boundary pipeline.
///
/// Supports entities, classifications, relations, legacy JSON structures, and
/// opt-in record formation.
pub struct BoundaryPipeline {
    tokenizer: RuntimeTokenizer,
    encoder: Encoder,
    base_encoder: Option<Encoder>,
    classifier: Classifier,
    marginals: MarginalModel,
    scorer: ScorerModel,
    explicit: ExplicitModel,
    records: RecordModel,
    relations: RelationModel,
    adapter_config: Option<AdapterConfig>,
    runtime: BoundaryRuntimeConfig,
    preprocessing: BoundaryPreprocessingPolicy,
}

#[derive(Default)]
struct BoundaryOutput {
    entities: Vec<EntityMatches>,
    classifications: Vec<(String, ClassificationOutput)>,
    structures: Vec<RawStructureOutput>,
    relations: Vec<RawRelationOutput>,
}

struct RawStructureOutput {
    name: String,
    instances: Vec<Vec<RawStructureField>>,
}

struct RawRelationOutput {
    name: String,
    edges: Vec<RelationEdge>,
}

struct RawStructureField {
    name: String,
    dtype: FieldDtype,
    value: RawStructureValue,
}

enum RawStructureValue {
    Spans(Vec<EntitySpan>),
    Choices(Vec<(String, f32)>),
}

struct PreparedBoundaryEncoding {
    prepared: PreparedTokens,
    embeddings: ExtractedEmbeddings,
    queries: BoundaryQueryEmbeddings,
}

impl BoundaryPipeline {
    /// Load and validate the boundary entity, classification, JSON/record and
    /// explicit-scoring graphs.
    pub fn from_dir(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        let runtime = BoundaryRuntimeConfig::from_dir(bundle)?;
        required_file(bundle, "tokenizer.json")?;
        let encoder = required_file(bundle, "encoder.onnx")?;
        let classifier = required_file(bundle, "classifier.onnx")?;
        let marginals = required_file(bundle, "boundary_marginals.onnx")?;
        let scorer = required_file(bundle, "boundary_scorer.onnx")?;
        let explicit = required_file(bundle, "boundary_explicit_scorer.onnx")?;
        let records = required_file(bundle, "boundary_records.onnx")?;
        let relations = required_file(bundle, "boundary_relations.onnx")?;
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
            explicit: ExplicitModel::new(&explicit).with_context(|| {
                format!(
                    "failed to load boundary explicit scorer at {}",
                    explicit.display()
                )
            })?,
            records: RecordModel::new(&records).with_context(|| {
                format!(
                    "failed to load boundary record head at {}",
                    records.display()
                )
            })?,
            relations: RelationModel::new(&relations).with_context(|| {
                format!(
                    "failed to load boundary relation head at {}",
                    relations.display()
                )
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
        Ok(self
            .run_schema(text, &schema, &RecordMetadata::new(), threshold)?
            .entities)
    }

    /// Score caller-supplied original-text spans for every ordered label query.
    ///
    /// Spans are nonempty half-open UTF-8 byte ranges and must exactly align to
    /// retained boundary-model words. This path performs no proposal pooling,
    /// thresholding, abstention, overlap resolution, or deduplication.
    pub fn score_explicit_spans(
        &self,
        text: &str,
        labels: &[String],
        spans: &[[usize; 2]],
    ) -> Result<Vec<ExplicitSpanScores>> {
        let prepared = self.preprocessing.prepare(text, &[]);
        // Preflight every supplied span before encoder/native-head work, even
        // when there are no labels.
        let mapped = map_byte_spans(&prepared, spans)?;

        if labels.is_empty() {
            return Ok(Vec::new());
        }
        if spans.is_empty() {
            return Ok(labels
                .iter()
                .cloned()
                .map(|label| ExplicitSpanScores {
                    label,
                    spans: Vec::new(),
                })
                .collect());
        }

        let query_count = labels.len();
        let candidate_count = mapped.len();
        let coordinate_count = query_count
            .checked_mul(candidate_count)
            .and_then(|count| count.checked_mul(2))
            .context("explicit-span candidate tensor size overflow")?;

        // Do not route through SchemaBuilder: duplicate labels are real,
        // ordered schema queries in this API.
        let schema_tokens = vec![build_entities_schema_tokens(labels, None)];
        let encoding = self.encode_prepared(prepared, &schema_tokens)?;
        let expected_metadata: Vec<_> = (0..labels.len())
            .map(|field_idx| QueryMetadata {
                schema_idx: 0,
                field_idx,
                kind: QueryKind::Entity,
            })
            .collect();
        ensure!(
            encoding.queries.metadata == expected_metadata,
            "explicit-span query routing mismatch: actual={:?}, expected={expected_metadata:?}",
            encoding.queries.metadata
        );
        ensure!(
            encoding.queries.query_emb.nrows() == labels.len()
                && encoding.queries.query_emb.ncols() == encoding.embeddings.text_emb.ncols(),
            "explicit-span query shape {:?} does not match {} labels and text width {}",
            encoding.queries.query_emb.dim(),
            labels.len(),
            encoding.embeddings.text_emb.ncols()
        );

        let mut coordinates = Vec::with_capacity(coordinate_count);
        for _ in 0..query_count {
            for &[start, end] in &mapped {
                coordinates.push(start);
                coordinates.push(end);
            }
        }
        ensure!(
            coordinates.len() == coordinate_count,
            "explicit-span candidate tensor size changed during construction"
        );

        let marginal =
            self.run_marginals(&encoding.embeddings.text_emb, &encoding.queries.query_emb)?;
        let output = self.explicit.infer(ExplicitInput {
            boundary_states: marginal.boundary_states,
            text_states: encoding
                .embeddings
                .text_emb
                .view()
                .insert_axis(Axis(0))
                .to_owned(),
            text_mask: Array2::from_elem((1, encoding.embeddings.text_emb.nrows()), true),
            query_states: encoding
                .queries
                .query_emb
                .view()
                .insert_axis(Axis(0))
                .to_owned(),
            query_mask: Array2::from_elem((1, query_count), true),
            start_logits: marginal.start_logits,
            end_logits: marginal.end_logits,
            inside_prefix: marginal.inside_prefix,
            inside_prefix_mean: marginal.inside_prefix_mean,
            candidate_indices: Array4::from_shape_vec(
                (1, query_count, candidate_count, 2),
                coordinates,
            )?,
            candidate_mask: Array3::from_elem((1, query_count, candidate_count), true),
        })?;
        ensure!(
            output.legal_mask.iter().all(|&legal| legal),
            "explicit scorer rejected a preflighted caller span"
        );

        labels
            .iter()
            .enumerate()
            .map(|(query, label)| {
                let scored = spans
                    .iter()
                    .enumerate()
                    .map(|(candidate, &[start, end])| {
                        let logit = output.pair_logits[(0, query, candidate)];
                        Ok(ExplicitSpanScore {
                            start,
                            end,
                            text: text
                                .get(start..end)
                                .context("preflighted explicit span is not a UTF-8 source slice")?
                                .to_owned(),
                            logit,
                            confidence: sigmoid_probability(logit, self.runtime.pair_temperature)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ExplicitSpanScores {
                    label: label.clone(),
                    spans: scored,
                })
            })
            .collect()
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
            .extract_internal(
                text,
                &schema,
                &RecordMetadata::new(),
                threshold,
                include_confidence,
                include_spans,
            )?
            .entities)
    }

    pub fn extract(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(
            text,
            schema,
            &RecordMetadata::new(),
            threshold,
            false,
            false,
        )
    }

    pub fn extract_with_confidence(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, &RecordMetadata::new(), threshold, true, false)
    }

    pub fn extract_with_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, &RecordMetadata::new(), threshold, false, true)
    }

    pub fn extract_with_confidence_and_spans(
        &self,
        text: &str,
        schema: &SchemaSpec,
        threshold: f32,
    ) -> Result<ExtractionResult> {
        self.extract_internal(text, schema, &RecordMetadata::new(), threshold, true, true)
    }

    /// Extract a combined schema with opt-in record formation metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_with_records(
        &self,
        text: &str,
        schema: &SchemaSpec,
        metadata: &RecordMetadata,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        self.extract_internal(
            text,
            schema,
            metadata,
            threshold,
            include_confidence,
            include_spans,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_internal(
        &self,
        text: &str,
        schema: &SchemaSpec,
        metadata: &RecordMetadata,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<ExtractionResult> {
        let raw = self.run_schema(text, schema, metadata, threshold)?;
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
        for structure in raw.structures {
            let mut instances = Vec::with_capacity(structure.instances.len());
            for fields in structure.instances {
                let mut instance = JsonRecord::new();
                for field in fields {
                    let value = match field.value {
                        RawStructureValue::Spans(spans) => format_boundary_entity_spans(
                            &spans,
                            field.dtype,
                            include_confidence,
                            include_spans,
                        ),
                        RawStructureValue::Choices(choices) => {
                            format_boundary_choices(&choices, field.dtype, include_confidence)
                        }
                    };
                    instance.insert(field.name, value);
                }
                instances.push(instance);
            }
            result.structures.insert(structure.name, instances);
        }
        for relation in raw.relations {
            let pairs = relation
                .edges
                .into_iter()
                .map(|edge| FormattedRelationPair {
                    head: EntitySpan {
                        start: edge.head.start,
                        end: edge.head.end,
                        text: edge.head.text,
                        score: edge.score,
                    }
                    .format(include_confidence, include_spans),
                    tail: EntitySpan {
                        start: edge.tail.start,
                        end: edge.tail.end,
                        text: edge.tail.text,
                        score: edge.score,
                    }
                    .format(include_confidence, include_spans),
                })
                .collect();
            result.relations.insert(relation.name, pairs);
        }
        Ok(result)
    }

    fn encode_schema(
        &self,
        text: &str,
        schema_tokens: &[Vec<String>],
        choice_prefix_tokens: &[String],
    ) -> Result<PreparedBoundaryEncoding> {
        let prepared = self.preprocessing.prepare(text, choice_prefix_tokens);
        self.encode_prepared(prepared, schema_tokens)
    }

    fn encode_prepared(
        &self,
        prepared: PreparedTokens,
        schema_tokens: &[Vec<String>],
    ) -> Result<PreparedBoundaryEncoding> {
        let formatted =
            format_input_with_mapping(&self.tokenizer, schema_tokens, &prepared.text_tokens)?;
        let sequence = formatted.input_ids.len();
        ensure!(
            formatted.attention_mask.len() == sequence,
            "boundary formatted input has {sequence} token IDs but {} attention values",
            formatted.attention_mask.len()
        );
        let input_ids = Array2::from_shape_vec((1, sequence), formatted.input_ids.clone())?;
        let attention_mask =
            Array2::from_shape_vec((1, sequence), formatted.attention_mask.clone())?;
        let hidden = self.encoder.infer(input_ids, attention_mask)?;
        let (batch, hidden_sequence, hidden_size) = hidden.dim();
        ensure!(
            batch == 1 && hidden_sequence == sequence && hidden_size > 0,
            "boundary encoder output shape {:?}, expected [1,{sequence},H] with H > 0",
            hidden.shape()
        );

        let embeddings = extract_embeddings(&hidden, &formatted, schema_tokens.len())?;
        ensure!(
            embeddings.schema_embs.len() == schema_tokens.len(),
            "boundary schema pooling produced {} schema groups for {} schemas",
            embeddings.schema_embs.len(),
            schema_tokens.len()
        );
        ensure!(
            embeddings.text_emb.dim() == (prepared.text_tokens.len(), hidden_size),
            "boundary text pooling produced shape {:?} for {} prepared tokens and encoder width {hidden_size}",
            embeddings.text_emb.dim(),
            prepared.text_tokens.len()
        );

        let queries = extract_boundary_queries(&hidden, &formatted, schema_tokens.len())?;
        ensure!(
            queries.query_emb.dim() == (queries.metadata.len(), hidden_size),
            "boundary query pooling produced shape {:?} for {} routed queries and encoder width {hidden_size}",
            queries.query_emb.dim(),
            queries.metadata.len()
        );

        Ok(PreparedBoundaryEncoding {
            prepared,
            embeddings,
            queries,
        })
    }

    fn run_marginals(
        &self,
        text_embeddings: &Array2<f32>,
        query_embeddings: &Array2<f32>,
    ) -> Result<MarginalOutput> {
        let text_length = text_embeddings.nrows();
        let query_count = query_embeddings.nrows();
        self.marginals.infer(MarginalInput {
            text_states: text_embeddings.view().insert_axis(Axis(0)).to_owned(),
            text_mask: Array2::from_elem((1, text_length), true),
            query_states: query_embeddings.view().insert_axis(Axis(0)).to_owned(),
            query_mask: Array2::from_elem((1, query_count), true),
        })
    }

    fn score_explicit_choices(
        &self,
        text_embeddings: &Array2<f32>,
        query_embeddings: &Array2<f32>,
        marginal: &MarginalOutput,
        query_id: usize,
        prefix_tokens: &[String],
        choices: &[String],
    ) -> Result<Vec<(String, f32)>> {
        ensure!(
            query_id < query_embeddings.nrows(),
            "explicit choice query ID {query_id} is out of range for {} queries",
            query_embeddings.nrows()
        );

        // Upstream deduplicates exact choice strings before looking up the first
        // case-insensitive, whole-prefix-token match. A multiword choice remains
        // one prefix word; substring matching would score the wrong coordinate.
        let present = present_choices(choices, prefix_tokens);
        if present.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_count = present.len();
        let mut coordinates = Vec::with_capacity(candidate_count * 2);
        for (_, index) in &present {
            coordinates.push(i64::try_from(*index).context("choice prefix index exceeds i64")?);
            coordinates
                .push(i64::try_from(index + 1).context("choice prefix end index exceeds i64")?);
        }
        let candidate_indices = Array4::from_shape_vec((1, 1, candidate_count, 2), coordinates)?;
        let output = self.explicit.infer(ExplicitInput {
            boundary_states: marginal.boundary_states.clone(),
            text_states: text_embeddings.view().insert_axis(Axis(0)).to_owned(),
            text_mask: Array2::from_elem((1, text_embeddings.nrows()), true),
            query_states: query_embeddings
                .slice(s![query_id..query_id + 1, ..])
                .insert_axis(Axis(0))
                .to_owned(),
            query_mask: Array2::from_elem((1, 1), true),
            start_logits: marginal
                .start_logits
                .slice(s![.., query_id..query_id + 1, ..])
                .to_owned(),
            end_logits: marginal
                .end_logits
                .slice(s![.., query_id..query_id + 1, ..])
                .to_owned(),
            inside_prefix: marginal
                .inside_prefix
                .slice(s![.., query_id..query_id + 1, ..])
                .to_owned(),
            inside_prefix_mean: marginal
                .inside_prefix_mean
                .slice(s![.., query_id..query_id + 1, ..])
                .to_owned(),
            candidate_indices,
            candidate_mask: Array3::from_elem((1, 1, candidate_count), true),
        })?;
        ensure!(
            output.legal_mask.iter().all(|legal| *legal),
            "explicit choice scorer rejected a schema-prefix coordinate"
        );

        present
            .into_iter()
            .enumerate()
            .map(|(index, (choice, _))| {
                let probability = sigmoid_probability(
                    output.pair_logits[(0, 0, index)],
                    self.runtime.pair_temperature,
                )?;
                Ok((choice, probability))
            })
            .collect()
    }

    fn run_shared_candidate_stage(
        &self,
        text_embeddings: &Array2<f32>,
        query_embeddings: &Array2<f32>,
        marginal: MarginalOutput,
    ) -> Result<(CandidatePool, ScorerOutput)> {
        let text_length = text_embeddings.nrows();
        let query_count = query_embeddings.nrows();
        let text_states = text_embeddings.view().insert_axis(Axis(0)).to_owned();
        let query_states = query_embeddings.view().insert_axis(Axis(0)).to_owned();
        let text_mask = Array2::from_elem((1, text_length), true);
        let query_mask = Array2::from_elem((1, query_count), true);

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
            query_mask,
            start_logits: marginal.start_logits,
            end_logits: marginal.end_logits,
            inside_prefix: marginal.inside_prefix,
            inside_prefix_mean: marginal.inside_prefix_mean,
            candidate_indices: pool.indices.clone().insert_axis(Axis(0)),
            candidate_mask: pool.mask.clone().insert_axis(Axis(0)),
            candidate_compat: pool.compat_logits.clone().insert_axis(Axis(0)),
        })?;

        Ok((pool, scorer))
    }

    fn run_schema(
        &self,
        text: &str,
        schema: &SchemaSpec,
        record_metadata: &RecordMetadata,
        default_threshold: f32,
    ) -> Result<BoundaryOutput> {
        validate_record_metadata(&schema.structures, record_metadata)?;
        validate_schema_thresholds(schema, default_threshold)?;

        // All task families share exactly one encoder invocation in upstream
        // prompt order: structures, entities, relations, classifications.
        let mut schema_tokens = Vec::new();
        let mut expected_queries = Vec::new();
        let mut structure_query_ids = Vec::with_capacity(schema.structures.len());
        let mut structure_schema_indices = Vec::with_capacity(schema.structures.len());
        let mut choice_prefix_tokens = Vec::new();
        for structure in &schema.structures {
            let schema_idx = schema_tokens.len();
            structure_schema_indices.push(schema_idx);
            schema_tokens.push(build_structure_schema_tokens(structure, None));
            choice_prefix_tokens.extend(build_structure_choice_prefix(structure));
            let mut query_ids = Vec::with_capacity(structure.fields.len());
            for field_idx in 0..structure.fields.len() {
                query_ids.push(expected_queries.len());
                expected_queries.push(QueryMetadata {
                    schema_idx,
                    field_idx,
                    kind: QueryKind::Content,
                });
            }
            structure_query_ids.push(query_ids);
        }

        let mut entity_query_ids = Vec::with_capacity(schema.entities.len());
        if !schema.entities.is_empty() {
            let schema_idx = schema_tokens.len();
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
            schema_tokens.push(tokens);
            for field_idx in 0..schema.entities.len() {
                entity_query_ids.push(expected_queries.len());
                expected_queries.push(QueryMetadata {
                    schema_idx,
                    field_idx,
                    kind: QueryKind::Entity,
                });
            }
        }

        let mut relation_query_ids = Vec::with_capacity(schema.relations.len());
        for relation in &schema.relations {
            let schema_idx = schema_tokens.len();
            schema_tokens.push(build_relation_schema_tokens(
                &relation.name,
                relation.description.as_deref(),
            ));
            let head_query_id = expected_queries.len();
            expected_queries.push(QueryMetadata {
                schema_idx,
                field_idx: 0,
                kind: QueryKind::Relation,
            });
            let tail_query_id = expected_queries.len();
            expected_queries.push(QueryMetadata {
                schema_idx,
                field_idx: 1,
                kind: QueryKind::Relation,
            });
            relation_query_ids.push([head_query_id, tail_query_id]);
        }

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

        let encoding = self.encode_schema(text, &schema_tokens, &choice_prefix_tokens)?;
        ensure!(
            encoding.queries.metadata == expected_queries,
            "boundary query routing mismatch: actual={:?}, expected={expected_queries:?}",
            encoding.queries.metadata
        );
        ensure!(
            structure_schema_indices == (0..schema.structures.len()).collect::<Vec<_>>(),
            "boundary structure schemas must occupy indices 0..{} in prompt order, got {structure_schema_indices:?}",
            schema.structures.len()
        );
        let record_specs = compile_record_specs(
            &schema.structures,
            record_metadata,
            &structure_schema_indices,
            &encoding.queries.metadata,
        )?;

        let mut output = BoundaryOutput::default();
        for (classification, &schema_index) in schema
            .classifications
            .iter()
            .zip(&classification_schema_indices)
        {
            let embeddings = encoding
                .embeddings
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
            let hidden_size = encoding.embeddings.text_emb.ncols();
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

        if encoding.queries.metadata.is_empty() {
            // Classification-only, empty-schema, and fieldless-structure
            // requests intentionally bypass the Q=0 boundary graphs.
            return Ok(output);
        }

        let marginal =
            self.run_marginals(&encoding.embeddings.text_emb, &encoding.queries.query_emb)?;
        let prefix_tokens = &encoding.prepared.text_tokens[..encoding.prepared.choice_prefix_words];
        let mut choice_scores = BTreeMap::new();
        for (structure, query_ids) in schema.structures.iter().zip(&structure_query_ids) {
            for (field, &query_id) in structure.fields.iter().zip(query_ids) {
                if !field.choices.is_empty() {
                    let scores = self.score_explicit_choices(
                        &encoding.embeddings.text_emb,
                        &encoding.queries.query_emb,
                        &marginal,
                        query_id,
                        prefix_tokens,
                        &field.choices,
                    )?;
                    choice_scores.insert(query_id, scores);
                }
            }
        }

        let (pool, scorer) = self.run_shared_candidate_stage(
            &encoding.embeddings.text_emb,
            &encoding.queries.query_emb,
            marginal,
        )?;
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
        let mut thresholds = vec![default_threshold; expected_queries.len()];
        for (structure, query_ids) in schema.structures.iter().zip(&structure_query_ids) {
            for (field, &query_id) in structure.fields.iter().zip(query_ids) {
                thresholds[query_id] = field.threshold.unwrap_or(default_threshold);
            }
        }
        for (entity, &query_id) in schema.entities.iter().zip(&entity_query_ids) {
            thresholds[query_id] = entity.threshold.unwrap_or(default_threshold);
        }
        let grouped_inputs: Vec<_> = thresholds
            .into_iter()
            .enumerate()
            .map(|(query_id, threshold)| QueryScores {
                indices: indices.clone(),
                valid_mask: valid_mask.clone(),
                pair_logits: scorer
                    .pair_logits
                    .slice(s![0, query_id, ..])
                    .iter()
                    .copied()
                    .collect(),
                query_valid: true,
                threshold,
                null_logit: Some(scorer.null_logits[(0, query_id)]),
            })
            .collect();
        let grouped = group_scored_candidates(
            &grouped_inputs,
            self.runtime.pair_temperature,
            self.runtime.abstention_threshold,
        )?;
        let query_mask = Array1::from_elem(encoding.queries.query_emb.nrows(), true);
        let mut decoded_record_groups = BTreeMap::new();
        for spec in record_specs {
            let decoded = match prepare_record_group(
                &spec,
                encoding.queries.query_emb.view(),
                query_mask.view(),
                &pool,
                &scorer,
            )? {
                Some(prepared_group) => {
                    let record_output = self.records.infer(&prepared_group.input)?;
                    let group = prepared_group.into_decode_group(record_output)?;
                    decode_group(
                        &group,
                        default_threshold,
                        default_threshold,
                        default_threshold,
                        self.runtime.record_temperature,
                    )?
                }
                None => Vec::new(),
            };
            decoded_record_groups.insert(spec.structure_index, (spec, decoded));
        }

        output.relations = self.run_relation_stage(
            &schema.relations,
            &relation_query_ids,
            default_threshold,
            &encoding.prepared,
            &encoding.embeddings.text_emb,
            &encoding.queries.query_emb,
            &pool,
            &scorer,
        )?;

        let prepared = encoding.prepared;
        let offsets: Vec<_> = prepared
            .original_offsets
            .iter()
            .map(|offset| WordOffset {
                start: offset.start,
                end: offset.end,
            })
            .collect();

        for (structure_index, (structure, query_ids)) in schema
            .structures
            .iter()
            .zip(&structure_query_ids)
            .enumerate()
        {
            if record_metadata.contains_key(&structure.name) {
                let instances = match decoded_record_groups.get(&structure_index) {
                    Some((spec, decoded)) => format_record_instances(
                        structure,
                        spec,
                        decoded,
                        &prepared,
                        &offsets,
                        &pool,
                        &scorer,
                        &choice_scores,
                        default_threshold,
                        self.runtime.pair_temperature,
                        self.runtime.overlap_policy,
                    )?,
                    None => Vec::new(),
                };
                if !instances.is_empty() {
                    output.structures.push(RawStructureOutput {
                        name: structure.name.clone(),
                        instances,
                    });
                }
                continue;
            }

            let mut fields = Vec::with_capacity(structure.fields.len());
            let mut any_value = false;
            for (field, &query_id) in structure.fields.iter().zip(query_ids) {
                let value = if field.choices.is_empty() {
                    let decoded = decode_to_utf8(
                        &prepared.original_text,
                        &offsets,
                        &grouped[query_id],
                        self.runtime.overlap_policy,
                        prepared.choice_prefix_words,
                    )?;
                    let spans: Vec<_> = decoded
                        .into_iter()
                        .filter(|span| {
                            field
                                .validators
                                .iter()
                                .all(|validator| validator.validate(&span.text))
                        })
                        .map(|span| EntitySpan {
                            start: span.start,
                            end: span.end,
                            text: span.text,
                            score: span.confidence,
                        })
                        .collect();
                    any_value |= !spans.is_empty();
                    RawStructureValue::Spans(spans)
                } else {
                    let scores = choice_scores
                        .get(&query_id)
                        .context("missing explicit choice scores for routed structure field")?;
                    let selected = select_choice_values(
                        scores.clone(),
                        field.dtype.clone(),
                        field.threshold.unwrap_or(default_threshold),
                    );
                    any_value |= !selected.is_empty();
                    RawStructureValue::Choices(selected)
                };
                fields.push(RawStructureField {
                    name: field.name.clone(),
                    dtype: field.dtype.clone(),
                    value,
                });
            }
            if any_value {
                output.structures.push(RawStructureOutput {
                    name: structure.name.clone(),
                    instances: vec![fields],
                });
            }
        }

        for (entity, &query_id) in schema.entities.iter().zip(&entity_query_ids) {
            let decoded = decode_to_utf8(
                &prepared.original_text,
                &offsets,
                &grouped[query_id],
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

    #[allow(clippy::too_many_arguments)]
    fn run_relation_stage(
        &self,
        relation_specs: &[crate::schema_spec::RelationSpec],
        relation_query_ids: &[[usize; 2]],
        default_threshold: f32,
        prepared: &PreparedTokens,
        text_embeddings: &Array2<f32>,
        query_embeddings: &Array2<f32>,
        pool: &CandidatePool,
        scorer: &ScorerOutput,
    ) -> Result<Vec<RawRelationOutput>> {
        ensure!(
            relation_specs.len() == relation_query_ids.len(),
            "relation schema/routing count mismatch: {} specs, {} routes",
            relation_specs.len(),
            relation_query_ids.len()
        );
        if relation_specs.is_empty() {
            return Ok(Vec::new());
        }

        let query_count = query_embeddings.nrows();
        let candidate_count = pool.indices.nrows();
        ensure!(
            scorer.pair_logits.dim() == (1, query_count, candidate_count),
            "relation proposal scorer shape {:?}, expected [1,{query_count},{candidate_count}]",
            scorer.pair_logits.dim()
        );
        let candidate_indices = pool.indices.view().insert_axis(Axis(0));
        let indices = candidate_indices
            .broadcast((query_count, candidate_count, 2))
            .context("failed to broadcast shared relation candidate indices")?;
        let candidate_mask = pool.mask.view().insert_axis(Axis(0));
        let valid_mask = candidate_mask
            .broadcast((query_count, candidate_count))
            .context("failed to broadcast shared relation candidate mask")?;
        let query_mask = Array1::from_elem(query_count, true);
        let typed_specs: Vec<_> = relation_specs
            .iter()
            .zip(relation_query_ids)
            .map(|(spec, &[head, tail])| {
                Ok(RelationTypeSpec {
                    relation_type: spec.name.clone(),
                    head_query_ids: vec![i64::try_from(head)?],
                    tail_query_ids: vec![i64::try_from(tail)?],
                    allow_self: false,
                })
            })
            .collect::<Result<_>>()?;
        let pairs = generate_relation_pairs(
            indices,
            valid_mask,
            query_mask.view(),
            scorer.pair_logits.index_axis(Axis(0), 0),
            &typed_specs,
            self.runtime.relation_proposals,
        )?;
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let hidden = query_embeddings.ncols();
        ensure!(
            text_embeddings.ncols() == hidden,
            "relation text/query hidden width mismatch: {} and {hidden}",
            text_embeddings.ncols()
        );
        let relation_width = hidden
            .checked_mul(2)
            .context("relation query width overflow")?;
        let mut relation_states = Array3::zeros((1, relation_specs.len(), relation_width));
        for (relation_index, &[head, tail]) in relation_query_ids.iter().enumerate() {
            ensure!(
                head < query_count && tail < query_count,
                "relation {relation_index} query route [{head},{tail}] exceeds {query_count} queries"
            );
            relation_states
                .slice_mut(s![0, relation_index, ..hidden])
                .assign(&query_embeddings.row(head));
            relation_states
                .slice_mut(s![0, relation_index, hidden..])
                .assign(&query_embeddings.row(tail));
        }

        let pair_array = |select: fn(&RelationPair) -> usize| -> Result<Array1<i64>> {
            pairs
                .iter()
                .map(|pair| i64::try_from(select(pair)).map_err(Into::into))
                .collect::<Result<Vec<_>>>()
                .map(Array1::from_vec)
        };
        let relation_output = self.relations.infer(&RelationInput {
            text_states: text_embeddings.view().insert_axis(Axis(0)).to_owned(),
            relation_query_states: relation_states,
            batch_index: Array1::zeros(pairs.len()),
            relation_index: pair_array(|pair| pair.relation_index)?,
            head_start: pair_array(|pair| pair.head[0])?,
            head_end: pair_array(|pair| pair.head[1])?,
            tail_start: pair_array(|pair| pair.tail[0])?,
            tail_end: pair_array(|pair| pair.tail[1])?,
            pair_mask: Array1::from_elem(pairs.len(), true),
        })?;

        let offsets: Vec<_> = prepared
            .original_offsets
            .iter()
            .map(|offset| WordOffset {
                start: offset.start,
                end: offset.end,
            })
            .collect();
        let mut grouped: Vec<RawRelationOutput> = Vec::new();
        for (pair, &logit) in pairs.iter().zip(relation_output.relation_logits.iter()) {
            let relation = relation_specs
                .get(pair.relation_index)
                .context("relation pair references an unknown schema entry")?;
            let threshold = relation.threshold.unwrap_or(default_threshold);
            let score = sigmoid_probability(logit, self.runtime.relation_temperature)?;
            if score < threshold {
                continue;
            }
            let Some(head) = map_relation_mention(prepared, &offsets, pair.head)? else {
                continue;
            };
            let Some(tail) = map_relation_mention(prepared, &offsets, pair.tail)? else {
                continue;
            };
            let edge = RelationEdge { head, tail, score };
            if let Some(existing) = grouped.iter_mut().find(|entry| entry.name == relation.name) {
                existing.edges.push(edge);
            } else {
                grouped.push(RawRelationOutput {
                    name: relation.name.clone(),
                    edges: vec![edge],
                });
            }
        }
        for relation in &mut grouped {
            relation.edges = deduplicate_relation_edges(&prepared.original_text, &relation.edges)?;
        }
        Ok(grouped)
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
            .extract_internal(
                text,
                &builder.build(),
                &RecordMetadata::new(),
                threshold,
                include_confidence,
                false,
            )?
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
        let mut outputs = self
            .run_schema(text, &schema, &RecordMetadata::new(), 0.5)?
            .classifications;
        ensure!(
            outputs.len() == 1,
            "single classification request produced {} task outputs",
            outputs.len()
        );
        Ok(outputs.remove(0).1)
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
        self.extract_json_with_records(
            text,
            schema,
            &RecordMetadata::new(),
            threshold,
            include_confidence,
            include_spans,
        )
    }

    /// Extract a JSON schema with opt-in record formation metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn extract_json_with_records(
        &self,
        text: &str,
        schema: &JsonSchema,
        metadata: &RecordMetadata,
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<JsonExtraction> {
        let structures = schema
            .structures
            .iter()
            .map(|structure| {
                let fields = structure
                    .fields
                    .iter()
                    .map(|field| crate::json::parse_field_spec(field))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StructureSpec {
                    name: structure.name.clone(),
                    fields,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let schema = SchemaSpec {
            structures,
            ..SchemaSpec::default()
        };
        Ok(self
            .extract_internal(
                text,
                &schema,
                metadata,
                threshold,
                include_confidence,
                include_spans,
            )?
            .structures)
    }

    fn extract_relations_with_specs(
        &self,
        text: &str,
        relations: &[crate::schema_spec::RelationSpec],
        threshold: f32,
        include_confidence: bool,
        include_spans: bool,
    ) -> Result<FormattedRelationExtraction> {
        let schema = SchemaSpec {
            relations: relations.to_vec(),
            ..SchemaSpec::default()
        };
        Ok(self
            .extract_internal(
                text,
                &schema,
                &RecordMetadata::new(),
                threshold,
                include_confidence,
                include_spans,
            )?
            .relations)
    }

    pub fn extract_relations(
        &self,
        text: &str,
        relation_types: &[String],
        threshold: f32,
    ) -> Result<RelationExtraction> {
        fn text_of(span: &crate::entities::FormattedEntitySpan) -> &str {
            match span {
                crate::entities::FormattedEntitySpan::Text(text)
                | crate::entities::FormattedEntitySpan::TextWithConfidence { text, .. }
                | crate::entities::FormattedEntitySpan::TextWithSpans { text, .. }
                | crate::entities::FormattedEntitySpan::TextWithConfidenceAndSpans {
                    text, ..
                } => text,
            }
        }

        let specs: Vec<_> = relation_types
            .iter()
            .cloned()
            .map(crate::schema_spec::RelationSpec::new)
            .collect();
        let formatted = self.extract_relations_with_specs(text, &specs, threshold, false, false)?;
        Ok(formatted
            .into_iter()
            .map(|(name, pairs)| {
                let pairs = pairs
                    .into_iter()
                    .map(|pair| {
                        (
                            text_of(&pair.head).to_owned(),
                            text_of(&pair.tail).to_owned(),
                        )
                    })
                    .collect();
                (name, pairs)
            })
            .collect())
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
        let specs = relation_types
            .iter()
            .cloned()
            .map(crate::schema_spec::RelationSpec::new)
            .collect::<Vec<_>>();
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
        ensure!(batch_size > 0, "batch_size must be > 0");
        let mut output = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch_size) {
            for text in chunk {
                output.push(self.extract_relations(text.as_ref(), relation_types, threshold)?);
            }
        }
        Ok(output)
    }
}

fn map_relation_mention(
    prepared: &PreparedTokens,
    offsets: &[WordOffset],
    [raw_start, raw_end]: [usize; 2],
) -> Result<Option<RelationMention>> {
    let Some(start) = raw_start.checked_sub(prepared.choice_prefix_words) else {
        return Ok(None);
    };
    let Some(end) = raw_end.checked_sub(prepared.choice_prefix_words) else {
        return Ok(None);
    };
    if start >= end || end > offsets.len() {
        return Ok(None);
    }
    let mapped = map_half_open_utf8(&prepared.original_text, offsets, start, end)?;
    if mapped.start >= mapped.end {
        return Ok(None);
    }
    let source = prepared
        .original_text
        .get(mapped.start..mapped.end)
        .context("mapped relation span is not a UTF-8 source slice")?;
    let surface = source.trim_matches(is_python_whitespace).to_owned();
    if surface.is_empty() {
        return Ok(None);
    }
    Ok(Some(RelationMention {
        text: surface,
        start: mapped.start,
        end: mapped.end,
    }))
}

fn is_python_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000D}'
            | '\u{001C}'..='\u{001F}'
            | '\u{0020}'
            | '\u{0085}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

#[allow(clippy::too_many_arguments)]
fn format_record_instances(
    structure: &StructureSpec,
    spec: &CompiledRecordSpec,
    records: &[DecodedRecord],
    prepared: &PreparedTokens,
    offsets: &[WordOffset],
    pool: &CandidatePool,
    scorer: &ScorerOutput,
    choice_scores: &BTreeMap<usize, Vec<(String, f32)>>,
    record_threshold: f32,
    pair_temperature: f32,
    overlap_policy: OverlapPolicy,
) -> Result<Vec<Vec<RawStructureField>>> {
    ensure!(
        structure.name == spec.name,
        "record structure name {:?} differs from compiled route {:?}",
        structure.name,
        spec.name
    );
    ensure!(
        structure.fields.len() == spec.fields.len(),
        "record structure {:?} has {} schema fields but {} compiled fields",
        structure.name,
        structure.fields.len(),
        spec.fields.len()
    );

    // Convert every decoded anchor exactly once. Invalid prefix, truncated, or
    // synthetic-only anchors remain `None` and cannot own source literals.
    let anchor_bytes = records
        .iter()
        .map(|record| {
            record
                .anchor_span
                .map(|span| map_record_word_span(prepared, offsets, span))
                .transpose()
                .map(Option::flatten)
                .map(|span| span.map(|span| [span.start, span.end]))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut instances = Vec::with_capacity(records.len());
    for (record_index, record) in records.iter().enumerate() {
        let mut fields = Vec::with_capacity(spec.fields.len());
        let mut any_value = false;
        for (field_index, (compiled, field)) in
            spec.fields.iter().zip(&structure.fields).enumerate()
        {
            ensure!(
                compiled.role_index == field_index && compiled.name == field.name,
                "record field layout mismatch at role {field_index}: compiled={:?}, schema={:?}",
                compiled.name,
                field.name
            );
            let raw_spans = record
                .fields
                .get(&compiled.query_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let assignment_scores = record
                .field_scores
                .get(&compiled.query_id)
                .map(Vec::as_slice);
            let formatted = format_record_field_spans(
                field,
                compiled.cardinality,
                compiled.query_id,
                raw_spans,
                assignment_scores,
                prepared,
                offsets,
                pool,
                scorer,
                record_threshold,
                pair_temperature,
                overlap_policy,
            )?;
            let is_scalar = cardinality_is_scalar(compiled.cardinality)
                || matches!(field.dtype, FieldDtype::Str);

            let value = if field.choices.is_empty() {
                RawStructureValue::Spans(formatted)
            } else {
                format_record_choice_field(
                    field,
                    compiled.query_id,
                    formatted,
                    is_scalar,
                    record_index,
                    &anchor_bytes,
                    &prepared.original_text,
                    choice_scores,
                    record_threshold,
                )?
            };
            any_value |= raw_value_has_content(&value);
            fields.push(RawStructureField {
                name: field.name.clone(),
                dtype: if is_scalar {
                    FieldDtype::Str
                } else {
                    FieldDtype::List
                },
                value,
            });
        }
        if any_value {
            instances.push(fields);
        }
    }
    Ok(instances)
}

#[allow(clippy::too_many_arguments)]
fn format_record_field_spans(
    field: &crate::schema_spec::StructureFieldSpec,
    cardinality: Cardinality,
    query_id: usize,
    raw_spans: &[[usize; 2]],
    assignment_scores: Option<&[f32]>,
    prepared: &PreparedTokens,
    offsets: &[WordOffset],
    pool: &CandidatePool,
    scorer: &ScorerOutput,
    record_threshold: f32,
    pair_temperature: f32,
    overlap_policy: OverlapPolicy,
) -> Result<Vec<EntitySpan>> {
    let mut filtered = Vec::new();
    for (span_index, &word_span) in raw_spans.iter().enumerate() {
        let Some(mapped) = map_record_word_span(prepared, offsets, word_span)? else {
            continue;
        };
        let candidate_probability =
            candidate_span_probability(pool, scorer, query_id, word_span, pair_temperature)?;
        let confidence = assignment_scores
            .and_then(|scores| scores.get(span_index))
            .map_or(candidate_probability, |assignment| {
                candidate_probability.min(*assignment)
            });
        if cardinality_allows_absent(cardinality) && candidate_probability < record_threshold {
            continue;
        }
        if field
            .threshold
            .is_some_and(|threshold| confidence < threshold)
        {
            continue;
        }
        if !field
            .validators
            .iter()
            .all(|validator| validator.validate(&mapped.text))
        {
            continue;
        }
        filtered.push(ScoredSpan {
            confidence,
            start: mapped.start,
            end: mapped.end,
        });
    }

    let resolved = resolve_overlaps(&filtered, overlap_policy)?;
    let is_scalar = cardinality_is_scalar(cardinality) || matches!(field.dtype, FieldDtype::Str);
    let limit = if is_scalar { 1 } else { usize::MAX };
    resolved
        .into_iter()
        .take(limit)
        .map(|span| {
            let surface = prepared
                .original_text
                .get(span.start..span.end)
                .context("resolved record span is not a UTF-8 source slice")?
                .trim()
                .to_owned();
            Ok(EntitySpan {
                start: span.start,
                end: span.end,
                text: surface,
                score: span.confidence,
            })
        })
        .collect()
}

fn map_record_word_span(
    prepared: &PreparedTokens,
    offsets: &[WordOffset],
    [raw_start, raw_end]: [usize; 2],
) -> Result<Option<super::decode::DecodedSpan>> {
    let Some(start) = raw_start.checked_sub(prepared.choice_prefix_words) else {
        return Ok(None);
    };
    let Some(end) = raw_end.checked_sub(prepared.choice_prefix_words) else {
        return Ok(None);
    };
    if start >= end || end > offsets.len() {
        return Ok(None);
    }
    // Match the existing boundary byte-offset contract: a span extending into
    // the appended punctuation is clamped to its original-text portion. Only
    // a wholly synthetic (empty mapped surface) span is discarded.
    let mapped = map_half_open_utf8(&prepared.original_text, offsets, start, end)?;
    Ok((!mapped.text.is_empty()).then_some(mapped))
}

fn candidate_span_probability(
    pool: &CandidatePool,
    scorer: &ScorerOutput,
    query_id: usize,
    [start, end]: [usize; 2],
    pair_temperature: f32,
) -> Result<f32> {
    ensure!(
        query_id < scorer.pair_logits.shape()[1],
        "record field query id {query_id} exceeds scorer query count {}",
        scorer.pair_logits.shape()[1]
    );
    let mut best = None;
    for candidate in 0..pool.mask.len() {
        if !pool.mask[candidate]
            || usize::try_from(pool.indices[(candidate, 0)]).ok() != Some(start)
            || usize::try_from(pool.indices[(candidate, 1)]).ok() != Some(end)
        {
            continue;
        }
        let probability = sigmoid_probability(
            scorer.pair_logits[(0, query_id, candidate)],
            pair_temperature,
        )?;
        best = Some(best.map_or(probability, |current: f32| current.max(probability)));
    }
    Ok(best.unwrap_or(0.0))
}

#[allow(clippy::too_many_arguments)]
fn format_record_choice_field(
    field: &crate::schema_spec::StructureFieldSpec,
    query_id: usize,
    formatted: Vec<EntitySpan>,
    is_scalar: bool,
    record_index: usize,
    anchors: &[Option<[usize; 2]>],
    text: &str,
    choice_scores: &BTreeMap<usize, Vec<(String, f32)>>,
    record_threshold: f32,
) -> Result<RawStructureValue> {
    let prefix_scores = choice_scores
        .get(&query_id)
        .context("missing explicit choice scores for routed record field")?;

    if let Some(anchor) = anchors.get(record_index).copied().flatten() {
        let (has_literals, local) = literal_choice_mentions(text, &field.choices, anchors)?;
        let owned = local.get(&record_index).map(Vec::as_slice).unwrap_or(&[]);
        let preferred = if is_scalar {
            nearest_choice(text, anchor, owned)?
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            owned.iter().collect()
        };
        if !preferred.is_empty() {
            let preferred_spans: Vec<_> = preferred
                .into_iter()
                .filter_map(|mention| {
                    prefix_scores
                        .iter()
                        .find(|(choice, _)| choice == &mention.choice)
                        .map(|(_, confidence)| EntitySpan {
                            start: mention.start,
                            end: mention.end,
                            text: mention.choice.clone(),
                            score: *confidence,
                        })
                })
                .collect();
            if !preferred_spans.is_empty() {
                return Ok(RawStructureValue::Spans(preferred_spans));
            }
            return Ok(prefix_choice_fallback(
                field,
                prefix_scores,
                record_threshold,
                is_scalar,
            ));
        }
        if has_literals {
            return Ok(RawStructureValue::Spans(Vec::new()));
        }
    }

    let mut matched = Vec::new();
    for span in formatted {
        if let Some(choice) = match_choice_surface(&span.text, &field.choices) {
            matched.push(EntitySpan {
                text: choice,
                ..span
            });
        }
    }
    if !matched.is_empty() {
        if is_scalar {
            matched.truncate(1);
        }
        return Ok(RawStructureValue::Spans(matched));
    }

    Ok(prefix_choice_fallback(
        field,
        prefix_scores,
        record_threshold,
        is_scalar,
    ))
}

fn prefix_choice_fallback(
    field: &crate::schema_spec::StructureFieldSpec,
    prefix_scores: &[(String, f32)],
    record_threshold: f32,
    is_scalar: bool,
) -> RawStructureValue {
    let dtype = if is_scalar {
        FieldDtype::Str
    } else {
        FieldDtype::List
    };
    RawStructureValue::Choices(select_choice_values(
        prefix_scores.to_vec(),
        dtype,
        field.threshold.unwrap_or(record_threshold),
    ))
}

fn cardinality_is_scalar(cardinality: Cardinality) -> bool {
    matches!(
        cardinality,
        Cardinality::OptionalOne | Cardinality::RequiredOne
    )
}

fn cardinality_allows_absent(cardinality: Cardinality) -> bool {
    matches!(
        cardinality,
        Cardinality::OptionalOne | Cardinality::ZeroOrMore
    )
}

fn raw_value_has_content(value: &RawStructureValue) -> bool {
    match value {
        RawStructureValue::Spans(spans) => !spans.is_empty(),
        RawStructureValue::Choices(choices) => !choices.is_empty(),
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

fn format_boundary_choices(
    choices: &[(String, f32)],
    dtype: FieldDtype,
    include_confidence: bool,
) -> FormattedEntityValue {
    let format = |(text, confidence): &(String, f32)| {
        if include_confidence {
            crate::entities::FormattedEntitySpan::TextWithConfidence {
                text: text.clone(),
                confidence: *confidence,
            }
        } else {
            crate::entities::FormattedEntitySpan::Text(text.clone())
        }
    };
    match dtype {
        FieldDtype::List => FormattedEntityValue::List(choices.iter().map(format).collect()),
        FieldDtype::Str => FormattedEntityValue::Single(choices.first().map(format)),
    }
}

fn select_choice_values(
    scores: Vec<(String, f32)>,
    dtype: FieldDtype,
    threshold: f32,
) -> Vec<(String, f32)> {
    match dtype {
        FieldDtype::List => scores
            .into_iter()
            .filter(|(_, probability)| *probability >= threshold)
            .collect(),
        FieldDtype::Str => {
            let mut best: Option<(String, f32)> = None;
            for candidate in scores {
                if best
                    .as_ref()
                    .is_none_or(|(_, probability)| candidate.1 > *probability)
                {
                    best = Some(candidate);
                }
            }
            best.filter(|(_, probability)| *probability >= threshold)
                .into_iter()
                .collect()
        }
    }
}

fn validate_schema_thresholds(schema: &SchemaSpec, default_threshold: f32) -> Result<()> {
    ensure!(
        default_threshold.is_finite() && (0.0..=1.0).contains(&default_threshold),
        "default threshold must be finite and in [0,1], got {default_threshold}"
    );
    for structure in &schema.structures {
        for field in &structure.fields {
            validate_optional_threshold("structure field", &field.name, field.threshold)?;
        }
    }
    for entity in &schema.entities {
        validate_optional_threshold("entity", &entity.name, entity.threshold)?;
    }
    for relation in &schema.relations {
        validate_optional_threshold("relation", &relation.name, relation.threshold)?;
    }
    for classification in &schema.classifications {
        ensure!(
            classification.cls_threshold.is_finite()
                && (0.0..=1.0).contains(&classification.cls_threshold),
            "classification threshold for {:?} must be finite and in [0,1], got {}",
            classification.task,
            classification.cls_threshold
        );
    }
    Ok(())
}

fn validate_optional_threshold(kind: &str, name: &str, threshold: Option<f32>) -> Result<()> {
    if let Some(threshold) = threshold {
        ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "{kind} threshold for {name:?} must be finite and in [0,1], got {threshold}"
        );
    }
    Ok(())
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

#[cfg(test)]
mod record_format_tests {
    use std::collections::BTreeMap;

    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3};

    use super::*;
    use crate::{
        boundary::{
            pool::CandidatePool,
            preprocessing::BoundaryPreprocessingPolicy,
            record_decode::RecordMode,
            record_schema::{CompiledRecordField, CompiledRecordSpec},
        },
        schema_spec::StructureFieldSpec,
        validators::RegexValidator,
    };

    fn pool_and_scores(
        spans: &[[i64; 2]],
        query_logits: &[Vec<f32>],
    ) -> (CandidatePool, ScorerOutput) {
        let candidates = spans.len();
        let queries = query_logits.len();
        let pool = CandidatePool {
            indices: Array2::from_shape_fn((candidates, 2), |(row, endpoint)| spans[row][endpoint]),
            mask: Array1::from_elem(candidates, true),
            compat_logits: Array1::zeros(candidates),
            proposal_logits: Array1::zeros(candidates),
        };
        let pair_logits =
            Array3::from_shape_fn((1, queries, candidates), |(_, query, candidate)| {
                query_logits[query][candidate]
            });
        let scores = ScorerOutput {
            pair_logits,
            candidate_states: Array3::zeros((1, candidates, 1)),
            null_logits: Array2::zeros((1, queries)),
            count_log_rates: Array2::zeros((1, queries)),
        };
        (pool, scores)
    }

    fn offsets(prepared: &PreparedTokens) -> Vec<WordOffset> {
        prepared
            .original_offsets
            .iter()
            .map(|offset| WordOffset {
                start: offset.start,
                end: offset.end,
            })
            .collect()
    }

    #[test]
    fn record_mapping_clamps_synthetic_suffix_like_existing_boundary_decode() -> Result<()> {
        let prepared = BoundaryPreprocessingPolicy::default().prepare("Alice", &[]);
        let offsets = offsets(&prepared);
        let end = offsets.len();
        let record = map_record_word_span(&prepared, &offsets, [0, end])?.unwrap();
        let ordinary = decode_to_utf8(
            &prepared.original_text,
            &offsets,
            &[ScoredSpan {
                confidence: 0.8,
                start: 0,
                end,
            }],
            OverlapPolicy::Disallow,
            0,
        )?;
        assert_eq!(
            (record.start, record.end, record.text.as_str()),
            (0, 5, "Alice")
        );
        assert_eq!(
            (record.start, record.end, &record.text),
            (ordinary[0].start, ordinary[0].end, &ordinary[0].text)
        );
        let synthetic = offsets.iter().position(|o| o.start == o.end).unwrap();
        assert!(map_record_word_span(&prepared, &offsets, [synthetic, synthetic + 1])?.is_none());
        Ok(())
    }

    #[test]
    fn required_bypasses_candidate_gate_but_configured_threshold_uses_combined_score() -> Result<()>
    {
        let prepared = BoundaryPreprocessingPolicy::default().prepare("alpha.", &[]);
        let offsets = offsets(&prepared);
        let (pool, scorer) = pool_and_scores(&[[0, 1]], &[vec![-2.0]]);
        let base = StructureFieldSpec::new("value").dtype(FieldDtype::Str);

        let optional = format_record_field_spans(
            &base,
            Cardinality::OptionalOne,
            0,
            &[[0, 1]],
            Some(&[0.8]),
            &prepared,
            &offsets,
            &pool,
            &scorer,
            0.5,
            1.0,
            OverlapPolicy::Disallow,
        )?;
        assert!(optional.is_empty());

        let required = format_record_field_spans(
            &base,
            Cardinality::RequiredOne,
            0,
            &[[0, 1]],
            Some(&[0.8]),
            &prepared,
            &offsets,
            &pool,
            &scorer,
            0.5,
            1.0,
            OverlapPolicy::Disallow,
        )?;
        assert_eq!(required.len(), 1);
        assert!((required[0].score - sigmoid_probability(-2.0, 1.0)?).abs() < 1e-6);

        let configured = StructureFieldSpec::new("value")
            .dtype(FieldDtype::Str)
            .threshold(0.2);
        assert!(
            format_record_field_spans(
                &configured,
                Cardinality::RequiredOne,
                0,
                &[[0, 1]],
                Some(&[0.8]),
                &prepared,
                &offsets,
                &pool,
                &scorer,
                0.5,
                1.0,
                OverlapPolicy::Disallow,
            )?
            .is_empty()
        );
        Ok(())
    }

    #[test]
    fn validators_run_before_byte_overlap_resolution() -> Result<()> {
        let prepared = BoundaryPreprocessingPolicy::default().prepare("alpha beta.", &[]);
        let offsets = offsets(&prepared);
        let (pool, scorer) = pool_and_scores(&[[0, 2], [1, 2]], &[vec![3.0, 2.0]]);
        let field = StructureFieldSpec::new("value")
            .dtype(FieldDtype::List)
            .validators(vec![RegexValidator::new("^beta$")]);
        let spans = format_record_field_spans(
            &field,
            Cardinality::OneOrMore,
            0,
            &[[0, 2], [1, 2]],
            None,
            &prepared,
            &offsets,
            &pool,
            &scorer,
            0.99,
            1.0,
            OverlapPolicy::Disallow,
        )?;
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "beta");
        Ok(())
    }

    #[test]
    fn literal_ownership_bypasses_threshold_and_unowned_record_does_not_fallback() -> Result<()> {
        let field = StructureFieldSpec::new("category")
            .dtype(FieldDtype::Str)
            .choices(vec!["books".into(), "hardware".into()])
            .threshold(1.0);
        let scores = BTreeMap::from([(0, vec![("books".into(), 0.01), ("hardware".into(), 0.99)])]);
        let value = format_record_choice_field(
            &field,
            0,
            Vec::new(),
            true,
            0,
            &[Some([0, 4])],
            "Acme sells books.",
            &scores,
            1.0,
        )?;
        let RawStructureValue::Spans(spans) = value else {
            panic!("literal source must retain spans")
        };
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "books");
        assert_eq!([spans[0].start, spans[0].end], [11, 16]);
        assert_eq!(spans[0].score, 0.01);

        let value = format_record_choice_field(
            &field,
            0,
            Vec::new(),
            true,
            1,
            &[Some([0, 1]), Some([8, 9])],
            "A books B",
            &scores,
            0.5,
        )?;
        assert!(matches!(value, RawStructureValue::Spans(spans) if spans.is_empty()));
        Ok(())
    }

    #[test]
    fn every_declared_field_is_emitted_and_empty_fields_do_not_keep_empty_records() -> Result<()> {
        let prepared = BoundaryPreprocessingPolicy::default().prepare("alpha.", &[]);
        let offsets = offsets(&prepared);
        let (pool, scorer) = pool_and_scores(&[[0, 1]], &[vec![2.0], vec![-2.0]]);
        let structure = StructureSpec {
            name: "record".into(),
            fields: vec![
                StructureFieldSpec::new("present").dtype(FieldDtype::Str),
                StructureFieldSpec::new("empty").dtype(FieldDtype::List),
            ],
        };
        let spec = CompiledRecordSpec {
            structure_index: 0,
            name: "record".into(),
            mode: RecordMode::Latent,
            fields: vec![
                CompiledRecordField {
                    name: "present".into(),
                    role_index: 0,
                    query_id: 0,
                    cardinality: Cardinality::RequiredOne,
                    exclusive: false,
                },
                CompiledRecordField {
                    name: "empty".into(),
                    role_index: 1,
                    query_id: 1,
                    cardinality: Cardinality::ZeroOrMore,
                    exclusive: false,
                },
            ],
            anchor_query_id: None,
        };
        let record = DecodedRecord {
            fields: BTreeMap::from([(0, vec![[0, 1]])]),
            field_scores: BTreeMap::from([(0, vec![0.9])]),
            anchor_span: None,
            score: 0.9,
        };
        let instances = format_record_instances(
            &structure,
            &spec,
            &[record],
            &prepared,
            &offsets,
            &pool,
            &scorer,
            &BTreeMap::new(),
            0.5,
            1.0,
            OverlapPolicy::Disallow,
        )?;
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].len(), 2);
        assert!(
            matches!(instances[0][1].value, RawStructureValue::Spans(ref spans) if spans.is_empty())
        );

        let empty_record = DecodedRecord {
            fields: BTreeMap::new(),
            field_scores: BTreeMap::new(),
            anchor_span: None,
            score: 0.9,
        };
        assert!(
            format_record_instances(
                &structure,
                &spec,
                &[empty_record],
                &prepared,
                &offsets,
                &pool,
                &scorer,
                &BTreeMap::new(),
                0.5,
                1.0,
                OverlapPolicy::Disallow,
            )?
            .is_empty()
        );
        Ok(())
    }
}
