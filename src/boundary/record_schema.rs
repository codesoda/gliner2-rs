//! Typed, opt-in record metadata and query-layout compilation.
//!
//! An absent [`RecordMetadata`] entry deliberately leaves that structure on the
//! legacy JSON path. This module never infers a record mode from field shape or
//! dtype. The validation performed for annotated structures is stricter than
//! legacy schema handling: keyed metadata must resolve unambiguously, field
//! names must be unique, and metadata typos are errors rather than ignored
//! configuration.
//!
//! This metadata controls inference-time record formation only. In particular,
//! it does not expose upstream training annotation options such as occurrence
//! policies.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail, ensure};

use super::record_decode::{Cardinality, RecordMode};
use crate::{
    embeddings::{QueryKind, QueryMetadata},
    schema_spec::{FieldDtype, StructureSpec},
};

/// Record configuration keyed by structure name.
///
/// Structures without an entry retain legacy JSON decoding.
pub type RecordMetadata = BTreeMap<String, RecordConfig>;

/// Explicit record-formation configuration for one structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordConfig {
    pub mode: RecordMode,
    pub anchor: Option<String>,
    pub fields: BTreeMap<String, RecordFieldOptions>,
}

impl RecordConfig {
    /// Form one record for each detected occurrence of `anchor`.
    pub fn natural(anchor: impl Into<String>) -> Self {
        Self {
            mode: RecordMode::Natural,
            anchor: Some(anchor.into()),
            fields: BTreeMap::new(),
        }
    }

    /// Let detected field mentions seed latent record instances.
    pub fn latent() -> Self {
        Self {
            mode: RecordMode::Latent,
            anchor: None,
            fields: BTreeMap::new(),
        }
    }

    /// Use learned document-conditioned record instances without an anchor.
    pub fn anchorless() -> Self {
        Self {
            mode: RecordMode::Anchorless,
            anchor: None,
            fields: BTreeMap::new(),
        }
    }

    /// Set inference options for a declared field.
    pub fn field(mut self, name: impl Into<String>, options: RecordFieldOptions) -> Self {
        self.fields.insert(name.into(), options);
        self
    }
}

/// Optional overrides for one record field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecordFieldOptions {
    /// `None` applies the upstream dtype/anchor default during compilation.
    pub cardinality: Option<Cardinality>,
    pub exclusive: bool,
}

impl RecordFieldOptions {
    pub fn cardinality(mut self, cardinality: Cardinality) -> Self {
        self.cardinality = Some(cardinality);
        self
    }

    pub fn exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }
}

/// One schema-ordered field compiled against the concrete global query layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledRecordField {
    pub(crate) name: String,
    pub(crate) role_index: usize,
    pub(crate) query_id: usize,
    pub(crate) cardinality: Cardinality,
    pub(crate) exclusive: bool,
}

/// An annotated non-empty structure compiled for the record runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompiledRecordSpec {
    pub(crate) structure_index: usize,
    pub(crate) name: String,
    pub(crate) mode: RecordMode,
    pub(crate) fields: Vec<CompiledRecordField>,
    pub(crate) anchor_query_id: Option<usize>,
}

/// Validate typed record metadata before encoder work begins.
///
/// Only explicitly annotated structures are subject to the additional strict
/// name checks. This preserves mixed schemas containing unrelated legacy
/// structures while ensuring a keyed record configuration is never applied
/// ambiguously or partially because of a typo.
pub fn validate_record_metadata(
    structures: &[StructureSpec],
    metadata: &RecordMetadata,
) -> Result<()> {
    if metadata.is_empty() {
        return Ok(());
    }

    for (structure_name, config) in metadata {
        let matching_structures: Vec<_> = structures
            .iter()
            .enumerate()
            .filter(|(_, structure)| structure.name == *structure_name)
            .collect();
        match matching_structures.as_slice() {
            [] => bail!("record metadata references unknown structure {structure_name:?}"),
            [(_, structure)] => {
                validate_structure_config(structure, config)?;
            }
            matches => bail!(
                "record metadata for structure {structure_name:?} is ambiguous: schema declares that structure name {} times",
                matches.len()
            ),
        }
    }

    Ok(())
}

fn validate_structure_config(structure: &StructureSpec, config: &RecordConfig) -> Result<()> {
    let mut field_names = BTreeSet::new();
    for field in &structure.fields {
        if !field_names.insert(field.name.as_str()) {
            bail!(
                "record metadata for structure {:?} is ambiguous: field {:?} is declared more than once",
                structure.name,
                field.name
            );
        }
    }

    match config.mode {
        RecordMode::Natural => {
            let anchor = config.anchor.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "record structure {:?} mode Natural requires a non-empty anchor",
                    structure.name
                )
            })?;
            ensure!(
                !anchor.is_empty(),
                "record structure {:?} mode Natural requires a non-empty anchor",
                structure.name
            );
            ensure!(
                field_names.contains(anchor),
                "record structure {:?} declares unknown anchor field {:?}",
                structure.name,
                anchor
            );
        }
        RecordMode::Latent | RecordMode::Anchorless => ensure!(
            config.anchor.is_none(),
            "record structure {:?} mode {:?} must not declare an anchor",
            structure.name,
            config.mode
        ),
    }

    for field_name in config.fields.keys() {
        ensure!(
            field_names.contains(field_name.as_str()),
            "record metadata for structure {:?} references unknown field {:?}",
            structure.name,
            field_name
        );
    }

    Ok(())
}

/// Compile explicitly annotated structures against the concrete global query
/// layout. Global query ids are positions in `queries`, not inferred offsets.
///
/// `structure_schema_indices[structure_index]` supplies the corresponding
/// schema index used by [`QueryMetadata`]. Annotated latent and anchorless
/// structures with no fields produce no group, matching the upstream no-query
/// behavior. Natural structures with no fields fail validation because their
/// anchor cannot resolve.
pub(crate) fn compile_record_specs(
    structures: &[StructureSpec],
    metadata: &RecordMetadata,
    structure_schema_indices: &[usize],
    queries: &[QueryMetadata],
) -> Result<Vec<CompiledRecordSpec>> {
    validate_record_metadata(structures, metadata)?;
    if metadata.is_empty() {
        return Ok(Vec::new());
    }

    ensure!(
        structure_schema_indices.len() == structures.len(),
        "structure schema-index mapping has {} entries for {} structures",
        structure_schema_indices.len(),
        structures.len()
    );

    let mut annotated_schema_indices = BTreeMap::new();
    for (structure_index, structure) in structures.iter().enumerate() {
        if !metadata.contains_key(&structure.name) {
            continue;
        }
        let schema_index = structure_schema_indices[structure_index];
        if let Some(previous_structure) =
            annotated_schema_indices.insert(schema_index, structure_index)
        {
            bail!(
                "annotated structures {:?} and {:?} both map to schema index {schema_index}",
                structures[previous_structure].name,
                structure.name
            );
        }
    }

    let mut specs = Vec::new();
    for (structure_index, structure) in structures.iter().enumerate() {
        let Some(config) = metadata.get(&structure.name) else {
            continue;
        };
        if structure.fields.is_empty() {
            continue;
        }

        let schema_index = structure_schema_indices[structure_index];
        let mut compiled_fields = Vec::with_capacity(structure.fields.len());
        let mut anchor_query_id = None;

        for (role_index, field) in structure.fields.iter().enumerate() {
            let matching_queries: Vec<_> = queries
                .iter()
                .enumerate()
                .filter(|(_, query)| {
                    query.schema_idx == schema_index && query.field_idx == role_index
                })
                .collect();
            let query_id = match matching_queries.as_slice() {
                [(query_id, query)] => {
                    ensure!(
                        query.kind == QueryKind::Content,
                        "record structure {:?} field {:?} (role {role_index}, schema index {schema_index}) requires a Content query, found {:?}",
                        structure.name,
                        field.name,
                        query.kind
                    );
                    *query_id
                }
                [] => bail!(
                    "record structure {:?} field {:?} (role {role_index}, schema index {schema_index}) requires exactly one Content query, found none",
                    structure.name,
                    field.name
                ),
                matches => bail!(
                    "record structure {:?} field {:?} (role {role_index}, schema index {schema_index}) requires exactly one Content query, found {} routes",
                    structure.name,
                    field.name,
                    matches.len()
                ),
            };

            let is_anchor = config.mode == RecordMode::Natural
                && config.anchor.as_deref() == Some(field.name.as_str());
            let options = config.fields.get(&field.name);
            let cardinality = options
                .and_then(|options| options.cardinality)
                .unwrap_or_else(|| default_cardinality(&field.dtype, is_anchor));
            let exclusive = options.is_some_and(|options| options.exclusive);

            if is_anchor {
                anchor_query_id = Some(query_id);
            }
            compiled_fields.push(CompiledRecordField {
                name: field.name.clone(),
                role_index,
                query_id,
                cardinality,
                exclusive,
            });
        }

        specs.push(CompiledRecordSpec {
            structure_index,
            name: structure.name.clone(),
            mode: config.mode,
            fields: compiled_fields,
            anchor_query_id,
        });
    }

    Ok(specs)
}

fn default_cardinality(dtype: &FieldDtype, is_anchor: bool) -> Cardinality {
    if is_anchor {
        Cardinality::RequiredOne
    } else {
        match dtype {
            FieldDtype::Str => Cardinality::OptionalOne,
            FieldDtype::List => Cardinality::ZeroOrMore,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_spec::StructureFieldSpec;

    fn field(name: &str, dtype: FieldDtype) -> StructureFieldSpec {
        StructureFieldSpec::new(name).dtype(dtype)
    }

    fn structure(name: &str, fields: Vec<StructureFieldSpec>) -> StructureSpec {
        StructureSpec {
            name: name.into(),
            fields,
        }
    }

    fn query(schema_idx: usize, field_idx: usize, kind: QueryKind) -> QueryMetadata {
        QueryMetadata {
            schema_idx,
            field_idx,
            kind,
        }
    }

    #[test]
    fn constructors_and_field_options_are_typed_and_fluent() {
        assert_eq!(
            RecordConfig::natural("name"),
            RecordConfig {
                mode: RecordMode::Natural,
                anchor: Some("name".into()),
                fields: BTreeMap::new(),
            }
        );
        assert_eq!(RecordConfig::latent().mode, RecordMode::Latent);
        assert_eq!(RecordConfig::anchorless().mode, RecordMode::Anchorless);
        assert_eq!(RecordFieldOptions::default().cardinality, None);
        assert!(!RecordFieldOptions::default().exclusive);

        let config = RecordConfig::latent().field(
            "items",
            RecordFieldOptions::default()
                .cardinality(Cardinality::OneOrMore)
                .exclusive(true),
        );
        assert_eq!(
            config.fields["items"],
            RecordFieldOptions {
                cardinality: Some(Cardinality::OneOrMore),
                exclusive: true,
            }
        );
    }

    #[test]
    fn compiles_all_modes_and_upstream_cardinality_defaults() -> Result<()> {
        let structures = vec![
            structure(
                "person",
                vec![
                    field("name", FieldDtype::Str),
                    field("title", FieldDtype::Str),
                    field("tags", FieldDtype::List),
                ],
            ),
            structure("product", vec![field("name", FieldDtype::Str)]),
            structure("meeting", vec![field("topics", FieldDtype::List)]),
        ];
        let metadata = BTreeMap::from([
            ("person".into(), RecordConfig::natural("name")),
            ("product".into(), RecordConfig::latent()),
            ("meeting".into(), RecordConfig::anchorless()),
        ]);
        let queries = vec![
            query(4, 0, QueryKind::Content),
            query(4, 1, QueryKind::Content),
            query(4, 2, QueryKind::Content),
            query(7, 0, QueryKind::Content),
            query(9, 0, QueryKind::Content),
        ];

        let specs = compile_record_specs(&structures, &metadata, &[4, 7, 9], &queries)?;
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].mode, RecordMode::Natural);
        assert_eq!(specs[0].anchor_query_id, Some(0));
        assert_eq!(
            specs[0]
                .fields
                .iter()
                .map(|field| field.cardinality)
                .collect::<Vec<_>>(),
            vec![
                Cardinality::RequiredOne,
                Cardinality::OptionalOne,
                Cardinality::ZeroOrMore,
            ]
        );
        assert_eq!(specs[1].mode, RecordMode::Latent);
        assert_eq!(specs[1].anchor_query_id, None);
        assert_eq!(specs[2].mode, RecordMode::Anchorless);
        assert_eq!(specs[2].anchor_query_id, None);
        Ok(())
    }

    #[test]
    fn explicit_anchor_override_required_list_and_exclusive_win() -> Result<()> {
        let structures = vec![structure(
            "event",
            vec![
                field("anchor", FieldDtype::Str),
                field("required", FieldDtype::List),
                field("scalar", FieldDtype::Str),
            ],
        )];
        let metadata = BTreeMap::from([(
            "event".into(),
            RecordConfig::natural("anchor")
                .field(
                    "anchor",
                    RecordFieldOptions::default().cardinality(Cardinality::OptionalOne),
                )
                .field(
                    "required",
                    RecordFieldOptions::default()
                        .cardinality(Cardinality::OneOrMore)
                        .exclusive(true),
                )
                .field(
                    "scalar",
                    RecordFieldOptions::default().cardinality(Cardinality::RequiredOne),
                ),
        )]);
        let queries = (0..3)
            .map(|field_idx| query(12, field_idx, QueryKind::Content))
            .collect::<Vec<_>>();

        let spec = &compile_record_specs(&structures, &metadata, &[12], &queries)?[0];
        assert_eq!(spec.fields[0].cardinality, Cardinality::OptionalOne);
        assert_eq!(spec.fields[1].cardinality, Cardinality::OneOrMore);
        assert!(spec.fields[1].exclusive);
        assert_eq!(spec.fields[2].cardinality, Cardinality::RequiredOne);
        assert!(spec.fields.iter().enumerate().all(|(index, field)| {
            field.role_index == index && field.name == structures[0].fields[index].name
        }));
        Ok(())
    }

    #[test]
    fn mixed_schema_indices_use_global_noncontiguous_query_ids() -> Result<()> {
        let structures = vec![
            structure("legacy", vec![field("ignored", FieldDtype::List)]),
            structure(
                "annotated",
                vec![
                    field("first", FieldDtype::Str),
                    field("second", FieldDtype::List),
                ],
            ),
            structure("also_legacy", vec![field("ignored", FieldDtype::Str)]),
        ];
        let metadata = BTreeMap::from([("annotated".into(), RecordConfig::latent())]);
        let queries = vec![
            query(1, 0, QueryKind::Entity),
            query(41, 0, QueryKind::Content),
            query(90, 0, QueryKind::Relation),
            query(2, 0, QueryKind::Entity),
            query(41, 1, QueryKind::Content),
            query(77, 0, QueryKind::Content),
        ];

        let specs = compile_record_specs(&structures, &metadata, &[5, 41, 77], &queries)?;
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].structure_index, 1);
        assert_eq!(specs[0].name, "annotated");
        assert_eq!(
            specs[0]
                .fields
                .iter()
                .map(|field| field.query_id)
                .collect::<Vec<_>>(),
            vec![1, 4]
        );
        Ok(())
    }

    #[test]
    fn absent_metadata_compiles_nothing_without_inspecting_legacy_layout() -> Result<()> {
        let structures = vec![
            structure(
                "duplicate_fields_are_legacy",
                vec![field("x", FieldDtype::Str), field("x", FieldDtype::List)],
            ),
            structure("unrouted", vec![field("y", FieldDtype::Str)]),
        ];

        assert!(compile_record_specs(&structures, &RecordMetadata::new(), &[], &[])?.is_empty());
        Ok(())
    }

    #[test]
    fn latent_and_anchorless_empty_structures_compile_no_group() -> Result<()> {
        let structures = vec![structure("latent", vec![]), structure("anchorless", vec![])];
        let metadata = BTreeMap::from([
            ("latent".into(), RecordConfig::latent()),
            ("anchorless".into(), RecordConfig::anchorless()),
        ]);

        assert!(compile_record_specs(&structures, &metadata, &[3, 8], &[])?.is_empty());
        Ok(())
    }

    #[test]
    fn natural_anchor_validation_rejects_missing_empty_unknown_and_empty_structure() {
        let structures = vec![structure("person", vec![field("name", FieldDtype::Str)])];
        let invalid = [
            RecordConfig {
                mode: RecordMode::Natural,
                anchor: None,
                fields: BTreeMap::new(),
            },
            RecordConfig::natural(""),
            RecordConfig::natural("missing"),
        ];
        for config in invalid {
            let error =
                validate_record_metadata(&structures, &BTreeMap::from([("person".into(), config)]))
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("anchor"), "{error}");
        }

        let error = validate_record_metadata(
            &[structure("empty", vec![])],
            &BTreeMap::from([("empty".into(), RecordConfig::natural("anchor"))]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown anchor field"), "{error}");
    }

    #[test]
    fn non_natural_modes_forbid_even_empty_anchor_values() {
        for mode in [RecordMode::Latent, RecordMode::Anchorless] {
            let config = RecordConfig {
                mode,
                anchor: Some(String::new()),
                fields: BTreeMap::new(),
            };
            let error = validate_record_metadata(
                &[structure("record", vec![])],
                &BTreeMap::from([("record".into(), config)]),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("must not declare an anchor"), "{error}");
        }
    }

    #[test]
    fn rejects_unknown_metadata_structure_and_field_names() {
        let structures = vec![structure("known", vec![field("field", FieldDtype::Str)])];
        let unknown_structure = validate_record_metadata(
            &structures,
            &BTreeMap::from([("typo".into(), RecordConfig::latent())]),
        )
        .unwrap_err()
        .to_string();
        assert!(unknown_structure.contains("unknown structure"));

        let unknown_field = validate_record_metadata(
            &structures,
            &BTreeMap::from([(
                "known".into(),
                RecordConfig::latent().field("typo", RecordFieldOptions::default()),
            )]),
        )
        .unwrap_err()
        .to_string();
        assert!(unknown_field.contains("unknown field"));
        assert!(unknown_field.contains("known"));
    }

    #[test]
    fn rejects_ambiguous_annotated_structure_and_field_names() {
        let duplicate_structures = vec![
            structure("same", vec![field("a", FieldDtype::Str)]),
            structure("same", vec![field("b", FieldDtype::Str)]),
        ];
        let error = validate_record_metadata(
            &duplicate_structures,
            &BTreeMap::from([("same".into(), RecordConfig::latent())]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("ambiguous"));
        assert!(error.contains("2 times"));

        let duplicate_fields = vec![structure(
            "same",
            vec![
                field("field", FieldDtype::Str),
                field("field", FieldDtype::List),
            ],
        )];
        let error = validate_record_metadata(
            &duplicate_fields,
            &BTreeMap::from([("same".into(), RecordConfig::latent())]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("ambiguous"));
        assert!(error.contains("field"));
    }

    #[test]
    fn rejects_missing_wrong_kind_and_duplicate_content_routes() {
        let structures = vec![structure("record", vec![field("value", FieldDtype::Str)])];
        let metadata = BTreeMap::from([("record".into(), RecordConfig::latent())]);

        let missing = compile_record_specs(&structures, &metadata, &[6], &[])
            .unwrap_err()
            .to_string();
        assert!(missing.contains("found none"), "{missing}");

        let wrong_kind = compile_record_specs(
            &structures,
            &metadata,
            &[6],
            &[query(6, 0, QueryKind::Entity)],
        )
        .unwrap_err()
        .to_string();
        assert!(wrong_kind.contains("found Entity"), "{wrong_kind}");

        let duplicate = compile_record_specs(
            &structures,
            &metadata,
            &[6],
            &[
                query(6, 0, QueryKind::Content),
                query(6, 0, QueryKind::Content),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("found 2"), "{duplicate}");
    }

    #[test]
    fn matches_four_frozen_annotated_record_schema_shapes() -> Result<()> {
        let cases = [
            (
                structure(
                    "person",
                    vec![
                        field("name", FieldDtype::Str),
                        field("occupation", FieldDtype::Str),
                        field("location", FieldDtype::Str),
                    ],
                ),
                RecordConfig::natural("name")
                    .field(
                        "name",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::RequiredOne)
                            .exclusive(true),
                    )
                    .field(
                        "occupation",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::OptionalOne)
                            .exclusive(true),
                    )
                    .field(
                        "location",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::OptionalOne)
                            .exclusive(true),
                    ),
                vec![
                    Cardinality::RequiredOne,
                    Cardinality::OptionalOne,
                    Cardinality::OptionalOne,
                ],
            ),
            (
                structure(
                    "vendor",
                    vec![
                        field("company", FieldDtype::Str),
                        field("category", FieldDtype::Str),
                    ],
                ),
                RecordConfig::natural("company")
                    .field(
                        "company",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::RequiredOne)
                            .exclusive(true),
                    )
                    .field(
                        "category",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::OptionalOne)
                            .exclusive(true),
                    ),
                vec![Cardinality::RequiredOne, Cardinality::OptionalOne],
            ),
            (
                structure(
                    "product",
                    vec![
                        field("name", FieldDtype::Str),
                        field("color", FieldDtype::Str),
                        field("price", FieldDtype::Str),
                    ],
                ),
                RecordConfig::latent()
                    .field(
                        "name",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::RequiredOne)
                            .exclusive(true),
                    )
                    .field(
                        "color",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::OptionalOne)
                            .exclusive(true),
                    )
                    .field(
                        "price",
                        RecordFieldOptions::default()
                            .cardinality(Cardinality::OptionalOne)
                            .exclusive(true),
                    ),
                vec![
                    Cardinality::RequiredOne,
                    Cardinality::OptionalOne,
                    Cardinality::OptionalOne,
                ],
            ),
            (
                structure(
                    "meeting",
                    vec![
                        field("topic", FieldDtype::List),
                        field("decision", FieldDtype::List),
                    ],
                ),
                RecordConfig::anchorless()
                    .field(
                        "topic",
                        RecordFieldOptions::default().cardinality(Cardinality::OneOrMore),
                    )
                    .field(
                        "decision",
                        RecordFieldOptions::default().cardinality(Cardinality::ZeroOrMore),
                    ),
                vec![Cardinality::OneOrMore, Cardinality::ZeroOrMore],
            ),
        ];

        for (schema_index, (structure, config, cardinalities)) in cases.into_iter().enumerate() {
            let field_count = structure.fields.len();
            let name = structure.name.clone();
            let queries = (0..field_count)
                .map(|field_idx| query(schema_index + 20, field_idx, QueryKind::Content))
                .collect::<Vec<_>>();
            let specs = compile_record_specs(
                &[structure],
                &BTreeMap::from([(name, config)]),
                &[schema_index + 20],
                &queries,
            )?;
            assert_eq!(
                specs[0]
                    .fields
                    .iter()
                    .map(|field| field.cardinality)
                    .collect::<Vec<_>>(),
                cardinalities
            );
        }
        Ok(())
    }
}
