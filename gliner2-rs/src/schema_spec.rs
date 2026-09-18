use std::collections::BTreeMap;

use crate::classification::{ClassAct, FormattedClassification};
use crate::entities::FormattedEntityValue;
use crate::relations::FormattedRelationPair;
use crate::validators::RegexValidator;

#[derive(Debug, Clone, PartialEq)]
pub enum FieldDtype {
    Str,
    List,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructureFieldSpec {
    pub name: String,
    pub dtype: FieldDtype,
    pub description: Option<String>,
    pub choices: Vec<String>,
    pub threshold: Option<f32>,
    pub validators: Vec<RegexValidator>,
}

impl StructureFieldSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype: FieldDtype::List,
            description: None,
            choices: Vec::new(),
            threshold: None,
            validators: Vec::new(),
        }
    }

    pub fn dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn choices(mut self, choices: Vec<String>) -> Self {
        self.choices = choices;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn validators(mut self, validators: Vec<RegexValidator>) -> Self {
        self.validators = validators;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructureSpec {
    pub name: String,
    pub fields: Vec<StructureFieldSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationSpec {
    pub task: String,
    pub labels: Vec<String>,
    pub label_descriptions: Vec<(String, String)>,
    pub multi_label: bool,
    pub cls_threshold: f32,
    pub class_act: ClassAct,
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntitySpec {
    pub name: String,
    pub description: Option<String>,
    pub dtype: FieldDtype,
    pub threshold: Option<f32>,
}

impl EntitySpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            dtype: FieldDtype::List,
            threshold: None,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn dtype(mut self, dtype: FieldDtype) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationSpec {
    pub name: String,
    pub description: Option<String>,
    pub threshold: Option<f32>,
}

impl RelationSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            threshold: None,
        }
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchemaSpec {
    pub entities: Vec<EntitySpec>,
    pub classifications: Vec<ClassificationSpec>,
    pub structures: Vec<StructureSpec>,
    pub relations: Vec<RelationSpec>,
}

impl Default for SchemaSpec {
    fn default() -> Self {
        Self {
            entities: Vec::new(),
            classifications: Vec::new(),
            structures: Vec::new(),
            relations: Vec::new(),
        }
    }
}

pub struct SchemaBuilder {
    spec: SchemaSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EntityLabels {
    Labels(Vec<String>),
    LabelsWithDescriptions(Vec<(String, String)>),
    Specs(Vec<EntitySpec>),
}

impl EntityLabels {
    fn into_specs(self, options: &EntityOptions) -> Vec<EntitySpec> {
        match self {
            Self::Labels(labels) => labels
                .into_iter()
                .map(|name| EntitySpec {
                    name,
                    description: None,
                    dtype: options.dtype.clone(),
                    threshold: options.threshold,
                })
                .collect(),
            Self::LabelsWithDescriptions(pairs) => pairs
                .into_iter()
                .map(|(name, description)| EntitySpec {
                    name,
                    description: Some(description),
                    dtype: options.dtype.clone(),
                    threshold: options.threshold,
                })
                .collect(),
            Self::Specs(specs) => specs,
        }
    }
}

impl From<Vec<String>> for EntityLabels {
    fn from(value: Vec<String>) -> Self {
        Self::Labels(value)
    }
}

impl From<Vec<(String, String)>> for EntityLabels {
    fn from(value: Vec<(String, String)>) -> Self {
        Self::LabelsWithDescriptions(value)
    }
}

impl From<BTreeMap<String, String>> for EntityLabels {
    fn from(value: BTreeMap<String, String>) -> Self {
        Self::LabelsWithDescriptions(value.into_iter().collect())
    }
}

impl From<Vec<EntitySpec>> for EntityLabels {
    fn from(value: Vec<EntitySpec>) -> Self {
        Self::Specs(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityOptions {
    pub dtype: FieldDtype,
    pub threshold: Option<f32>,
}

impl Default for EntityOptions {
    fn default() -> Self {
        Self {
            dtype: FieldDtype::List,
            threshold: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationLabels {
    Labels(Vec<String>),
    LabelsWithDescriptions(Vec<(String, String)>),
}

impl ClassificationLabels {
    fn into_parts(self) -> (Vec<String>, Vec<(String, String)>) {
        match self {
            Self::Labels(labels) => (labels, Vec::new()),
            Self::LabelsWithDescriptions(pairs) => {
                let labels = pairs.iter().map(|(l, _)| l.clone()).collect();
                (labels, pairs)
            }
        }
    }
}

impl From<Vec<String>> for ClassificationLabels {
    fn from(value: Vec<String>) -> Self {
        Self::Labels(value)
    }
}

impl From<Vec<(String, String)>> for ClassificationLabels {
    fn from(value: Vec<(String, String)>) -> Self {
        Self::LabelsWithDescriptions(value)
    }
}

impl From<BTreeMap<String, String>> for ClassificationLabels {
    fn from(value: BTreeMap<String, String>) -> Self {
        Self::LabelsWithDescriptions(value.into_iter().collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuickClassificationTask {
    Labels(ClassificationLabels),
    Config {
        labels: ClassificationLabels,
        options: ClassificationOptions,
    },
}

impl QuickClassificationTask {
    pub fn labels(labels: impl Into<ClassificationLabels>) -> Self {
        Self::Labels(labels.into())
    }

    pub fn config(labels: impl Into<ClassificationLabels>, options: ClassificationOptions) -> Self {
        Self::Config {
            labels: labels.into(),
            options,
        }
    }
}

impl From<Vec<String>> for QuickClassificationTask {
    fn from(value: Vec<String>) -> Self {
        Self::labels(value)
    }
}

impl From<Vec<(String, String)>> for QuickClassificationTask {
    fn from(value: Vec<(String, String)>) -> Self {
        Self::labels(value)
    }
}

impl From<ClassificationLabels> for QuickClassificationTask {
    fn from(value: ClassificationLabels) -> Self {
        Self::Labels(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationOptions {
    pub multi_label: bool,
    pub cls_threshold: f32,
    pub class_act: ClassAct,
    pub prompt: Option<String>,
}

impl Default for ClassificationOptions {
    fn default() -> Self {
        Self {
            multi_label: false,
            cls_threshold: 0.5,
            class_act: ClassAct::Auto,
            prompt: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationLabels {
    Labels(Vec<String>),
    LabelsWithDescriptions(Vec<(String, String)>),
    Specs(Vec<RelationSpec>),
}

impl RelationLabels {
    fn into_specs(self) -> Vec<RelationSpec> {
        match self {
            Self::Labels(labels) => labels.into_iter().map(RelationSpec::new).collect(),
            Self::LabelsWithDescriptions(pairs) => pairs
                .into_iter()
                .map(|(name, description)| RelationSpec::new(name).description(description))
                .collect(),
            Self::Specs(specs) => specs,
        }
    }
}

impl From<Vec<String>> for RelationLabels {
    fn from(value: Vec<String>) -> Self {
        Self::Labels(value)
    }
}

impl From<Vec<(String, String)>> for RelationLabels {
    fn from(value: Vec<(String, String)>) -> Self {
        Self::LabelsWithDescriptions(value)
    }
}

impl From<BTreeMap<String, String>> for RelationLabels {
    fn from(value: BTreeMap<String, String>) -> Self {
        Self::LabelsWithDescriptions(value.into_iter().collect())
    }
}

impl From<Vec<RelationSpec>> for RelationLabels {
    fn from(value: Vec<RelationSpec>) -> Self {
        Self::Specs(value)
    }
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self {
            spec: SchemaSpec::default(),
        }
    }

    pub fn entities(self, entities: impl Into<EntityLabels>) -> Self {
        self.entities_with_options(entities, EntityOptions::default())
    }

    pub fn entities_with_options(
        mut self,
        entities: impl Into<EntityLabels>,
        options: EntityOptions,
    ) -> Self {
        let incoming = entities.into().into_specs(&options);
        for mut spec in incoming {
            if let Some(existing) = self.spec.entities.iter_mut().find(|e| e.name == spec.name) {
                if spec.description.is_some() {
                    existing.description = spec.description.take();
                }
                existing.dtype = spec.dtype;
                if spec.threshold.is_some() {
                    existing.threshold = spec.threshold;
                }
            } else {
                self.spec.entities.push(spec);
            }
        }

        self
    }

    pub fn classification(
        self,
        task: impl Into<String>,
        labels: impl Into<ClassificationLabels>,
    ) -> Self {
        self.classification_with_options(task, labels, ClassificationOptions::default())
    }

    pub fn classification_with_options(
        mut self,
        task: impl Into<String>,
        labels: impl Into<ClassificationLabels>,
        options: ClassificationOptions,
    ) -> Self {
        let (labels, label_descriptions) = labels.into().into_parts();
        self.spec.classifications.push(ClassificationSpec {
            task: task.into(),
            labels,
            label_descriptions,
            multi_label: options.multi_label,
            cls_threshold: options.cls_threshold,
            class_act: options.class_act,
            prompt: options.prompt,
        });
        self
    }

    pub fn relations(mut self, relations: impl Into<RelationLabels>) -> Self {
        let incoming = relations.into().into_specs();
        for mut spec in incoming {
            if let Some(existing) = self.spec.relations.iter_mut().find(|r| r.name == spec.name) {
                if spec.description.is_some() {
                    existing.description = spec.description.take();
                }
                if spec.threshold.is_some() {
                    existing.threshold = spec.threshold;
                }
            } else {
                self.spec.relations.push(spec);
            }
        }
        self
    }

    pub fn structure(self, name: impl Into<String>) -> StructureBuilder {
        StructureBuilder {
            builder: self,
            current: StructureSpec {
                name: name.into(),
                fields: Vec::new(),
            },
        }
    }

    pub fn build(self) -> SchemaSpec {
        self.spec
    }
}

pub struct StructureBuilder {
    builder: SchemaBuilder,
    current: StructureSpec,
}

impl StructureBuilder {
    pub fn field(mut self, field: StructureFieldSpec) -> Self {
        self.current.fields.push(field);
        self
    }

    pub fn finish(mut self) -> SchemaBuilder {
        self.builder.spec.structures.push(self.current);
        self.builder
    }
}

/// Placeholder output type for combined extraction.
///
/// This will evolve as we implement tutorials #3–#6.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ExtractionResult {
    pub entities: BTreeMap<String, FormattedEntityValue>,
    pub classifications: BTreeMap<String, FormattedClassification>,
    pub structures: BTreeMap<String, Vec<BTreeMap<String, FormattedEntityValue>>>,
    pub relations: BTreeMap<String, Vec<FormattedRelationPair>>,
}
