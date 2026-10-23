//! Knowledge-grounded senses, typed lexical relations and attributed event evidence.
//!
//! Sparse fingerprints are retrieval features. The semantic scorer retains the
//! sense graph and event bindings rather than inferring meaning from bit overlap.

mod context;
mod context_evaluation;
mod corpus;
mod evidence;
mod information;
mod lexicon;
mod pos_evaluation;
mod relations;
mod training;

pub use context::{
    ByteSpan, ContextComparison, ContextSense, ContextualWord, PartOfSpeech, SensePrediction,
};
pub use context_evaluation::{
    evaluate_context_pairs, read_context_pairs, ContextEvaluation, ContextMetrics, ContextPair,
    ContextPairPrediction,
};
pub use corpus::{read_annotated_corpus, AnnotatedDocument, AnnotatedToken, ParserProvenance};
pub use evidence::{Argument, Event, EventRole, EvidenceMatch, EvidenceStatus};
pub use lexicon::{LexicalSense, Lexicon, LexiconSource, SenseLemma};
pub use pos_evaluation::{evaluate_pos_pairs, read_pos_word_pairs, PosPairPrediction, PosWordPair};
pub use relations::{Relation, SemanticMatch, SenseExplanation, WordQuery};
pub use training::train_semantic;

use crate::{FingerprintConfig, ProteusError, Result, TextConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

/// Whether senses describe lexical possibilities or observed corpus usages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SenseInventory {
    /// Preserve the original surface/lemma union across all lexical POS.
    #[default]
    Lexical,
    /// Require an observed lemma and compatible parser POS for each occurrence.
    Observed,
}

impl SenseInventory {
    fn is_lexical(&self) -> bool {
        *self == Self::Lexical
    }

    fn algorithm(self) -> &'static str {
        match self {
            Self::Lexical => "knowledge-grounded-senses-events-v1",
            Self::Observed => "knowledge-grounded-observed-senses-events-v2",
        }
    }
}

impl std::str::FromStr for SenseInventory {
    type Err = ProteusError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "lexical" => Ok(Self::Lexical),
            "observed" => Ok(Self::Observed),
            _ => Err(ProteusError::Config(
                "Sense inventory must be 'lexical' or 'observed'".into(),
            )),
        }
    }
}

/// Explicit choices for the knowledge-grounded backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Square feature-universe side length; these coordinates have no spatial topology.
    pub dimension: usize,
    /// Observed semantic-feature cardinality caps and legacy text-pooling policy.
    pub fingerprint: FingerprintConfig,
    /// Shared training, phrase lookup and inference normalization.
    pub text: TextConfig,
    /// Minimum corpus occurrence count for a retained word or phrase.
    pub min_count: usize,
    /// Maximum original parser-token distance for local associations.
    pub context_window: usize,
    /// Dimensionless exponent for context-marginal smoothing, in [0, 1].
    pub context_alpha: f64,
    /// Maximum positive PPMI coordinates retained per vocabulary entry.
    pub max_associations: usize,
    /// Maximum attributed, high-posterior example ranges retained per sense.
    pub max_examples_per_sense: usize,
    /// Maximum parser-token span considered for a lexicon-recognized phrase.
    pub max_phrase_tokens: usize,
    /// Positive lexical-prior mass in pseudo-occurrences.
    pub lexical_prior_strength: f64,
    /// Posterior cutoff for counting distinct text-content support; not calibrated confidence.
    pub min_sense_support: f64,
    /// Positive temperature for the gloss-overlap sense posterior.
    pub sense_temperature: f64,
    /// Candidate inventory and occurrence conditioning; absent old metadata is lexical.
    #[serde(default, skip_serializing_if = "SenseInventory::is_lexical")]
    pub sense_inventory: SenseInventory,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            fingerprint: FingerprintConfig {
                sparsity: 0.02,
                min_active_bits: 0,
                max_active_bits: usize::MAX,
                weighted_aggregation: true,
            },
            text: TextConfig::default(),
            min_count: 2,
            context_window: 5,
            context_alpha: 0.75,
            max_associations: 32,
            max_examples_per_sense: 3,
            max_phrase_tokens: 5,
            lexical_prior_strength: 5.0,
            min_sense_support: 0.5,
            sense_temperature: 0.15,
            sense_inventory: SenseInventory::Lexical,
        }
    }
}

impl SemanticConfig {
    /// Validate dimensions, finite numerical choices and fingerprint caps.
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0
            || self
                .dimension
                .checked_mul(self.dimension)
                .is_none_or(|size| size > u32::MAX as usize)
            || self.min_count == 0
            || self.context_window == 0
            || self.max_associations == 0
            || self.max_phrase_tokens == 0
            || !self.lexical_prior_strength.is_finite()
            || self.lexical_prior_strength <= 0.0
            || !self.min_sense_support.is_finite()
            || !(0.0..=1.0).contains(&self.min_sense_support)
            || !self.context_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.context_alpha)
            || !self.sense_temperature.is_finite()
            || self.sense_temperature <= 0.0
        {
            return Err(ProteusError::Config(
                "Invalid semantic training configuration".into(),
            ));
        }
        self.fingerprint.validate()
    }
}

/// Evidence for one lexical sense; probabilities describe an explicit model,
/// not a calibrated probability that a linguistic interpretation is correct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseEvidence {
    /// Stable identifier in the supplied lexical source.
    pub sense: String,
    /// Normalized corpus-adapted sense mass for a context-free query.
    pub prior: f64,
    /// Sum of context-posterior weights, in equivalent occurrences.
    pub occurrence_mass: f64,
    /// Distinct text-content groups meeting the declared posterior cutoff.
    pub supporting_documents: usize,
    /// Retained examples, each referencing the original document text.
    pub examples: Vec<OccurrenceReference>,
}

/// Original source range supporting a lexical occurrence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccurrenceReference {
    /// Identity of an attributed source document.
    pub document: String,
    /// Inclusive UTF-8 byte offset in that document.
    pub start: usize,
    /// Exclusive UTF-8 byte offset in that document.
    pub end: usize,
}

/// One retained surface word or explicitly recognized phrase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntry {
    /// Observed occurrences of this normalized word or explicit phrase.
    pub count: usize,
    /// Parser lemmas observed for this surface entry.
    pub lemmas: Vec<String>,
    /// Candidate senses and their corpus-adapted evidence.
    pub senses: Vec<SenseEvidence>,
    /// Occurrences admitting observed lemma/POS evidence; absent in lexical v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sense_occurrences: Option<usize>,
    /// L2-normalized, truncated PPMI coordinates for the association channel.
    pub associations: BTreeMap<String, f64>,
}

/// Reproducible identity and resolved settings for semantic training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticReport {
    /// Versioned numerical and extraction contract identifier.
    pub algorithm: String,
    /// Resolved configuration used to construct this model.
    pub config: SemanticConfig,
    /// SHA-256 of length-prefixed canonical annotated document records.
    pub corpus_sha256: String,
    /// SHA-256 of the canonical serialized lexical source.
    pub lexicon_sha256: String,
    /// Complete original documents, including parser identity and byte ranges.
    pub documents: usize,
    /// Normalized, nonempty source tokens; phrase occurrences are additional targets.
    pub tokens: usize,
    /// Number of retained word and phrase entries.
    pub vocabulary: usize,
    /// Entries with explicit lexical evidence; other entries remain unsupported.
    pub words_with_senses: usize,
    /// Extracted, source-attributed predicate/argument bindings.
    pub events: usize,
    /// Distinct parser, model and source-input identities.
    pub parsers: Vec<ParserProvenance>,
}

/// Persisted semantics and the source text needed to inspect retained evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticModel {
    /// Training contract and source identities.
    pub(crate) report: SemanticReport,
    /// Explicit external lexical knowledge, including its license.
    pub(crate) lexicon: Lexicon,
    /// Vocabulary in canonical key order; queries never silently expand it.
    pub(crate) entries: BTreeMap<String, SemanticEntry>,
    /// Complete original documents, including parser identity and byte ranges.
    pub(crate) documents: Vec<AnnotatedDocument>,
    /// Extracted, source-attributed predicate/argument bindings.
    pub(crate) events: Vec<Event>,
    #[serde(skip)]
    index: OnceLock<std::result::Result<Arc<lexicon::LexiconIndex>, String>>,
    #[serde(skip)]
    information_content: OnceLock<std::result::Result<Arc<Vec<f64>>, String>>,
}

impl SemanticModel {
    /// Resolved training configuration and source identities.
    pub fn report(&self) -> &SemanticReport {
        &self.report
    }

    /// Immutable lexical source; mutation cannot invalidate the cached graph.
    pub fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    /// Retained words and phrases in canonical key order.
    pub fn entries(&self) -> &BTreeMap<String, SemanticEntry> {
        &self.entries
    }

    /// Complete original documents with parser annotations.
    pub fn documents(&self) -> &[AnnotatedDocument] {
        &self.documents
    }

    /// Bound, source-attributed events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    fn index(&self) -> Result<&lexicon::LexiconIndex> {
        self.index
            .get_or_init(|| {
                self.lexicon
                    .build_index(&self.report.config.text)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map(Arc::as_ref)
            .map_err(|message| ProteusError::InvalidRetinaFormat(message.clone()))
    }

    /// Canonical phrase normalization; unlike old single-token lookup this also
    /// permits phrases represented explicitly by the lexical source.
    pub fn normalize(&self, text: &str) -> Option<String> {
        let words =
            crate::Tokenizer::new(self.report.config.text.clone()).tokenize_to_strings(text);
        (!words.is_empty()).then(|| words.join(" "))
    }

    /// Validate report hashes, corpus ranges, sense evidence and extracted bindings.
    pub fn validate(&self) -> Result<()> {
        use sha2::{Digest, Sha256};

        self.report.config.validate()?;
        let index = self.index()?;
        if self.report.algorithm != self.report.config.sense_inventory.algorithm()
            || self.report.vocabulary != self.entries.len()
            || self.report.documents != self.documents.len()
            || self.report.events != self.events.len()
            || self.report.words_with_senses
                != self
                    .entries
                    .values()
                    .filter(|e| !e.senses.is_empty())
                    .count()
            || !is_sha256(&self.report.corpus_sha256)
            || !is_sha256(&self.report.lexicon_sha256)
        {
            return Err(invalid("Semantic report does not match model"));
        }
        let mut documents = BTreeMap::new();
        let mut corpus_digest = Sha256::new();
        let mut token_count = 0;
        let tokenizer = crate::Tokenizer::new(self.report.config.text.clone());
        for document in &self.documents {
            document.validate()?;
            let serialized =
                serde_json::to_vec(document).map_err(|error| invalid(&error.to_string()))?;
            corpus_digest.update((serialized.len() as u64).to_le_bytes());
            corpus_digest.update(serialized);
            token_count += document
                .tokens
                .iter()
                .filter(|token| {
                    !tokenizer
                        .tokenize_to_strings(&document.text[token.start..token.end])
                        .is_empty()
                })
                .count();
            if documents.insert(document.id.as_str(), document).is_some() {
                return Err(invalid("Duplicate document identity"));
            }
        }
        let lexical_bytes =
            serde_json::to_vec(&self.lexicon).map_err(|error| invalid(&error.to_string()))?;
        let expected_parsers: Vec<_> = self
            .documents
            .iter()
            .map(|document| document.parser.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if format!("{:x}", corpus_digest.finalize()) != self.report.corpus_sha256
            || format!("{:x}", Sha256::digest(lexical_bytes)) != self.report.lexicon_sha256
            || token_count != self.report.tokens
            || expected_parsers != self.report.parsers
        {
            return Err(invalid(
                "Semantic source digest or parser provenance mismatch",
            ));
        }
        if self.report.config.sense_inventory == SenseInventory::Observed {
            training::validate_observed_inventory(self, index, &tokenizer)?;
        }
        for (word, entry) in &self.entries {
            if self.normalize(word).as_deref() != Some(word) || entry.count == 0 {
                return Err(invalid("Invalid semantic vocabulary entry"));
            }
            let mut mass = 0.0;
            let mut occurrence_mass = 0.0;
            let mut seen = std::collections::BTreeSet::new();
            for sense in &entry.senses {
                if !index.ids.contains_key(&sense.sense)
                    || !seen.insert(&sense.sense)
                    || !sense.prior.is_finite()
                    || sense.prior <= 0.0
                    || !sense.occurrence_mass.is_finite()
                    || sense.occurrence_mass < 0.0
                    || sense.supporting_documents > self.documents.len()
                    || sense.examples.len() > self.report.config.max_examples_per_sense
                {
                    return Err(invalid("Invalid semantic sense evidence"));
                }
                mass += sense.prior;
                occurrence_mass += sense.occurrence_mass;
                for example in &sense.examples {
                    let document = documents
                        .get(example.document.as_str())
                        .ok_or_else(|| invalid("Unknown occurrence document"))?;
                    if example.start >= example.end
                        || document.text.get(example.start..example.end).is_none()
                    {
                        return Err(invalid("Invalid occurrence range"));
                    }
                    if self
                        .normalize(&document.text[example.start..example.end])
                        .as_deref()
                        != Some(word)
                    {
                        return Err(invalid("Occurrence text and lexical entry differ"));
                    }
                }
            }
            match (self.report.config.sense_inventory, entry.sense_occurrences) {
                (SenseInventory::Lexical, None) => {}
                (SenseInventory::Observed, Some(count))
                    if count <= entry.count
                        && (count == 0) == entry.senses.is_empty()
                        && (occurrence_mass - count as f64).abs()
                            <= 1e-9 * (count as f64).max(1.0) => {}
                _ => return Err(invalid("Invalid sense occurrence accounting")),
            }
            if !entry.senses.is_empty() && (mass - 1.0).abs() > 1e-8 {
                return Err(invalid("Sense prior mass must sum to one"));
            }
            if entry
                .associations
                .values()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            {
                return Err(invalid("Invalid association coordinate"));
            }
        }
        for event in &self.events {
            event.validate(&documents)?;
        }
        let expected_events: Vec<_> = self
            .documents
            .iter()
            .flat_map(|document| evidence::extract_events(document, &self.report.config.text))
            .collect();
        if expected_events != self.events {
            return Err(invalid(
                "Stored event bindings differ from attributed annotations",
            ));
        }
        Ok(())
    }
}

pub(super) fn invalid(message: &str) -> ProteusError {
    ProteusError::InvalidRetinaFormat(message.into())
}

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
