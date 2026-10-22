//! Target-aware sense inference and comparison of two independently conditioned usages.

use super::information::canonical_pos;
use super::{invalid, Relation, SemanticMatch, SemanticModel, WordQuery};
use crate::{ProteusError, Result, Tokenizer};
use serde::{Deserialize, Serialize};

/// Half-open UTF-8 byte range in the original context string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteSpan {
    /// Inclusive source byte offset.
    pub start: usize,
    /// Exclusive source byte offset.
    pub end: usize,
}

/// Lexical POS restriction. Adjective satellites belong to Adjective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartOfSpeech {
    /// Noun sense.
    Noun,
    /// Verb sense.
    Verb,
    /// Adjective or adjective-satellite sense.
    Adjective,
    /// Adverb sense.
    Adverb,
}

impl PartOfSpeech {
    fn lexical_code(self) -> &'static str {
        match self {
            Self::Noun => "n",
            Self::Verb => "v",
            Self::Adjective => "a",
            Self::Adverb => "r",
        }
    }
}

impl std::str::FromStr for PartOfSpeech {
    type Err = ProteusError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "n" | "noun" => Ok(Self::Noun),
            "v" | "verb" => Ok(Self::Verb),
            "a" | "s" | "adjective" => Ok(Self::Adjective),
            "r" | "adverb" => Ok(Self::Adverb),
            _ => Err(ProteusError::Config(
                "Unknown contextual part of speech".into(),
            )),
        }
    }
}

/// One retained word interpreted in its own context. An optional target range
/// removes that occurrence from evidence and checks its observed lemma identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualWord {
    /// Corpus vocabulary key or its normalized spelling.
    pub word: String,
    /// Original UTF-8 context, without normalization or reflow.
    pub context: String,
    /// Optional known target POS; no POS tagger runs during inference.
    #[serde(default)]
    pub part_of_speech: Option<PartOfSpeech>,
    /// Optional occurrence to exclude. Without a range, all context is supplied evidence.
    #[serde(default)]
    pub target: Option<ByteSpan>,
    /// Optional caller-supplied lemma for an annotated target form not observed
    /// in the corpus. Its identity must match the requested word or stored lemma.
    #[serde(default)]
    pub target_lemma: Option<String>,
}

impl ContextualWord {
    /// Construct a query using supplied surrounding text and no POS restriction.
    pub fn new(word: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            word: word.into(),
            context: context.into(),
            part_of_speech: None,
            target: None,
            target_lemma: None,
        }
    }
}

/// Source-grounded mass and literal overlap for one candidate sense.
#[derive(Debug, Clone, Serialize)]
pub struct ContextSense {
    /// Lexical sense identifier.
    pub id: String,
    /// Source definition and retained examples.
    pub definition: String,
    /// Prior conditioned on the requested POS.
    pub prior: f64,
    /// Normalized context-weighted mass; not calibrated correctness probability.
    pub posterior: f64,
    /// Distinct normalized words shared by the target-excluded context and gloss.
    pub matching_terms: Vec<String>,
}

/// Inspectable target-aware prediction, including unavailable coverage and ambiguity.
#[derive(Debug, Clone, Serialize)]
pub struct SensePrediction {
    /// Named inference contract, independent of the stored training algorithm.
    pub algorithm: &'static str,
    /// Canonical requested vocabulary key.
    pub word: String,
    /// Applied target POS restriction.
    pub part_of_speech: Option<PartOfSpeech>,
    /// Original byte range excluded from context evidence.
    pub target: Option<ByteSpan>,
    /// Applied caller-supplied target lemma, if any; no lemmatizer is invoked.
    pub target_lemma: Option<String>,
    /// Highest-posterior sense, with lexical identifier tie-breaking.
    pub selected_sense: Option<String>,
    /// Why no sense is available; this is not an artificial zero score.
    pub unavailable: Option<String>,
    /// Whether any candidate gloss has a literal overlap after target exclusion.
    pub has_gloss_overlap: bool,
    /// Shannon entropy of candidate masses, in natural-log units (nats).
    pub entropy_nats: Option<f64>,
    /// Highest minus second-highest mass; a distribution diagnostic only.
    pub posterior_margin: Option<f64>,
    /// Every candidate in decreasing posterior and then lexical identifier order.
    pub senses: Vec<ContextSense>,
}

/// Bilateral contextual comparison; each usage has a separate posterior.
#[derive(Debug, Clone, Serialize)]
pub struct ContextComparison {
    /// Inspectable first usage.
    pub left: SensePrediction,
    /// Inspectable second usage.
    pub right: SensePrediction,
    /// Whether MAP sense identities agree; None when either usage is unsupported.
    pub same_sense: Option<bool>,
    /// Sum of products for shared sense identities; not calibrated equivalence confidence.
    pub posterior_agreement: Option<f64>,
    /// Posterior-weighted taxonomy similarity across both sense distributions.
    pub expected_similarity: Option<f64>,
}

impl SemanticModel {
    /// Infer senses after explicit POS filtering and optional target exclusion.
    /// No new vocabulary, stemming, parser invocation or download is implicit.
    pub fn disambiguate(&self, query: &ContextualWord) -> Result<SensePrediction> {
        let word = self
            .normalize(&query.word)
            .ok_or_else(|| ProteusError::Config("Empty contextual word query".into()))?;
        if let Some(span) = query.target {
            if span.start >= span.end || query.context.get(span.start..span.end).is_none() {
                return Err(ProteusError::Config(
                    "Invalid UTF-8 target byte range".into(),
                ));
            }
        }
        if query.target.is_none() && query.target_lemma.is_some() {
            return Err(ProteusError::Config(
                "A target lemma requires an explicit target range".into(),
            ));
        }
        let mut output = SensePrediction {
            algorithm: "target-gloss-v1",
            word: word.clone(),
            part_of_speech: query.part_of_speech,
            target: query.target,
            target_lemma: query.target_lemma.clone(),
            selected_sense: None,
            unavailable: None,
            has_gloss_overlap: false,
            entropy_nats: None,
            posterior_margin: None,
            senses: Vec::new(),
        };
        let Some(entry) = self.entries.get(&word) else {
            output.unavailable = Some("word_not_in_corpus".into());
            return Ok(output);
        };
        let context = self.target_excluded_context(query, &word)?;
        let index = self.index()?;
        let mut candidates = Vec::new();
        for sense in &entry.senses {
            let sense_id = *index
                .ids
                .get(&sense.sense)
                .ok_or_else(|| invalid("Unknown contextual lexical sense"))?;
            if query.part_of_speech.is_none_or(|pos| {
                canonical_pos(&self.lexicon.senses[sense_id].pos) == pos.lexical_code()
            }) {
                candidates.push((sense_id, sense.prior));
            }
        }
        if candidates.is_empty() {
            output.unavailable = Some("no_lexical_sense_for_requested_pos".into());
            return Ok(output);
        }
        let total: f64 = candidates.iter().map(|(_, prior)| prior).sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid("Invalid contextual prior mass"));
        }
        for (_, prior) in &mut candidates {
            *prior /= total;
        }
        let posterior = index.posterior(&candidates, &context, &self.report.config);
        let terms: std::collections::BTreeSet<_> = Tokenizer::new(self.report.config.text.clone())
            .tokenize_to_strings(&context)
            .into_iter()
            .collect();
        for ((sense_id, prior), probability) in candidates.into_iter().zip(posterior) {
            let sense = &self.lexicon.senses[sense_id];
            output.senses.push(ContextSense {
                id: sense.id.clone(),
                definition: sense.definition.clone(),
                prior,
                posterior: probability,
                matching_terms: index.glosses[sense_id]
                    .intersection(&terms)
                    .cloned()
                    .collect(),
            });
        }
        output.senses.sort_by(|left, right| {
            right
                .posterior
                .total_cmp(&left.posterior)
                .then(left.id.cmp(&right.id))
        });
        output.selected_sense = Some(output.senses[0].id.clone());
        output.has_gloss_overlap = output
            .senses
            .iter()
            .any(|sense| !sense.matching_terms.is_empty());
        output.entropy_nats = Some(
            -output
                .senses
                .iter()
                .filter(|sense| sense.posterior > 0.0)
                .map(|sense| sense.posterior * sense.posterior.ln())
                .sum::<f64>(),
        );
        output.posterior_margin = Some(
            output.senses[0].posterior - output.senses.get(1).map_or(0.0, |sense| sense.posterior),
        );
        Ok(output)
    }

    fn target_excluded_context(&self, query: &ContextualWord, word: &str) -> Result<String> {
        let Some(span) = query.target else {
            return Ok(query.context.clone());
        };
        let tokenizer = Tokenizer::new(self.report.config.text.clone());
        if tokenizer.tokenize(&query.context).iter().any(|token| {
            token.start < span.end
                && token.end > span.start
                && (token.start < span.start || token.end > span.end)
        }) {
            return Err(ProteusError::Config(
                "Target range cuts through a normalized word".into(),
            ));
        }
        let surface = self.normalize(
            query
                .target_lemma
                .as_deref()
                .unwrap_or(&query.context[span.start..span.end]),
        );
        let matches = surface.as_deref().is_some_and(|surface| {
            surface == word
                || self
                    .entries
                    .get(surface)
                    .is_some_and(|entry| entry.lemmas.iter().any(|lemma| lemma == word))
                || self.entries[word]
                    .lemmas
                    .iter()
                    .any(|lemma| lemma == surface)
        });
        if !matches {
            return Err(ProteusError::Config(
                "Target identity does not match the word or an observed lemma form".into(),
            ));
        }
        Ok(format!(
            "{} {}",
            &query.context[..span.start],
            &query.context[span.end..]
        ))
    }

    /// Compare two independently interpreted usages. Argument reversal preserves
    /// all scalar results; missing lexical coverage remains None.
    pub fn compare_contexts(
        &self,
        left: &ContextualWord,
        right: &ContextualWord,
    ) -> Result<ContextComparison> {
        let left = self.disambiguate(left)?;
        let right = self.disambiguate(right)?;
        let available = left.selected_sense.is_some() && right.selected_sense.is_some();
        let index = self.index()?;
        let mut agreement = 0.0;
        let mut expected = 0.0;
        if available {
            for first in &left.senses {
                for second in &right.senses {
                    let weight = first.posterior * second.posterior;
                    if first.id == second.id {
                        agreement += weight;
                    }
                    expected += weight
                        * index.sense_similarity(
                            index.ids[&first.id],
                            index.ids[&second.id],
                            &self.lexicon,
                        );
                }
            }
        }
        Ok(ContextComparison {
            same_sense: available.then(|| left.selected_sense == right.selected_sense),
            posterior_agreement: available.then_some(agreement.clamp(0.0, 1.0)),
            expected_similarity: available.then_some(expected.clamp(0.0, 1.0)),
            left,
            right,
        })
    }

    /// Rank the retained vocabulary using one target-aware posterior, computed
    /// once. Query masses average the declared sense-conditioned relation score.
    pub fn find_similar_contextual(
        &self,
        query: &ContextualWord,
        relation: Relation,
        k: usize,
    ) -> Result<Vec<SemanticMatch>> {
        if relation == Relation::Association {
            return Err(ProteusError::Config(
                "Target-aware ranking requires a lexical relation".into(),
            ));
        }
        let prediction = self.disambiguate(query)?;
        if prediction.selected_sense.is_none() {
            return Err(ProteusError::WordNotFound(query.word.clone()));
        }
        let mut results = Vec::new();
        for word in self.entries.keys().filter(|word| **word != prediction.word) {
            let mut result: Option<SemanticMatch> = None;
            let mut best_contribution = -1.0;
            for sense in &prediction.senses {
                if let Some(scored) = self.score(
                    WordQuery {
                        word: &prediction.word,
                        context: None,
                        sense: Some(&sense.id),
                    },
                    word,
                    relation,
                )? {
                    let contribution = sense.posterior * scored.score;
                    let candidate = result.get_or_insert_with(|| SemanticMatch {
                        score: 0.0,
                        ..scored.clone()
                    });
                    candidate.score += contribution;
                    if contribution > best_contribution {
                        candidate.query_sense = scored.query_sense;
                        candidate.candidate_sense = scored.candidate_sense;
                        best_contribution = contribution;
                    }
                }
            }
            if let Some(mut result) = result {
                result.score = result.score.clamp(0.0, 1.0);
                if result.score > 0.0
                    || matches!(
                        relation,
                        Relation::Similarity
                            | Relation::ExpectedSimilarity
                            | Relation::InformationSimilarity
                    )
                {
                    results.push(result);
                }
            }
        }
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then(left.word.cmp(&right.word))
        });
        results.truncate(k);
        Ok(results)
    }
}
