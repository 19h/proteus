//! Separate lexical-relation scores and context-conditioned sense explanations.

use super::SemanticModel;
use crate::Result;
use serde::{Deserialize, Serialize};

/// Explicit semantic question; symmetric and directed scores remain distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Relation {
    /// Best sense-pair lexical taxonomy score, with explicit opposition excluded.
    Similarity,
    /// Prior-weighted average of all sense-pair taxonomy scores.
    ExpectedSimilarity,
    /// Whether a pair shares an explicit lexical sense.
    Synonym,
    /// Whether a pair has an explicit opposition edge in the lexical source.
    Antonym,
    /// Directed score for a candidate that is an ancestor of the query sense.
    Hypernym,
    /// Directed score for a candidate that is a descendant of the query sense.
    Hyponym,
    /// Cosine between retained corpus PPMI coordinates, independent of taxonomy.
    Association,
}

impl std::str::FromStr for Relation {
    type Err = crate::ProteusError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "similarity" => Ok(Self::Similarity),
            "expected-similarity" => Ok(Self::ExpectedSimilarity),
            "synonym" => Ok(Self::Synonym),
            "antonym" => Ok(Self::Antonym),
            "hypernym" => Ok(Self::Hypernym),
            "hyponym" => Ok(Self::Hyponym),
            "association" => Ok(Self::Association),
            _ => Err(query_error("Unknown semantic relation")),
        }
    }
}

/// Word or phrase query with optional context and explicit sense selection.
#[derive(Debug, Clone, Copy)]
pub struct WordQuery<'a> {
    /// Normalized corpus vocabulary entry.
    pub word: &'a str,
    /// Optional surrounding text for gloss-overlap sense weighting.
    pub context: Option<&'a str>,
    /// Optional explicit lexical sense identifier, checked against this entry.
    pub sense: Option<&'a str>,
}

impl<'a> WordQuery<'a> {
    /// Create a context-free query over the retained vocabulary.
    pub fn new(word: &'a str) -> Self {
        Self {
            word,
            context: None,
            sense: None,
        }
    }
}

/// Inspectable sense mass and attributed examples.
#[derive(Debug, Clone, Serialize)]
pub struct SenseExplanation {
    /// Stable lexical sense identifier.
    pub id: String,
    /// Original lexical gloss and examples.
    pub definition: String,
    /// Lexical alternatives belonging to the sense.
    pub lemmas: Vec<String>,
    /// Corpus-adapted, context-free sense mass.
    pub prior: f64,
    /// Normalized overlap-model mass under the supplied query context.
    pub posterior: f64,
    /// Sum of corpus context-posterior weights.
    pub occurrence_mass: f64,
    /// Number of distinct supporting text-content groups.
    pub supporting_documents: usize,
    /// Retained source document ranges.
    pub examples: Vec<super::OccurrenceReference>,
}

/// One candidate under a declared relation scorer.
#[derive(Debug, Clone, Serialize)]
pub struct SemanticMatch {
    /// Normalized corpus vocabulary entry.
    pub word: String,
    /// Declared relation score, dimensionless and bounded by [0, 1].
    pub score: f64,
    /// Best contributing query sense; not necessarily the only contributor.
    pub query_sense: Option<String>,
    /// Best contributing candidate sense.
    pub candidate_sense: Option<String>,
    /// Observed occurrence count for this candidate.
    pub corpus_count: usize,
}

impl SemanticModel {
    /// Return source glosses, context-dependent masses and retained occurrence references.
    pub fn explain_senses(&self, query: WordQuery<'_>) -> Result<Vec<SenseExplanation>> {
        let word = self
            .normalize(query.word)
            .ok_or_else(|| query_error("Empty word query"))?;
        let entry = self
            .entries
            .get(&word)
            .ok_or_else(|| crate::ProteusError::WordNotFound(query.word.into()))?;
        let index = self.index()?;
        let candidates: Vec<_> = entry
            .senses
            .iter()
            .map(|sense| Ok((sense_id(index, &sense.sense)?, sense.prior)))
            .collect::<Result<_>>()?;
        let posterior = if let Some(context) = query.context {
            index.posterior(&candidates, context, &self.report.config)
        } else {
            entry.senses.iter().map(|sense| sense.prior).collect()
        };
        let mut output = Vec::new();
        for (sense, probability) in entry.senses.iter().zip(posterior) {
            if query.sense.is_some_and(|selected| selected != sense.sense) {
                continue;
            }
            let lexical = &self.lexicon.senses[sense_id(index, &sense.sense)?];
            output.push(SenseExplanation {
                id: sense.sense.clone(),
                definition: lexical.definition.clone(),
                lemmas: lexical
                    .lemmas
                    .iter()
                    .map(|lemma| lemma.text.clone())
                    .collect(),
                prior: sense.prior,
                posterior: if query.sense.is_some() {
                    1.0
                } else {
                    probability
                },
                occurrence_mass: sense.occurrence_mass,
                supporting_documents: sense.supporting_documents,
                examples: sense.examples.clone(),
            });
        }
        if query.sense.is_some() && output.is_empty() {
            return Err(query_error(
                "Selected sense does not belong to the query entry",
            ));
        }
        output.sort_by(|left, right| {
            right
                .posterior
                .total_cmp(&left.posterior)
                .then(left.id.cmp(&right.id))
        });
        Ok(output)
    }

    /// Best-sense taxonomy similarity is the default lexical score. It is not
    /// bitmap cosine or a calibrated equivalence probability. Expected similarity
    /// is exposed separately to make the ambiguity/coverage tradeoff measurable.
    pub fn similarity(&self, left: &str, right: &str, relation: Relation) -> Result<Option<f64>> {
        Ok(self
            .score(WordQuery::new(left), right, relation)?
            .map(|result| result.score))
    }

    /// Score one retained entry, returning None for unavailable lexical or association evidence.
    pub fn score(
        &self,
        query: WordQuery<'_>,
        candidate: &str,
        relation: Relation,
    ) -> Result<Option<SemanticMatch>> {
        let Some(word) = self.normalize(query.word) else {
            return Ok(None);
        };
        let Some(candidate) = self.normalize(candidate) else {
            return Ok(None);
        };
        let Some(left) = self.entries.get(&word) else {
            return Ok(None);
        };
        let Some(right) = self.entries.get(&candidate) else {
            return Ok(None);
        };
        if relation == Relation::Association {
            if query.context.is_some() || query.sense.is_some() {
                return Err(query_error("Association scores are word-level; context and sense selection require a lexical relation"));
            }
            if left.associations.is_empty() || right.associations.is_empty() {
                return Ok(None);
            }
            let numerator: f64 = left
                .associations
                .iter()
                .map(|(feature, weight)| {
                    weight * right.associations.get(feature).copied().unwrap_or(0.0)
                })
                .sum();
            let norm = |row: &std::collections::BTreeMap<String, f64>| {
                row.values()
                    .map(|weight| weight * weight)
                    .sum::<f64>()
                    .sqrt()
            };
            return Ok(Some(SemanticMatch {
                word: candidate,
                score: (numerator / (norm(&left.associations) * norm(&right.associations)))
                    .clamp(0.0, 1.0),
                query_sense: None,
                candidate_sense: None,
                corpus_count: right.count,
            }));
        }
        if left.senses.is_empty() || right.senses.is_empty() {
            return Ok(None);
        }
        let index = self.index()?;
        let candidates: Vec<_> = left
            .senses
            .iter()
            .map(|sense| Ok((sense_id(index, &sense.sense)?, sense.prior)))
            .collect::<Result<_>>()?;
        let weights = if let Some(context) = query.context {
            index.posterior(&candidates, context, &self.report.config)
        } else {
            left.senses.iter().map(|sense| sense.prior).collect()
        };
        let mut best = (0.0_f64, None, None);
        let mut aggregate = 0.0;
        let mut selected_found = query.sense.is_none();
        for (left_sense, posterior) in left.senses.iter().zip(weights) {
            if query
                .sense
                .is_some_and(|selected| selected != left_sense.sense)
            {
                continue;
            }
            selected_found = true;
            let posterior = if query.sense.is_some() {
                1.0
            } else {
                posterior
            };
            let left_id = sense_id(index, &left_sense.sense)?;
            let mut per_sense_best = 0.0_f64;
            for right_sense in &right.senses {
                let right_id = sense_id(index, &right_sense.sense)?;
                let score = match relation {
                    Relation::Similarity | Relation::ExpectedSimilarity => {
                        index.sense_similarity(left_id, right_id, &self.lexicon)
                    }
                    Relation::Synonym => f64::from(left_id == right_id),
                    Relation::Antonym => f64::from(index.antonyms[left_id].contains(&right_id)),
                    Relation::Hypernym => index.ancestors[left_id]
                        .get(&right_id)
                        .filter(|&&distance| distance > 0)
                        .map_or(0.0, |&distance| 1.0 / (distance as f64 + 1.0)),
                    Relation::Hyponym => index.ancestors[right_id]
                        .get(&left_id)
                        .filter(|&&distance| distance > 0)
                        .map_or(0.0, |&distance| 1.0 / (distance as f64 + 1.0)),
                    Relation::Association => {
                        unreachable!("Association returned before sense scoring")
                    }
                };
                if score > best.0 || best.1.is_none() {
                    best = (
                        score,
                        Some(left_sense.sense.clone()),
                        Some(right_sense.sense.clone()),
                    );
                }
                per_sense_best = per_sense_best.max(score);
                if relation == Relation::ExpectedSimilarity {
                    aggregate += posterior * right_sense.prior * score;
                }
            }
            if query.context.is_some()
                && query.sense.is_none()
                && relation != Relation::ExpectedSimilarity
            {
                aggregate += posterior * per_sense_best;
            }
        }
        if !selected_found {
            return Err(query_error(
                "Selected sense does not belong to the query entry",
            ));
        }
        let score = if relation == Relation::ExpectedSimilarity
            || (query.context.is_some() && query.sense.is_none())
        {
            aggregate
        } else {
            best.0
        };
        Ok(Some(SemanticMatch {
            word: candidate,
            score: score.clamp(0.0, 1.0),
            query_sense: best.1,
            candidate_sense: best.2,
            corpus_count: right.count,
        }))
    }

    /// Exhaustive relation ranking. No bitmap candidate cap can silently remove
    /// a semantic winner. Unknown lexical coverage remains distinct from zero.
    pub fn find_similar(
        &self,
        query: WordQuery<'_>,
        relation: Relation,
        k: usize,
    ) -> Result<Vec<SemanticMatch>> {
        let normalized = self
            .normalize(query.word)
            .ok_or_else(|| crate::ProteusError::WordNotFound(query.word.into()))?;
        if !self.entries.contains_key(&normalized) {
            return Err(crate::ProteusError::WordNotFound(query.word.into()));
        }
        let entry = &self.entries[&normalized];
        if (relation == Relation::Association && entry.associations.is_empty())
            || (relation != Relation::Association && entry.senses.is_empty())
        {
            return Err(crate::ProteusError::WordNotFound(format!(
                "{} (required semantic evidence unavailable)",
                query.word
            )));
        }
        let mut results = Vec::new();
        for word in self.entries.keys() {
            if word != &normalized {
                if let Some(result) = self.score(query, word, relation)? {
                    if result.score > 0.0
                        || matches!(
                            relation,
                            Relation::Similarity
                                | Relation::ExpectedSimilarity
                                | Relation::Association
                        )
                    {
                        results.push(result);
                    }
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

fn query_error(message: &str) -> crate::ProteusError {
    crate::ProteusError::Config(message.into())
}

fn sense_id(index: &super::lexicon::LexiconIndex, sense: &str) -> Result<usize> {
    index
        .ids
        .get(sense)
        .copied()
        .ok_or_else(|| super::invalid("Semantic entry references an unknown lexical sense"))
}
