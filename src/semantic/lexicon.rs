//! Validated lexical graph with explicitly directed taxonomy edges.

use super::invalid;
use crate::{Result, TextConfig, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// External lexical resource identity and license.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconSource {
    /// Lexical resource name.
    pub name: String,
    /// Declared lexical resource version.
    pub version: String,
    /// Canonical resource or publisher URL.
    pub url: String,
    /// SHA-256 of the resource archive used by preparation.
    pub sha256: String,
    /// Verbatim license supplied with the lexical resource.
    pub license: String,
}

/// One lexical alternative and its resource-supplied usage count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseLemma {
    /// Lexical lemma, including spaces for multiword units.
    pub text: String,
    /// Resource-supplied tagged sense frequency; zero means no tagged occurrences.
    pub count: usize,
}

/// A source-defined meaning and its typed outgoing lexical relations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalSense {
    /// Stable sense identity in the lexical resource namespace.
    pub id: String,
    /// WordNet POS: n, v, a, s or r. Satellite adjectives share the a namespace.
    pub pos: String,
    /// Resource gloss and examples used by the explicit overlap disambiguator.
    pub definition: String,
    /// Lexical alternatives belonging to this sense.
    pub lemmas: Vec<SenseLemma>,
    /// Directed hypernym or instance-hypernym sense identifiers.
    pub parents: Vec<String>,
    /// Explicit similar-to or verb-group sense identifiers; not hierarchy edges.
    pub similar: Vec<String>,
    /// Explicit opposing sense identifiers.
    pub antonyms: Vec<String>,
}

/// Portable lexical preparation schema; no download occurs in the native reader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lexicon {
    /// Preparation schema version; currently 1.
    pub schema: u32,
    /// Resource identity and redistribution attribution.
    pub source: LexiconSource,
    /// Complete lexical graph; unresolved references and hierarchy cycles are rejected.
    pub senses: Vec<LexicalSense>,
}

#[derive(Debug, Clone)]
pub(super) struct LexiconIndex {
    pub ids: BTreeMap<String, usize>,
    /// Lexical alternatives belonging to this sense.
    pub lemmas: BTreeMap<String, Vec<usize>>,
    pub ancestors: Vec<BTreeMap<usize, u32>>,
    pub depth: Vec<u32>,
    /// Explicit similar-to or verb-group sense identifiers; not hierarchy edges.
    pub similar: Vec<BTreeSet<usize>>,
    /// Explicit opposing sense identifiers.
    pub antonyms: Vec<BTreeSet<usize>>,
    pub glosses: Vec<BTreeSet<String>>,
    pub idf: BTreeMap<String, f64>,
}

impl Lexicon {
    /// Read and validate a lexical graph without downloading external resources.
    pub fn read(reader: impl std::io::Read) -> Result<Self> {
        let lexicon: Self = serde_json::from_reader(reader)
            .map_err(|error| invalid(&format!("Invalid lexical source: {error}")))?;
        lexicon.build_index(&TextConfig::default())?;
        Ok(lexicon)
    }

    pub(super) fn build_index(&self, text: &TextConfig) -> Result<LexiconIndex> {
        if self.schema != 1
            || self.senses.is_empty()
            || self.source.name.is_empty()
            || self.source.version.is_empty()
            || self.source.url.is_empty()
            || self.source.license.is_empty()
            || !super::is_sha256(&self.source.sha256)
        {
            return Err(invalid("Invalid lexical schema or provenance"));
        }
        let tokenizer = Tokenizer::new(text.clone());
        let mut ids = BTreeMap::new();
        for (position, sense) in self.senses.iter().enumerate() {
            if sense.id.is_empty()
                || sense.id.len() > 256
                || sense.lemmas.is_empty()
                || !matches!(sense.pos.as_str(), "n" | "v" | "a" | "s" | "r")
                || ids.insert(sense.id.clone(), position).is_some()
            {
                return Err(invalid("Invalid or duplicate lexical sense"));
            }
        }
        let count = self.senses.len();
        let mut parents = vec![BTreeSet::new(); count];
        let mut children = vec![Vec::new(); count];
        let mut similar = vec![BTreeSet::new(); count];
        let mut antonyms = vec![BTreeSet::new(); count];
        let mut lemmas = BTreeMap::<String, Vec<usize>>::new();
        let mut glosses = Vec::with_capacity(count);
        let mut document_frequency = BTreeMap::<String, usize>::new();
        for (position, sense) in self.senses.iter().enumerate() {
            for lemma in &sense.lemmas {
                if lemma.text.trim().is_empty() {
                    return Err(invalid("Empty source sense lemma"));
                }
                let normalized = tokenizer
                    .tokenize_to_strings(&lemma.text.replace('_', " "))
                    .join(" ");
                if normalized.is_empty() {
                    // A lexical source can contain one-letter or punctuation
                    // lemmas excluded by the selected tokenizer. Retain the
                    // sense as a graph node without inventing a lookup key.
                    continue;
                }
                lemmas.entry(normalized).or_default().push(position);
            }
            for parent in &sense.parents {
                let target = *ids
                    .get(parent)
                    .ok_or_else(|| invalid("Unknown taxonomy parent"))?;
                if target == position || !parents[position].insert(target) {
                    return Err(invalid("Self or duplicate taxonomy edge"));
                }
                children[target].push(position);
            }
            for target in &sense.similar {
                let target = *ids
                    .get(target)
                    .ok_or_else(|| invalid("Unknown similarity sense"))?;
                similar[position].insert(target);
                similar[target].insert(position);
            }
            for target in &sense.antonyms {
                let target = *ids
                    .get(target)
                    .ok_or_else(|| invalid("Unknown antonym sense"))?;
                if target == position {
                    return Err(invalid("A sense cannot oppose itself"));
                }
                antonyms[position].insert(target);
                antonyms[target].insert(position);
            }
            let gloss: BTreeSet<_> = tokenizer
                .tokenize_to_strings(&sense.definition)
                .into_iter()
                .collect();
            for word in &gloss {
                *document_frequency.entry(word.clone()).or_default() += 1;
            }
            glosses.push(gloss);
        }
        for values in lemmas.values_mut() {
            values.sort_unstable();
            values.dedup();
        }
        let mut remaining: Vec<_> = parents.iter().map(BTreeSet::len).collect();
        let mut queue: VecDeque<_> = remaining
            .iter()
            .enumerate()
            .filter_map(|(position, &degree)| (degree == 0).then_some(position))
            .collect();
        let mut ancestors = vec![BTreeMap::new(); count];
        let mut depth = vec![1_u32; count];
        let mut visited = 0;
        while let Some(position) = queue.pop_front() {
            visited += 1;
            let mut current = BTreeMap::from([(position, 0_u32)]);
            for &parent in &parents[position] {
                depth[position] = depth[position].max(depth[parent] + 1);
                for (&ancestor, &distance) in &ancestors[parent] {
                    let entry = current.entry(ancestor).or_insert(distance + 1);
                    *entry = (*entry).min(distance + 1);
                }
            }
            // Bound adversarial DAG expansion before exposing a loaded model.
            if depth[position] > 256 || current.len() > 4096 {
                return Err(invalid("Taxonomy exceeds depth or ancestor limits"));
            }
            ancestors[position] = current;
            for &child in &children[position] {
                remaining[child] -= 1;
                if remaining[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        if visited != count {
            return Err(invalid("Cycle in lexical taxonomy"));
        }
        let idf = document_frequency
            .into_iter()
            .map(|(word, frequency)| {
                (
                    word,
                    ((count as f64 + 1.0) / (frequency as f64 + 1.0)).ln() + 1.0,
                )
            })
            .collect();
        Ok(LexiconIndex {
            ids,
            lemmas,
            ancestors,
            depth,
            similar,
            antonyms,
            glosses,
            idf,
        })
    }
}

impl LexiconIndex {
    pub fn sense_similarity(&self, left: usize, right: usize, lexicon: &Lexicon) -> f64 {
        if left == right {
            return 1.0;
        }
        if self.antonyms[left].contains(&right) {
            return 0.0;
        }
        if self.similar[left].contains(&right) {
            return 0.9;
        }
        let pos = |index: usize| {
            if lexicon.senses[index].pos == "s" {
                "a"
            } else {
                &lexicon.senses[index].pos
            }
        };
        if pos(left) != pos(right) {
            return 0.0;
        }
        let common_depth = self.ancestors[left]
            .keys()
            .filter(|ancestor| self.ancestors[right].contains_key(ancestor))
            .map(|&ancestor| self.depth[ancestor])
            .max()
            .unwrap_or(0);
        2.0 * common_depth as f64 / (self.depth[left] + self.depth[right]) as f64
    }

    pub fn posterior(
        &self,
        candidates: &[(usize, f64)],
        context: &str,
        config: &super::SemanticConfig,
    ) -> Vec<f64> {
        let context: BTreeSet<_> = Tokenizer::new(config.text.clone())
            .tokenize_to_strings(context)
            .into_iter()
            .collect();
        let mut scores = Vec::with_capacity(candidates.len());
        for &(sense, prior) in candidates {
            let shared: f64 = self.glosses[sense]
                .intersection(&context)
                .map(|word| self.idf.get(word).copied().unwrap_or(1.0).powi(2))
                .sum();
            let norm = |words: &BTreeSet<String>| {
                words
                    .iter()
                    .map(|word| self.idf.get(word).copied().unwrap_or(1.0).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            let denominator = norm(&context) * norm(&self.glosses[sense]);
            let overlap = if denominator > 0.0 {
                shared / denominator
            } else {
                0.0
            };
            scores.push((prior.ln(), overlap));
        }
        let maximum_overlap = scores
            .iter()
            .map(|&(_, overlap)| overlap)
            .fold(f64::NEG_INFINITY, f64::max);
        let scores: Vec<_> = scores
            .into_iter()
            .map(|(log_prior, overlap)| {
                log_prior + (overlap - maximum_overlap) / config.sense_temperature
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<_> = scores
            .into_iter()
            .map(|score| (score - maximum).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        weights
    }
}
