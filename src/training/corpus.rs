//! Canonical normalized corpus shared by the training backends.

use super::{read_snippets, TrainingConfig};
use crate::{ProteusError, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::BufRead;

pub(super) struct PreparedCorpus {
    pub snippets: Vec<Vec<String>>,
    pub word_counts: BTreeMap<String, usize>,
    pub document_frequency: BTreeMap<String, usize>,
    pub digest: String,
}

impl PreparedCorpus {
    pub(super) fn read(reader: impl BufRead, config: &TrainingConfig) -> Result<Self> {
        let snippets = read_snippets(reader, config)?;
        if snippets.is_empty() {
            return Err(ProteusError::EmptyInput(
                "No normalized corpus tokens".into(),
            ));
        }
        let mut digest = Sha256::new();
        let mut word_counts = BTreeMap::<String, usize>::new();
        let mut document_frequency = BTreeMap::<String, usize>::new();
        for snippet in &snippets {
            digest.update((snippet.len() as u64).to_le_bytes());
            let mut distinct_words = std::collections::BTreeSet::new();
            for word in snippet {
                digest.update((word.len() as u64).to_le_bytes());
                digest.update(word.as_bytes());
                *word_counts.entry(word.clone()).or_default() += 1;
                distinct_words.insert(word);
            }
            for word in distinct_words {
                *document_frequency.entry(word.clone()).or_default() += 1;
            }
        }
        word_counts.retain(|_, count| *count >= config.min_count);
        if word_counts.is_empty() {
            return Err(ProteusError::Training("No words meet min_count".into()));
        }
        Ok(Self {
            snippets,
            word_counts,
            document_frequency,
            digest: format!("{:x}", digest.finalize()),
        })
    }

    pub(super) fn feature_names(&self, maximum_document_frequency: f64) -> Result<Vec<&str>> {
        let features: Vec<_> = self
            .word_counts
            .keys()
            .filter(|word| {
                self.document_frequency[*word] as f64 / self.snippets.len() as f64
                    <= maximum_document_frequency
            })
            .map(String::as_str)
            .collect();
        if features.is_empty() {
            return Err(ProteusError::Training(
                "No features remain; increase max_document_frequency or supply more diverse contexts".into(),
            ));
        }
        Ok(features)
    }

    pub(super) fn inverse_document_frequency(&self, features: &[&str]) -> Vec<f32> {
        features
            .iter()
            .map(|word| {
                ((1.0 + self.snippets.len() as f64) / (1.0 + self.document_frequency[*word] as f64))
                    .ln() as f32
                    + 1.0
            })
            .collect()
    }
}
