//! Word fingerprint generation and storage.

use crate::config::FingerprintConfig;
use crate::error::{ProteusError, Result};
use crate::fingerprint::Sdr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A word fingerprint with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordFingerprint {
    /// The word.
    pub word: String,
    /// The fingerprint (SDR).
    pub fingerprint: Sdr,
    /// Word frequency in the corpus (normalized).
    pub frequency: f64,
    /// Part-of-speech tags observed for this word.
    pub pos_tags: Vec<String>,
}

impl WordFingerprint {
    /// Creates a new word fingerprint.
    pub fn new(word: String, positions: &[u32], grid_size: u32) -> Self {
        Self {
            word,
            fingerprint: Sdr::from_positions(positions, grid_size),
            frequency: 0.0,
            pos_tags: Vec::new(),
        }
    }

    /// Creates a word fingerprint with frequency information.
    pub fn with_frequency(
        word: String,
        positions: &[u32],
        grid_size: u32,
        frequency: f64,
    ) -> Self {
        Self {
            word,
            fingerprint: Sdr::from_positions(positions, grid_size),
            frequency,
            pos_tags: Vec::new(),
        }
    }

    /// Adds a POS tag to this word.
    pub fn add_pos_tag(&mut self, tag: String) {
        if !self.pos_tags.contains(&tag) {
            self.pos_tags.push(tag);
        }
    }

    /// Returns the number of active bits.
    #[inline]
    pub fn cardinality(&self) -> u64 {
        self.fingerprint.cardinality()
    }
}

/// Generates word fingerprints from SOM training results.
pub struct WordFingerprinter {
    /// Configuration.
    config: FingerprintConfig,
    /// Grid size (dimension * dimension).
    grid_size: u32,
    /// Word fingerprints.
    fingerprints: HashMap<String, WordFingerprint>,
}

impl WordFingerprinter {
    /// Creates a new word fingerprinter.
    pub fn new(config: FingerprintConfig, grid_size: u32) -> Self {
        Self {
            config,
            grid_size,
            fingerprints: HashMap::new(),
        }
    }

    /// Creates fingerprints from word-to-BMU mappings.
    ///
    /// Each word's fingerprint consists of the positions where it was most
    /// frequently matched during SOM training.
    pub fn create_fingerprints(
        &mut self,
        word_to_bmus: &HashMap<String, Vec<usize>>,
        word_frequencies: Option<&HashMap<String, f64>>,
    ) {
        for (word, bmus) in word_to_bmus {
            // Count frequency of each position
            let mut position_counts: HashMap<usize, usize> = HashMap::new();
            for &bmu in bmus {
                *position_counts.entry(bmu).or_insert(0) += 1;
            }

            // Sort by count and take top positions
            let mut sorted: Vec<(usize, usize)> = position_counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            let target_bits = self.config.max_active_bits;
            let positions: Vec<u32> = sorted
                .into_iter()
                .take(target_bits)
                .map(|(pos, _)| pos as u32)
                .collect();

            let frequency = word_frequencies
                .and_then(|f| f.get(word))
                .copied()
                .unwrap_or(0.0);

            let fingerprint = WordFingerprint::with_frequency(
                word.clone(),
                &positions,
                self.grid_size,
                frequency,
            );

            self.fingerprints.insert(word.clone(), fingerprint);
        }
    }

    /// Creates fingerprints from weighted position counts.
    ///
    /// Each position is weighted by how strongly it was activated.
    pub fn create_weighted_fingerprints(
        &mut self,
        word_to_positions: &HashMap<String, Vec<(usize, f64)>>,
        word_frequencies: Option<&HashMap<String, f64>>,
    ) {
        for (word, weighted_positions) in word_to_positions {
            // Sort by weight and take top positions
            let mut sorted = weighted_positions.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let target_bits = self.config.max_active_bits;
            let positions: Vec<u32> = sorted
                .into_iter()
                .take(target_bits)
                .map(|(pos, _)| pos as u32)
                .collect();

            let frequency = word_frequencies
                .and_then(|f| f.get(word))
                .copied()
                .unwrap_or(0.0);

            let fingerprint = WordFingerprint::with_frequency(
                word.clone(),
                &positions,
                self.grid_size,
                frequency,
            );

            self.fingerprints.insert(word.clone(), fingerprint);
        }
    }

    /// Gets a word's fingerprint.
    pub fn get(&self, word: &str) -> Option<&WordFingerprint> {
        self.fingerprints.get(word)
    }

    /// Gets a word's fingerprint, returning an empty fingerprint for unknown words.
    pub fn get_or_empty(&self, word: &str) -> Sdr {
        self.fingerprints
            .get(word)
            .map(|wf| wf.fingerprint.clone())
            .unwrap_or_else(|| Sdr::new(self.grid_size))
    }

    /// Checks if a word exists in the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.fingerprints.contains_key(word)
    }

    /// Returns the number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Checks if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Returns an iterator over all word fingerprints.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &WordFingerprint)> {
        self.fingerprints.iter()
    }

    /// Returns the grid size.
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }

    /// Adds or updates a fingerprint.
    pub fn insert(&mut self, fingerprint: WordFingerprint) {
        self.fingerprints.insert(fingerprint.word.clone(), fingerprint);
    }

    /// Removes a word's fingerprint.
    pub fn remove(&mut self, word: &str) -> Option<WordFingerprint> {
        self.fingerprints.remove(word)
    }

    /// Computes similarity between two words.
    pub fn word_similarity(&self, word1: &str, word2: &str) -> Result<f64> {
        let fp1 = self
            .fingerprints
            .get(word1)
            .ok_or_else(|| ProteusError::WordNotFound(word1.to_string()))?;
        let fp2 = self
            .fingerprints
            .get(word2)
            .ok_or_else(|| ProteusError::WordNotFound(word2.to_string()))?;

        Ok(fp1.fingerprint.cosine_similarity(&fp2.fingerprint))
    }

    /// Finds the most similar words to a given word.
    pub fn find_similar(&self, word: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let target = self
            .fingerprints
            .get(word)
            .ok_or_else(|| ProteusError::WordNotFound(word.to_string()))?;

        let mut similarities: Vec<(String, f64)> = self
            .fingerprints
            .iter()
            .filter(|(w, _)| w.as_str() != word)
            .map(|(w, wf)| {
                let sim = target.fingerprint.cosine_similarity(&wf.fingerprint);
                (w.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Finds words similar to a given fingerprint.
    pub fn find_similar_to_fingerprint(&self, fp: &Sdr, k: usize) -> Vec<(String, f64)> {
        let mut similarities: Vec<(String, f64)> = self
            .fingerprints
            .iter()
            .map(|(w, wf)| {
                let sim = fp.cosine_similarity(&wf.fingerprint);
                (w.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        similarities
    }

    /// Consumes the fingerprinter and returns the fingerprints.
    pub fn into_fingerprints(self) -> HashMap<String, WordFingerprint> {
        self.fingerprints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> FingerprintConfig {
        FingerprintConfig {
            max_active_bits: 10,
            ..Default::default()
        }
    }

    #[test]
    fn test_word_fingerprint_creation() {
        let wf = WordFingerprint::new("test".to_string(), &[1, 5, 10], 100);
        assert_eq!(wf.word, "test");
        assert_eq!(wf.cardinality(), 3);
    }

    #[test]
    fn test_create_fingerprints() {
        let config = test_config();
        let mut fingerprinter = WordFingerprinter::new(config, 100);

        let mut word_to_bmus = HashMap::new();
        word_to_bmus.insert("hello".to_string(), vec![1, 1, 2, 3, 3, 3]);
        word_to_bmus.insert("world".to_string(), vec![10, 20, 20]);

        fingerprinter.create_fingerprints(&word_to_bmus, None);

        assert!(fingerprinter.contains("hello"));
        assert!(fingerprinter.contains("world"));

        let hello = fingerprinter.get("hello").unwrap();
        // Position 3 should be first (highest count)
        assert!(hello.fingerprint.contains(3));
    }

    #[test]
    fn test_word_similarity() {
        let config = test_config();
        let mut fingerprinter = WordFingerprinter::new(config, 100);

        fingerprinter.insert(WordFingerprint::new("a".to_string(), &[1, 2, 3, 4], 100));
        fingerprinter.insert(WordFingerprint::new("b".to_string(), &[3, 4, 5, 6], 100));
        fingerprinter.insert(WordFingerprint::new("c".to_string(), &[10, 20, 30, 40], 100));

        let sim_ab = fingerprinter.word_similarity("a", "b").unwrap();
        let sim_ac = fingerprinter.word_similarity("a", "c").unwrap();

        // a and b share {3, 4}, so should be more similar than a and c
        assert!(sim_ab > sim_ac);
    }

    #[test]
    fn test_find_similar() {
        let config = test_config();
        let mut fingerprinter = WordFingerprinter::new(config, 100);

        fingerprinter.insert(WordFingerprint::new("a".to_string(), &[1, 2, 3, 4], 100));
        fingerprinter.insert(WordFingerprint::new("b".to_string(), &[1, 2, 5, 6], 100));
        fingerprinter.insert(WordFingerprint::new("c".to_string(), &[10, 20, 30, 40], 100));

        let similar = fingerprinter.find_similar("a", 2).unwrap();
        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0, "b"); // Most similar
    }
}
