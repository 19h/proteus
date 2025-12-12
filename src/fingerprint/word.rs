//! Word fingerprint generation and storage.

use crate::config::FingerprintConfig;
use crate::error::{ProteusError, Result};
use crate::fingerprint::Sdr;
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    ///
    /// Word frequencies are computed from BMU hit counts (normalized).
    /// If a progress bar is provided, it will be updated during processing.
    pub fn create_fingerprints(
        &mut self,
        word_to_bmus: &HashMap<String, Vec<usize>>,
        progress: Option<&ProgressBar>,
    ) {
        // Compute total BMU hits for frequency normalization
        let total_hits: usize = word_to_bmus.values().map(|bmus| bmus.len()).sum();
        let total_hits_f64 = total_hits as f64;

        let target_bits = self.config.max_active_bits;
        let grid_size = self.grid_size;

        // Update progress message
        if let Some(pb) = progress {
            pb.set_message("Generating fingerprints (parallel)...");
        }

        // Process fingerprints in parallel
        let progress_counter = AtomicUsize::new(0);
        let fingerprints: HashMap<String, WordFingerprint> = word_to_bmus
            .par_iter()
            .map(|(word, bmus)| {
                // Count frequency of each position
                let mut position_counts: HashMap<usize, usize> = HashMap::new();
                for &bmu in bmus {
                    *position_counts.entry(bmu).or_insert(0) += 1;
                }

                // Sort by count and take top positions
                let mut sorted: Vec<(usize, usize)> = position_counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));

                let positions: Vec<u32> = sorted
                    .into_iter()
                    .take(target_bits)
                    .map(|(pos, _)| pos as u32)
                    .collect();

                // Compute word frequency (normalized by total hits)
                let word_frequency = bmus.len() as f64 / total_hits_f64;

                let fingerprint = WordFingerprint::with_frequency(
                    word.clone(),
                    &positions,
                    grid_size,
                    word_frequency,
                );

                // Update progress
                let count = progress_counter.fetch_add(1, Ordering::Relaxed);
                if let Some(pb) = progress {
                    if count % 10_000 == 0 {
                        pb.set_position(count as u64);
                    }
                }

                (word.clone(), fingerprint)
            })
            .collect();

        self.fingerprints = fingerprints;

        if let Some(pb) = progress {
            pb.set_position(word_to_bmus.len() as u64);
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
    /// Filters out likely stopwords using heuristics.
    pub fn find_similar(&self, word: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let target = self
            .fingerprints
            .get(word)
            .ok_or_else(|| ProteusError::WordNotFound(word.to_string()))?;

        // Compute stopword scores for all words
        let stopword_scores = self.compute_stopword_scores();

        let mut similarities: Vec<(String, f64)> = self
            .fingerprints
            .iter()
            .filter(|(w, _)| w.as_str() != word)
            .filter(|(w, _)| {
                // Filter out likely stopwords (score > 0.35)
                stopword_scores.get(*w).copied().unwrap_or(0.0) < 0.35
            })
            .map(|(w, wf)| {
                let sim = target.fingerprint.cosine_similarity(&wf.fingerprint);
                (w.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Computes a stopword score for each word (0.0 = content word, 1.0 = likely stopword).
    ///
    /// Uses multiple heuristics:
    /// - Short length (1-3 chars)
    /// - High overlap with other words (generic fingerprint)
    /// - High frequency (if tracked)
    fn compute_stopword_scores(&self) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = HashMap::new();

        // First, compute average overlap for each word's fingerprint
        // Stopwords tend to have fingerprints that overlap with many other words
        let all_fingerprints: Vec<(&String, &WordFingerprint)> = self.fingerprints.iter().collect();

        for (word, wf) in &all_fingerprints {
            let mut score = 0.0;

            // Heuristic 1: Short words are more likely to be stopwords
            let len = word.len();
            if len == 1 {
                score += 0.5;
            } else if len == 2 {
                score += 0.4;
            } else if len == 3 {
                score += 0.25;
            } else if len == 4 {
                score += 0.1;
            }

            // Heuristic 2: Words with apostrophes are often stopwords (contractions)
            if word.contains('\'') || word.contains('\u{2019}') {
                score += 0.2;
            }

            // Heuristic 3: High frequency words (if frequency is tracked)
            // Thresholds tuned for large corpora where even common words may be < 0.01
            if wf.frequency > 0.005 {
                score += 0.4;  // Very high frequency - almost certainly a stopword
            } else if wf.frequency > 0.002 {
                score += 0.3;
            } else if wf.frequency > 0.001 {
                score += 0.2;
            } else if wf.frequency > 0.0005 {
                score += 0.1;
            }

            // Heuristic 4: Words that overlap highly with many other words
            // Sample a subset for efficiency
            let sample_size = 100usize.min(all_fingerprints.len());
            let mut overlap_count = 0usize;
            let mut sampled = 0usize;
            for (other_word, other_wf) in all_fingerprints.iter().take(sample_size) {
                if *other_word != *word {
                    let overlap = wf.fingerprint.overlap_count(&other_wf.fingerprint);
                    let max_overlap = wf.cardinality().min(other_wf.cardinality()) as f64;
                    if max_overlap > 0.0 && (overlap as f64 / max_overlap) > 0.3 {
                        overlap_count += 1;
                    }
                    sampled += 1;
                }
            }
            if sampled > 0 {
                let overlap_ratio = overlap_count as f64 / sampled as f64;
                if overlap_ratio > 0.5 {
                    score += 0.3;
                } else if overlap_ratio > 0.3 {
                    score += 0.2;
                } else if overlap_ratio > 0.15 {
                    score += 0.1;
                }
            }

            let final_score: f64 = score;
            scores.insert((*word).clone(), final_score.min(1.0));
        }

        scores
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
