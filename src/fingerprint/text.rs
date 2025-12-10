//! Text fingerprint generation by aggregating word fingerprints.

use crate::config::FingerprintConfig;
use crate::fingerprint::{Sdr, WordFingerprinter};
use crate::text::Tokenizer;
use std::collections::HashMap;

/// Generates fingerprints for text by aggregating word fingerprints.
pub struct TextFingerprinter<'a> {
    /// Word fingerprinter reference.
    word_fingerprinter: &'a WordFingerprinter,
    /// Configuration.
    config: FingerprintConfig,
    /// Tokenizer.
    tokenizer: Tokenizer,
}

impl<'a> TextFingerprinter<'a> {
    /// Creates a new text fingerprinter.
    pub fn new(word_fingerprinter: &'a WordFingerprinter, config: FingerprintConfig) -> Self {
        Self {
            word_fingerprinter,
            config,
            tokenizer: Tokenizer::default_config(),
        }
    }

    /// Creates a text fingerprinter with a custom tokenizer.
    pub fn with_tokenizer(
        word_fingerprinter: &'a WordFingerprinter,
        config: FingerprintConfig,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            word_fingerprinter,
            config,
            tokenizer,
        }
    }

    /// Generates a fingerprint for text using simple union.
    ///
    /// Each word's fingerprint is OR'd together, then sparsified.
    pub fn fingerprint_union(&self, text: &str) -> Sdr {
        let tokens = self.tokenizer.tokenize_to_strings(text);
        let grid_size = self.word_fingerprinter.grid_size();

        let mut result = Sdr::new(grid_size);

        for token in &tokens {
            let word_fp = self.word_fingerprinter.get_or_empty(token);
            result = result.union(&word_fp);
        }

        // Sparsify to target density
        let max_bits = self.config.max_active_bits;
        result.sparsify(max_bits);

        result
    }

    /// Generates a fingerprint for text using weighted aggregation.
    ///
    /// Each position is weighted by how many words activate it,
    /// then the top positions are selected.
    pub fn fingerprint_weighted(&self, text: &str) -> Sdr {
        let tokens = self.tokenizer.tokenize_to_strings(text);
        let grid_size = self.word_fingerprinter.grid_size();

        // Count how many times each position is activated
        let mut position_counts: HashMap<u32, usize> = HashMap::new();

        for token in &tokens {
            if let Some(word_fp) = self.word_fingerprinter.get(token) {
                for pos in word_fp.fingerprint.iter() {
                    *position_counts.entry(pos).or_insert(0) += 1;
                }
            }
        }

        // Sort by count and take top positions
        let mut sorted: Vec<(u32, usize)> = position_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let target_bits = self.config.max_active_bits;
        let positions: Vec<u32> = sorted
            .into_iter()
            .take(target_bits)
            .map(|(pos, _)| pos)
            .collect();

        Sdr::from_positions(&positions, grid_size)
    }

    /// Generates a fingerprint using IDF-weighted aggregation.
    ///
    /// Positions activated by rare words get higher weight.
    pub fn fingerprint_idf_weighted(
        &self,
        text: &str,
        document_frequencies: &HashMap<String, f64>,
        num_documents: usize,
    ) -> Sdr {
        let tokens = self.tokenizer.tokenize_to_strings(text);
        let grid_size = self.word_fingerprinter.grid_size();

        // Accumulate IDF-weighted position scores
        let mut position_scores: HashMap<u32, f64> = HashMap::new();

        for token in &tokens {
            if let Some(word_fp) = self.word_fingerprinter.get(token) {
                // Compute IDF
                let df = document_frequencies.get(token).copied().unwrap_or(1.0);
                let idf = (num_documents as f64 / df).ln();

                for pos in word_fp.fingerprint.iter() {
                    *position_scores.entry(pos).or_insert(0.0) += idf;
                }
            }
        }

        // Sort by score and take top positions
        let mut sorted: Vec<(u32, f64)> = position_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let target_bits = self.config.max_active_bits;
        let positions: Vec<u32> = sorted
            .into_iter()
            .take(target_bits)
            .map(|(pos, _)| pos)
            .collect();

        Sdr::from_positions(&positions, grid_size)
    }

    /// Generates a fingerprint for text with TF-IDF weighting.
    pub fn fingerprint_tfidf(
        &self,
        text: &str,
        document_frequencies: &HashMap<String, f64>,
        num_documents: usize,
    ) -> Sdr {
        let tokens = self.tokenizer.tokenize_to_strings(text);
        let grid_size = self.word_fingerprinter.grid_size();

        // Count term frequencies
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        // Accumulate TF-IDF weighted position scores
        let mut position_scores: HashMap<u32, f64> = HashMap::new();

        for (token, tf) in &term_freq {
            if let Some(word_fp) = self.word_fingerprinter.get(token) {
                let df = document_frequencies.get(token).copied().unwrap_or(1.0);
                let idf = (num_documents as f64 / df).ln();
                let tfidf = (*tf as f64) * idf;

                for pos in word_fp.fingerprint.iter() {
                    *position_scores.entry(pos).or_insert(0.0) += tfidf;
                }
            }
        }

        // Sort by score and take top positions
        let mut sorted: Vec<(u32, f64)> = position_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let target_bits = self.config.max_active_bits;
        let positions: Vec<u32> = sorted
            .into_iter()
            .take(target_bits)
            .map(|(pos, _)| pos)
            .collect();

        Sdr::from_positions(&positions, grid_size)
    }

    /// Generates a fingerprint using the default method (weighted).
    pub fn fingerprint(&self, text: &str) -> Sdr {
        if self.config.weighted_aggregation {
            self.fingerprint_weighted(text)
        } else {
            self.fingerprint_union(text)
        }
    }

    /// Returns coverage statistics for a text.
    ///
    /// Returns (known_tokens, total_tokens, coverage_ratio).
    pub fn coverage(&self, text: &str) -> (usize, usize, f64) {
        let tokens = self.tokenizer.tokenize_to_strings(text);
        let total = tokens.len();

        if total == 0 {
            return (0, 0, 1.0);
        }

        let known = tokens
            .iter()
            .filter(|t| self.word_fingerprinter.contains(t))
            .count();

        (known, total, known as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::WordFingerprint;

    fn create_test_fingerprinter() -> WordFingerprinter {
        let config = FingerprintConfig {
            max_active_bits: 10,
            ..Default::default()
        };
        let mut wf = WordFingerprinter::new(config, 100);

        wf.insert(WordFingerprint::new("hello".to_string(), &[1, 2, 3], 100));
        wf.insert(WordFingerprint::new("world".to_string(), &[3, 4, 5], 100));
        wf.insert(WordFingerprint::new("test".to_string(), &[10, 20, 30], 100));

        wf
    }

    #[test]
    fn test_fingerprint_union() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig::default();
        let tf = TextFingerprinter::new(&wf, config);

        let fp = tf.fingerprint_union("hello world");

        // Should contain union of hello and world positions
        assert!(fp.contains(1));
        assert!(fp.contains(2));
        assert!(fp.contains(3));
        assert!(fp.contains(4));
        assert!(fp.contains(5));
    }

    #[test]
    fn test_fingerprint_weighted() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig {
            max_active_bits: 5,
            ..Default::default()
        };
        let tf = TextFingerprinter::new(&wf, config);

        let fp = tf.fingerprint_weighted("hello world");

        // Position 3 is in both hello and world, so should be included
        assert!(fp.contains(3));
        assert!(fp.cardinality() <= 5);
    }

    #[test]
    fn test_fingerprint_default() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig::default();
        let tf = TextFingerprinter::new(&wf, config);

        let fp = tf.fingerprint("hello world test");
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_coverage() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig::default();
        let tf = TextFingerprinter::new(&wf, config);

        let (known, total, ratio) = tf.coverage("hello world unknown");
        assert_eq!(known, 2);
        assert_eq!(total, 3);
        assert!((ratio - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_text() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig::default();
        let tf = TextFingerprinter::new(&wf, config);

        let fp = tf.fingerprint("");
        assert!(fp.is_empty());
    }

    #[test]
    fn test_unknown_words() {
        let wf = create_test_fingerprinter();
        let config = FingerprintConfig::default();
        let tf = TextFingerprinter::new(&wf, config);

        let fp = tf.fingerprint("unknown words only");
        assert!(fp.is_empty());
    }
}
