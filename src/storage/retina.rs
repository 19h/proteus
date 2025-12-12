//! Retina management - the core semantic fingerprint database.

use crate::config::FingerprintConfig;
use crate::error::{ProteusError, Result};
use crate::fingerprint::{Sdr, TextFingerprinter, WordFingerprint, WordFingerprinter};
use crate::index::InvertedIndex;
use crate::similarity::{CosineSimilarity, SimilarityMeasure};
use crate::storage::format::RetinaFormat;
use std::collections::HashMap;
use std::path::Path;

/// A semantic retina containing word fingerprints and an inverted index.
///
/// The retina is the central component for semantic operations:
/// - Word to fingerprint lookup
/// - Text to fingerprint conversion
/// - Similarity search
/// - Semantic comparisons
pub struct Retina {
    /// Word fingerprinter containing all word SDRs.
    word_fingerprinter: WordFingerprinter,
    /// Inverted index for similarity search.
    inverted_index: Option<InvertedIndex>,
    /// Grid dimension.
    dimension: u32,
    /// Configuration.
    config: FingerprintConfig,
}

impl Retina {
    /// Creates a new retina from word fingerprints.
    pub fn new(
        fingerprints: HashMap<String, WordFingerprint>,
        dimension: u32,
        config: FingerprintConfig,
    ) -> Self {
        let grid_size = dimension * dimension;
        let mut word_fingerprinter = WordFingerprinter::new(config.clone(), grid_size);

        for (_, fp) in fingerprints {
            word_fingerprinter.insert(fp);
        }

        Self {
            word_fingerprinter,
            inverted_index: None,
            dimension,
            config,
        }
    }

    /// Creates a retina with an inverted index for fast similarity search.
    pub fn with_index(
        fingerprints: HashMap<String, WordFingerprint>,
        dimension: u32,
        config: FingerprintConfig,
    ) -> Self {
        let grid_size = dimension * dimension;
        let mut word_fingerprinter = WordFingerprinter::new(config.clone(), grid_size);

        for (_, fp) in &fingerprints {
            word_fingerprinter.insert(fp.clone());
        }

        // Build inverted index
        let inverted_index = InvertedIndex::from_fingerprints(&fingerprints, grid_size);

        Self {
            word_fingerprinter,
            inverted_index: Some(inverted_index),
            dimension,
            config,
        }
    }

    /// Loads a retina from a binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (header, fingerprints) = RetinaFormat::read(path)?;

        Ok(Self::with_index(
            fingerprints,
            header.dimension,
            FingerprintConfig::default(),
        ))
    }

    /// Saves the retina to a binary file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let fingerprints = self.word_fingerprinter.iter().map(|(w, wf)| (w.clone(), wf.clone())).collect();
        RetinaFormat::write(path, &fingerprints, self.dimension)
    }

    /// Returns the word fingerprinter.
    pub fn word_fingerprinter(&self) -> &WordFingerprinter {
        &self.word_fingerprinter
    }

    /// Returns the grid dimension.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Returns the total grid size.
    pub fn grid_size(&self) -> u32 {
        self.dimension * self.dimension
    }

    /// Returns the fingerprint configuration.
    pub fn config(&self) -> &FingerprintConfig {
        &self.config
    }

    /// Returns the number of words.
    pub fn vocabulary_size(&self) -> usize {
        self.word_fingerprinter.len()
    }

    /// Checks if a word exists in the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.word_fingerprinter.contains(word)
    }

    /// Gets a word's fingerprint.
    pub fn get_word_fingerprint(&self, word: &str) -> Option<&Sdr> {
        self.word_fingerprinter.get(word).map(|wf| &wf.fingerprint)
    }

    /// Generates a fingerprint for text.
    pub fn fingerprint_text(&self, text: &str) -> Sdr {
        let tf = TextFingerprinter::new(&self.word_fingerprinter, self.config.clone());
        tf.fingerprint(text)
    }

    /// Computes similarity between two words.
    pub fn word_similarity(&self, word1: &str, word2: &str) -> Result<f64> {
        self.word_fingerprinter.word_similarity(word1, word2)
    }

    /// Computes similarity between two texts.
    pub fn text_similarity(&self, text1: &str, text2: &str) -> f64 {
        let fp1 = self.fingerprint_text(text1);
        let fp2 = self.fingerprint_text(text2);
        fp1.cosine_similarity(&fp2)
    }

    /// Finds words similar to a given word.
    pub fn find_similar_words(&self, word: &str, k: usize) -> Result<Vec<(String, f64)>> {
        self.word_fingerprinter.find_similar(word, k)
    }

    /// Finds words similar to a given fingerprint.
    pub fn find_similar_to_fingerprint(&self, fp: &Sdr, k: usize) -> Vec<(String, f64)> {
        self.word_fingerprinter.find_similar_to_fingerprint(fp, k)
    }

    /// Finds words similar using the inverted index (faster for large vocabularies).
    pub fn find_similar_indexed(&self, fp: &Sdr, k: usize) -> Result<Vec<(String, f64)>> {
        let index = self
            .inverted_index
            .as_ref()
            .ok_or_else(|| ProteusError::Fingerprint("Inverted index not built".to_string()))?;

        let similarity = CosineSimilarity;
        let candidates = index.search(fp, k * 2);

        let mut results: Vec<(String, f64)> = candidates
            .iter()
            .filter_map(|word| {
                self.word_fingerprinter.get(word).map(|wf| {
                    let sim = similarity.similarity(&fp, &wf.fingerprint);
                    (word.clone(), sim)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Computes the semantic "center" of multiple words.
    ///
    /// Returns a fingerprint representing the common semantic ground.
    pub fn semantic_center(&self, words: &[&str]) -> Sdr {
        let grid_size = self.grid_size();
        let mut position_counts: HashMap<u32, usize> = HashMap::new();

        for word in words {
            if let Some(wf) = self.word_fingerprinter.get(*word) {
                for pos in wf.fingerprint.iter() {
                    *position_counts.entry(pos).or_insert(0) += 1;
                }
            }
        }

        // Keep positions that appear in at least half the words
        let threshold = words.len() / 2;
        let positions: Vec<u32> = position_counts
            .into_iter()
            .filter(|(_, count)| *count > threshold)
            .map(|(pos, _)| pos)
            .collect();

        Sdr::from_positions(&positions, grid_size)
    }

    /// Computes the semantic difference between two concepts.
    ///
    /// Returns positions unique to the first concept.
    pub fn semantic_difference(&self, word1: &str, word2: &str) -> Option<Sdr> {
        let fp1 = self.word_fingerprinter.get(word1)?;
        let fp2 = self.word_fingerprinter.get(word2)?;

        // XOR to find differences, then AND with fp1 to get positions unique to fp1
        let diff = &fp1.fingerprint ^ &fp2.fingerprint;
        Some(&diff & &fp1.fingerprint)
    }

    /// Performs analogy: A is to B as C is to ?
    ///
    /// Returns words that are semantically related to C in the same way B is to A.
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let fp_a = self
            .word_fingerprinter
            .get(a)
            .ok_or_else(|| ProteusError::WordNotFound(a.to_string()))?;
        let fp_b = self
            .word_fingerprinter
            .get(b)
            .ok_or_else(|| ProteusError::WordNotFound(b.to_string()))?;
        let fp_c = self
            .word_fingerprinter
            .get(c)
            .ok_or_else(|| ProteusError::WordNotFound(c.to_string()))?;

        // Compute the relationship vector: B - A
        let relationship = &fp_b.fingerprint ^ &fp_a.fingerprint;

        // Apply relationship to C: C + relationship
        let target = &fp_c.fingerprint | &relationship;

        // Sparsify target
        let mut target = target;
        target.sparsify(self.config.max_active_bits);

        // Find most similar words, excluding a, b, c
        let results = self.find_similar_to_fingerprint(&target, k + 3);

        let filtered: Vec<(String, f64)> = results
            .into_iter()
            .filter(|(w, _)| w != a && w != b && w != c)
            .take(k)
            .collect();

        Ok(filtered)
    }

    /// Builds the inverted index for fast similarity search.
    pub fn build_index(&mut self) {
        let fingerprints: HashMap<String, WordFingerprint> = self
            .word_fingerprinter
            .iter()
            .map(|(w, wf)| (w.clone(), wf.clone()))
            .collect();

        self.inverted_index = Some(InvertedIndex::from_fingerprints(&fingerprints, self.grid_size()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_retina() -> Retina {
        let mut fps = HashMap::new();

        // Create fingerprints with overlapping positions
        fps.insert(
            "king".to_string(),
            WordFingerprint::new("king".to_string(), &[1, 2, 3, 4, 100, 200, 300], 16384),
        );
        fps.insert(
            "queen".to_string(),
            WordFingerprint::new("queen".to_string(), &[1, 2, 3, 5, 100, 201, 301], 16384),
        );
        fps.insert(
            "man".to_string(),
            WordFingerprint::new("man".to_string(), &[1, 2, 4, 6, 400, 500, 600], 16384),
        );
        fps.insert(
            "woman".to_string(),
            WordFingerprint::new("woman".to_string(), &[1, 2, 5, 7, 400, 501, 601], 16384),
        );
        fps.insert(
            "dog".to_string(),
            WordFingerprint::new("dog".to_string(), &[10, 20, 30, 40, 50], 16384),
        );

        Retina::with_index(fps, 128, FingerprintConfig::default())
    }

    #[test]
    fn test_retina_creation() {
        let retina = create_test_retina();
        assert_eq!(retina.vocabulary_size(), 5);
        assert!(retina.contains("king"));
        assert!(!retina.contains("prince"));
    }

    #[test]
    fn test_word_fingerprint() {
        let retina = create_test_retina();
        let fp = retina.get_word_fingerprint("king").unwrap();
        assert!(fp.contains(1));
        assert!(fp.contains(100));
    }

    #[test]
    fn test_word_similarity() {
        let retina = create_test_retina();

        let sim_king_queen = retina.word_similarity("king", "queen").unwrap();
        let sim_king_dog = retina.word_similarity("king", "dog").unwrap();

        // King and queen should be more similar than king and dog
        assert!(sim_king_queen > sim_king_dog);
    }

    #[test]
    fn test_find_similar() {
        let retina = create_test_retina();
        let similar = retina.find_similar_words("king", 2).unwrap();

        assert!(!similar.is_empty());
        // Queen should be most similar to king
        assert_eq!(similar[0].0, "queen");
    }

    #[test]
    fn test_semantic_center() {
        let retina = create_test_retina();
        let center = retina.semantic_center(&["king", "queen"]);

        // Should contain positions common to both
        assert!(center.contains(1));
        assert!(center.contains(2));
        assert!(center.contains(3));
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.retina");

        let retina = create_test_retina();
        retina.save(&path).unwrap();

        let loaded = Retina::load(&path).unwrap();
        assert_eq!(loaded.vocabulary_size(), 5);
        assert!(loaded.contains("king"));
    }
}
