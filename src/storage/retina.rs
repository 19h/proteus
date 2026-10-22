//! Retina management - the core semantic fingerprint database.

use crate::config::FingerprintConfig;
use crate::config::TextConfig;
use crate::error::{ProteusError, Result};
use crate::fingerprint::{Sdr, TextFingerprinter, WordFingerprint, WordFingerprinter};
use crate::index::InvertedIndex;
use crate::similarity::{CosineSimilarity, SimilarityMeasure};
use crate::storage::format::RetinaFormat;
use crate::training::TrainingReport;
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
    text_config: TextConfig,
    training_report: Option<TrainingReport>,
    angular_report: Option<crate::training::angular::AngularReport>,
    semantic_model: Option<crate::semantic::SemanticModel>,
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
            text_config: TextConfig::default(),
            training_report: None,
            angular_report: None,
            semantic_model: None,
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

        for fp in fingerprints.values() {
            word_fingerprinter.insert(fp.clone());
        }

        // Build inverted index
        let inverted_index = InvertedIndex::from_fingerprints(&fingerprints, grid_size);

        Self {
            word_fingerprinter,
            inverted_index: Some(inverted_index),
            dimension,
            config,
            text_config: TextConfig::default(),
            training_report: None,
            angular_report: None,
            semantic_model: None,
        }
    }

    /// Loads a retina from a binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (header, fingerprints, metadata, index) = RetinaFormat::read_full(path)?;
        let mut retina = Self::new(fingerprints, header.dimension, metadata.fingerprint_config);
        retina.text_config = metadata.text_config;
        retina.training_report = metadata.training_report;
        retina.angular_report = metadata.angular_report;
        retina.semantic_model = metadata.semantic_model;
        retina.inverted_index = index;
        if retina.inverted_index.is_none() {
            retina.build_index();
        }
        Ok(retina)
    }

    /// Saves the retina to a binary file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let fingerprints = self
            .word_fingerprinter
            .iter()
            .map(|(w, wf)| (w.clone(), wf.clone()))
            .collect();
        RetinaFormat::write_with_metadata_and_index(
            path,
            &fingerprints,
            self.dimension,
            crate::storage::format::RetinaMetadata {
                fingerprint_config: self.config.clone(),
                text_config: self.text_config.clone(),
                training_report: self.training_report.clone(),
                angular_report: self.angular_report.clone(),
                semantic_model: self.semantic_model.clone(),
                word_metadata: Default::default(),
            },
            self.inverted_index.as_ref(),
        )
    }

    /// Saved training parameters and diagnostics. Absent for legacy files.
    pub fn training_report(&self) -> Option<&TrainingReport> {
        self.training_report.as_ref()
    }

    /// Attach the resolved training parameters and use their tokenizer at inference.
    pub fn set_training_report(&mut self, report: TrainingReport) {
        self.text_config = report.config.text.clone();
        self.training_report = Some(report);
        self.angular_report = None;
        self.semantic_model = None;
    }

    /// Provenance for a non-spatial angular retina, when present.
    pub fn angular_report(&self) -> Option<&crate::training::angular::AngularReport> {
        self.angular_report.as_ref()
    }

    /// Attach angular-training provenance and its inference normalization.
    pub fn set_angular_report(&mut self, report: crate::training::angular::AngularReport) {
        self.text_config = report.config.text.clone();
        self.angular_report = Some(report);
        self.training_report = None;
        self.semantic_model = None;
    }

    /// Bound senses, typed relations and attributed event evidence, if present.
    pub fn semantic_model(&self) -> Option<&crate::semantic::SemanticModel> {
        self.semantic_model.as_ref()
    }

    /// Attach validated semantics. Version 5 prevents older readers from silently
    /// replacing relation scores with bitmap cosine.
    pub fn set_semantic_model(&mut self, model: crate::semantic::SemanticModel) -> Result<()> {
        model.validate()?;
        if model.report.config.dimension != self.dimension as usize
            || model.report.config.fingerprint != self.config
            || model.entries.len() != self.word_fingerprinter.len()
            || model
                .entries
                .keys()
                .any(|word| !self.word_fingerprinter.contains(word))
        {
            return Err(ProteusError::Config(
                "Semantic model and fingerprints differ".into(),
            ));
        }
        self.text_config = model.report.config.text.clone();
        self.training_report = None;
        self.angular_report = None;
        self.semantic_model = Some(model);
        Ok(())
    }

    /// Whether neighboring bit positions represent learned spatial neighbors.
    /// Legacy models retain their original spatial interpretation.
    pub fn has_spatial_topology(&self) -> bool {
        self.angular_report.is_none() && self.semantic_model.is_none()
    }

    fn normalize_word(&self, word: &str) -> Option<String> {
        if let Some(model) = &self.semantic_model {
            return model.normalize(word);
        }
        let tokens = crate::Tokenizer::new(self.text_config.clone()).tokenize_to_strings(word);
        if tokens.len() == 1 {
            tokens.into_iter().next()
        } else {
            None
        }
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
        self.normalize_word(word)
            .is_some_and(|word| self.word_fingerprinter.contains(&word))
    }

    /// Gets a word's fingerprint.
    pub fn get_word_fingerprint(&self, word: &str) -> Option<&Sdr> {
        self.word_fingerprinter
            .get(&self.normalize_word(word)?)
            .map(|wf| &wf.fingerprint)
    }

    /// Generates a fingerprint for text.
    pub fn fingerprint_text(&self, text: &str) -> Sdr {
        let tf = TextFingerprinter::with_tokenizer(
            &self.word_fingerprinter,
            self.config.clone(),
            crate::Tokenizer::new(self.text_config.clone()),
        );
        tf.fingerprint(text)
    }

    /// Computes similarity between two words.
    pub fn word_similarity(&self, word1: &str, word2: &str) -> Result<f64> {
        if let Some(model) = &self.semantic_model {
            return model
                .similarity(word1, word2, crate::semantic::Relation::Similarity)?
                .ok_or_else(|| {
                    ProteusError::WordNotFound(format!(
                        "{word1} / {word2} (lexical evidence unavailable)"
                    ))
                });
        }
        let word1 = self
            .normalize_word(word1)
            .ok_or_else(|| ProteusError::WordNotFound(word1.into()))?;
        let word2 = self
            .normalize_word(word2)
            .ok_or_else(|| ProteusError::WordNotFound(word2.into()))?;
        self.word_fingerprinter.word_similarity(&word1, &word2)
    }

    /// Computes similarity between two texts.
    pub fn text_similarity(&self, text1: &str, text2: &str) -> f64 {
        let fp1 = self.fingerprint_text(text1);
        let fp2 = self.fingerprint_text(text2);
        fp1.cosine_similarity(&fp2)
    }

    /// Finds words similar to a given word.
    pub fn find_similar_words(&self, word: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if let Some(model) = &self.semantic_model {
            return Ok(model
                .find_similar(
                    crate::semantic::WordQuery::new(word),
                    crate::semantic::Relation::Similarity,
                    k,
                )?
                .into_iter()
                .map(|result| (result.word, result.score))
                .collect());
        }
        let normalized = self
            .normalize_word(word)
            .ok_or_else(|| ProteusError::WordNotFound(word.into()))?;
        let target = self
            .get_word_fingerprint(&normalized)
            .ok_or_else(|| ProteusError::WordNotFound(word.into()))?;
        if self.inverted_index.is_some() {
            let mut results = self.find_similar_indexed(target, k.saturating_add(1))?;
            results.retain(|(w, _)| w != &normalized);
            results.truncate(k);
            Ok(results)
        } else {
            self.word_fingerprinter.find_similar(&normalized, k)
        }
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
        // Every positive-cosine result must share a bit. Raw-overlap top-2k
        // pruning is unsound when fingerprint cardinalities differ.
        let candidates = index.search(fp, usize::MAX);

        let mut results: Vec<(String, f64)> = candidates
            .iter()
            .filter_map(|word| {
                self.word_fingerprinter.get(word).map(|wf| {
                    let sim = similarity.similarity(fp, &wf.fingerprint);
                    (word.clone(), sim)
                })
            })
            .collect();

        // Include zero-overlap candidates so this API exactly matches exhaustive
        // ranking, including k larger than the number of positive matches.
        if results.len() < k {
            for (word, wf) in self.word_fingerprinter.iter() {
                if wf.fingerprint.overlap_count(fp) == 0 {
                    results.push((word.clone(), 0.0));
                }
            }
        }
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
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
            if let Some(fp) = self.get_word_fingerprint(word) {
                for pos in fp.iter() {
                    *position_counts.entry(pos).or_insert(0) += 1;
                }
            }
        }

        // Keep positions supported by a strict majority of the supplied words.
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
        let fp1 = self.get_word_fingerprint(word1)?;
        let fp2 = self.get_word_fingerprint(word2)?;

        // XOR to find differences, then AND with fp1 to get positions unique to fp1
        let diff = fp1 ^ fp2;
        Some(&diff & fp1)
    }

    /// Performs analogy: A is to B as C is to ?
    ///
    /// Returns words that are semantically related to C in the same way B is to A.
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Result<Vec<(String, f64)>> {
        let a = self
            .normalize_word(a)
            .ok_or_else(|| ProteusError::WordNotFound(a.into()))?;
        let b = self
            .normalize_word(b)
            .ok_or_else(|| ProteusError::WordNotFound(b.into()))?;
        let c = self
            .normalize_word(c)
            .ok_or_else(|| ProteusError::WordNotFound(c.into()))?;
        let fp_a = self
            .word_fingerprinter
            .get(&a)
            .ok_or_else(|| ProteusError::WordNotFound(a.to_string()))?;
        let fp_b = self
            .word_fingerprinter
            .get(&b)
            .ok_or_else(|| ProteusError::WordNotFound(b.to_string()))?;
        let fp_c = self
            .word_fingerprinter
            .get(&c)
            .ok_or_else(|| ProteusError::WordNotFound(c.to_string()))?;

        // Signed binary-vector arithmetic: cosine(B + C - A, candidate).
        // XOR loses the sign of a removed feature and is not subtraction.
        let mut target = HashMap::<u32, f64>::new();
        for p in fp_b.fingerprint.iter().chain(fp_c.fingerprint.iter()) {
            *target.entry(p).or_default() += 1.0;
        }
        for p in fp_a.fingerprint.iter() {
            *target.entry(p).or_default() -= 1.0;
        }
        let norm = target.values().map(|x| x * x).sum::<f64>().sqrt();
        let mut results: Vec<_> = self
            .word_fingerprinter
            .iter()
            .filter(|(w, _)| *w != &a && *w != &b && *w != &c)
            .map(|(word, wf)| {
                let denominator = norm * (wf.cardinality() as f64).sqrt();
                let score = if denominator == 0.0 {
                    0.0
                } else {
                    wf.fingerprint
                        .iter()
                        .map(|p| target.get(&p).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        / denominator
                };
                (word.clone(), score)
            })
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        results.truncate(k);
        Ok(results)
    }

    /// Builds the inverted index for fast similarity search.
    pub fn build_index(&mut self) {
        let fingerprints: HashMap<String, WordFingerprint> = self
            .word_fingerprinter
            .iter()
            .map(|(w, wf)| (w.clone(), wf.clone()))
            .collect();

        self.inverted_index = Some(InvertedIndex::from_fingerprints(
            &fingerprints,
            self.grid_size(),
        ));
    }
    /// Returns words that have a specific grid position active in their fingerprint.
    ///
    /// Requires the inverted index to be built.
    pub fn words_at_position(&self, position: u32) -> Result<Vec<String>> {
        let index = self.inverted_index.as_ref().ok_or_else(|| {
            ProteusError::Fingerprint(
                "Inverted index not built. Run 'proteus migrate' first.".to_string(),
            )
        })?;

        Ok(index.words_at_position(position))
    }

    /// Returns words at multiple positions, along with how many of those positions each word covers.
    ///
    /// Returns (word, overlap_count) pairs sorted by overlap count (descending).
    pub fn words_at_positions(&self, positions: &[u32], k: usize) -> Result<Vec<(String, u32)>> {
        let index = self.inverted_index.as_ref().ok_or_else(|| {
            ProteusError::Fingerprint(
                "Inverted index not built. Run 'proteus migrate' first.".to_string(),
            )
        })?;

        // Count how many of the query positions each word has
        let mut word_scores: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();

        for &pos in positions {
            for word in index.words_at_position(pos) {
                *word_scores.entry(word).or_insert(0) += 1;
            }
        }

        // Sort by score descending
        let mut results: Vec<(String, u32)> = word_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        results.truncate(k);

        Ok(results)
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
